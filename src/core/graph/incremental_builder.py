import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.config import Settings as Config
from .client import Neo4jClient
from .stable_builder import StableKnowledgeGraphBuilder
from .config import GraphConfig

logger = logging.getLogger(__name__)

class IncrementalConfig:
    """增量构建配置"""
    def __init__(self, 
                 enable_content_hash: bool = True,
                 enable_timestamp_check: bool = True,
                 merge_strategy: str = "merge",
                 max_retention_days: int = 30,
                 batch_size: int = 10):
        self.enable_content_hash = enable_content_hash
        self.enable_timestamp_check = enable_timestamp_check
        self.merge_strategy = merge_strategy
        self.max_retention_days = max_retention_days
        self.batch_size = batch_size

class IncrementalKnowledgeGraphBuilder:
    """增量知识图谱构建器"""
    
    def __init__(self, config: Config, incremental_config: Optional[IncrementalConfig] = None):
        self.config = config
        self.incremental_config = incremental_config or IncrementalConfig()
        
        # 创建GraphConfig对象（从环境变量读取）
        import os
        graph_config = GraphConfig(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "graphrag123"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            # DeepSeek配置
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", config.deepseek_api_key if hasattr(config, 'deepseek_api_key') else ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            # GraphRAG功能开关
            graphrag_enabled=os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"
        )
        
        self.neo4j_client = Neo4jClient(graph_config)
        self.base_builder = StableKnowledgeGraphBuilder(graph_config)
        
        logger.info(f"增量构建器初始化完成，配置: {self.incremental_config.__dict__}")
    
    async def detect_document_changes(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测文档变化，返回需要处理的文档"""
        changed_docs = []
        skipped_info = []  # 记录跳过的文档信息
        
        for doc in documents:
            doc_id = doc.get('document_id')
            content = doc.get('content', '')
            
            if not doc_id or not content:
                logger.warning(f"❌ 文档缺少ID或内容: {doc_id}")
                skipped_info.append({'id': doc_id, 'reason': '缺少ID或内容'})
                continue
            
            # 计算内容哈希（如果文档中已提供哈希，使用它确保一致性）
            import hashlib
            
            # 如果文档中已有哈希，使用它（确保与上传时计算的哈希一致）
            if "content_hash" in doc:
                content_hash = doc["content_hash"]
                logger.debug(f"📝 使用文档提供的哈希: {content_hash}")
            else:
                # 否则计算哈希
                logger.debug(f"📝 计算哈希 - 文档ID: {doc_id}")
                logger.debug(f"📝 内容长度: {len(content)} 字符")
                if content:
                    logger.debug(f"📝 内容预览: {content[:200]}...")
                
                content_hash = hashlib.md5(content.encode()).hexdigest()
                logger.debug(f"📝 计算出的哈希: {content_hash}")
            
            # 检查文档是否已存在且未变化
            try:
                existing_doc = self.neo4j_client.get_document(doc_id)
                
                if existing_doc:
                    # 方案A：文档ID存在就视为需要更新（完全替换）
                    # 不比较哈希，因为用户通过文件名控制意图
                    # 相同文件名=替换，不同文件名=新增
                    logger.info(f"📄 文档已存在，准备更新: {doc_id}")
                    # 不跳过，继续处理（将在处理阶段删除旧数据）
                else:
                    logger.info(f"🆕 新文档: {doc_id} (哈希: {content_hash[:8]}...)")
                    
            except Exception as e:
                logger.error(f"❌ 检查文档失败 {doc_id}: {e}")
                skipped_info.append({'id': doc_id, 'reason': f'检查失败: {str(e)[:50]}'})
                # 出错时也处理文档，避免丢失数据
                logger.warning(f"⚠️ 检查失败，但仍处理文档: {doc_id}")
            
            # 添加哈希到文档元数据
            doc['content_hash'] = content_hash
            doc['processed_at'] = datetime.now().isoformat()
            changed_docs.append(doc)
        
        # 详细记录检测结果
        logger.info(f"📊 检测结果: {len(changed_docs)}个需要处理, {len(skipped_info)}个跳过")
        
        if skipped_info:
            for skipped in skipped_info:
                logger.debug(f"跳过的文档: ID={skipped['id']}, 原因={skipped['reason']}")
        
        # 返回changed_docs和skipped_info（通过doc的额外字段）
        for doc in changed_docs:
            doc['_skipped_info'] = skipped_info
        
        return changed_docs
    
    async def incremental_build(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
        """增量构建知识图谱"""
        try:
            logger.info(f"开始增量构建，文档数: {len(documents)}")
            
            # 检测变化
            changed_docs = await self.detect_document_changes(documents)
            
            if not changed_docs:
                # 从第一个文档获取跳过的信息（如果有）
                skipped_info = []
                if documents and '_skipped_info' in documents[0]:
                    skipped_info = documents[0].get('_skipped_info', [])
                
                logger.info(f"没有文档需要更新，跳过了 {len(skipped_info)} 个文档")
                stats = self.neo4j_client.get_statistics()
                return {
                    "success": True,
                    "message": f"没有文档需要更新（跳过了 {len(skipped_info)} 个文档）",
                    "statistics": stats,
                    "skipped_documents": skipped_info
                }
            
            total_entities = 0
            total_relations = 0
            total_processed = 0
            
            # 处理每个变化的文档
            for doc in changed_docs:
                doc_id = doc.get('document_id')
                content = doc.get('content', '')
                
                # 调试日志：检查文档字典内容
                logger.info(f"增量构建器处理的文档字典: keys={list(doc.keys())}")
                logger.info(f"增量构建器获取的doc_id: {doc_id}")
                logger.info(f"文档中document_id字段: {doc.get('document_id')}")
                logger.info(f"文档中id字段: {doc.get('id')}")
                
                if not content:
                    logger.warning(f"文档内容为空: {doc_id}")
                    continue
                
                try:
                    # 检查文档是否已存在
                    existing_doc = self.neo4j_client.get_document(doc_id)
                    current_hash = doc.get('content_hash', '')
                    
                    if existing_doc:
                        # 方案A：文档ID存在就删除旧数据（完全替换）
                        logger.info(f"🔄 文档已存在，删除旧数据准备更新: {doc_id}")
                        # 删除文档及其所有关系
                        self.neo4j_client.delete_document_and_relations(doc_id)
                        
                        # 同时删除Qdrant中的旧向量点
                        try:
                            from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
                            await real_qdrant_manager.delete_document_points("default", doc_id)
                            logger.info(f"✅ 已删除Qdrant中的旧向量点: {doc_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ 删除Qdrant向量点失败: {e}")
                            # 继续处理，不中断
                    
                    # 创建或更新文档节点
                    self.neo4j_client.create_document_node(
                        document_id=doc_id,
                        metadata={
                            'processed_at': doc.get('processed_at', ''),
                            'source': 'incremental_build'
                        },
                        content=content[:500],  # 只存储前500字符
                        content_hash=doc.get('content_hash', '')
                    )
                    
                    # 3. 使用基础构建器处理文档（带异常处理）
                    try:
                        # 检查base_builder是否存在
                        if self.base_builder is None:
                            logger.error("❌ 基础构建器为None，无法处理文档")
                            result = None
                        else:
                            result = await self.base_builder.build_from_documents([doc])
                            logger.info(f"基础构建器返回结果: {result is not None}")
                            if result is not None:
                                logger.debug(f"基础构建器返回详情: {result.get('message', '无消息')}")
                    except Exception as build_err:
                        logger.error(f"基础构建器失败: {build_err}")
                        import traceback
                        logger.error(f"详细堆栈: {traceback.format_exc()}")
                        result = None
                    
                    if result:
                        stats = result.get("statistics", {})
                        total_entities += stats.get("entities", {}).get("total", 0)
                        total_relations += stats.get("relations", {}).get("total", 0)
                        logger.info(f"构建统计: {stats}")
                    else:
                        logger.warning(f"基础构建器返回None，文档可能处理失败: {doc_id}")
                    
                    total_processed += 1
                    
                    logger.debug(f"文档更新完成: {doc_id}")
                    
                except Exception as e:
                    logger.error(f"更新文档失败 {doc.get('document_id', 'unknown')}: {e}")
            
            # 清理过期数据
            await self.cleanup_old_data()
            
            # 获取统计信息
            stats = self.neo4j_client.get_statistics()
            
            return {
                "success": True,
                "message": f"增量构建完成，处理了 {total_processed} 个文档",
                "statistics": {
                    "documents": {"processed": total_processed, "total": stats.get("documents", {}).get("total", 0)},
                    "entities": {"added": total_entities, "total": stats.get("entities", {}).get("total", 0)},
                    "relations": {"added": total_relations, "total": stats.get("relations", {}).get("total", 0)}
                }
            }
            
        except Exception as e:
            logger.error(f"增量构建失败: {e}")
            return {
                "success": False,
                "message": f"增量构建失败: {str(e)}",
                "statistics": {}
            }
    
    async def cleanup_old_data(self):
        """清理过期数据"""
        try:
            retention_days = self.incremental_config.max_retention_days
            
            # 清理过期文档（这里只是示例，实际需要根据业务逻辑实现）
            logger.info(f"清理超过 {retention_days} 天的过期数据")
            
            # 可以在这里添加清理逻辑
            # 例如：删除超过 retention_days 天未更新的文档
            
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
    
    async def close(self):
        """清理资源"""
        try:
            if hasattr(self.base_builder, 'close'):
                if asyncio.iscoroutinefunction(self.base_builder.close):
                    await self.base_builder.close()
                else:
                    self.base_builder.close()
            
            self.neo4j_client.close()
            logger.info("增量构建器资源已清理")
        except Exception as e:
            logger.error(f"关闭资源失败: {e}")
