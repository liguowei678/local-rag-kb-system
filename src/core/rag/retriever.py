"""
检索器
负责从向量数据库中检索相关文档
"""

import time
from typing import List, Dict, Any, Optional

from src.core.config import settings
from src.core.logging import get_logger
from src.core.embedding import get_embedding_service
from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
from .types import RAGQuery, RetrievedDocument

logger = get_logger(__name__)


class Retriever:
    """检索器"""
    
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.embedding_service = get_embedding_service()
        self.vector_manager = real_qdrant_manager
        
        logger.info("检索器初始化",
                   collection_name=collection_name)
    
    async def retrieve(self, query: RAGQuery) -> List[RetrievedDocument]:
        """检索相关文档"""
        start_time = time.time()
        
        try:
            # 1. 查询文本嵌入
            embedding_result = await self.embedding_service.embed_text(query.query)
            if not embedding_result:
                logger.error("查询文本嵌入失败", query=query.query[:100])
                return []
            
            query_vector = embedding_result.vector
            
            # 2. 判断是否使用动态阈值
            # 使用 score_threshold=0.0 触发动态阈值（0.0是有效值但通常不会使用）
            use_dynamic_threshold = query.score_threshold < 0.1  # 0.0或接近0的值触发动态阈值
            
            logger.debug("阈值模式判断",
                       query=query.query[:50],
                       input_threshold=query.score_threshold,
                       use_dynamic=use_dynamic_threshold)
            
            if use_dynamic_threshold:
                # 动态阈值模式：获取更多结果，本地计算阈值并过滤
                # 获取足够多的结果用于分数分布分析
                search_limit = max(query.top_k * 3, 30)
                qdrant_threshold = 0.0  # Qdrant端不设阈值
            else:
                # 固定阈值模式：使用原有限制和阈值
                search_limit = query.top_k
                qdrant_threshold = query.score_threshold
            
            # 3. 向量搜索
            search_results = await self.vector_manager.search(
                collection_name=query.collection_name,
                query_vector=query_vector,
                limit=search_limit,
                score_threshold=qdrant_threshold,
                with_payload=True
            )
            
            # 4. 处理动态阈值过滤
            if use_dynamic_threshold and search_results:
                # 提取所有结果的分数
                all_scores = [result.get("score", 0.0) for result in search_results]
                
                # 调试：记录前几个分数
                if all_scores:
                    logger.debug("原始搜索分数样本",
                               scores_sample=all_scores[:3])
                
                # 计算动态阈值（使用0.3作为基准）
                dynamic_threshold = self._calculate_dynamic_threshold(all_scores, 0.3)
                
                # 调试：打印分数分布
                logger.debug("动态阈值调试",
                           all_scores_count=len(all_scores),
                           all_scores_min=min(all_scores) if all_scores else 0,
                           all_scores_max=max(all_scores) if all_scores else 0,
                           all_scores_avg=sum(all_scores)/len(all_scores) if all_scores else 0,
                           calculated_threshold=dynamic_threshold)
                
                # 应用动态阈值过滤
                filtered_results = []
                filtered_count = 0
                for result in search_results:
                    score = result.get("score", 0.0)
                    if score >= dynamic_threshold:
                        filtered_results.append(result)
                    else:
                        filtered_count += 1
                        logger.debug("文档被动态阈值过滤",
                                   doc_id=result.get("id", "未知"),
                                   score=score,
                                   threshold=dynamic_threshold)
                
                logger.info("动态阈值过滤结果",
                           total_docs=len(search_results),
                           filtered_out=filtered_count,
                           remaining=len(filtered_results),
                           dynamic_threshold=dynamic_threshold)
                
                # 限制返回数量
                search_results = filtered_results[:query.top_k]
                
                logger.info("动态阈值应用",
                           query=query.query[:50],
                           all_results=len(all_scores),
                           avg_score=sum(all_scores)/len(all_scores) if all_scores else 0,
                           dynamic_threshold=dynamic_threshold,
                           filtered_results=len(search_results))
                
                actual_threshold = dynamic_threshold
                
                # 在第一个文档的metadata中记录动态阈值信息
                if search_results:
                    first_result = search_results[0]
                    if 'payload' in first_result:
                        if 'metadata' not in first_result['payload']:
                            first_result['payload']['metadata'] = {}
                        first_result['payload']['metadata']['dynamic_threshold_applied'] = True
                        first_result['payload']['metadata']['dynamic_threshold_value'] = dynamic_threshold
                        first_result['payload']['metadata']['original_score_threshold'] = query.score_threshold
            else:
                actual_threshold = query.score_threshold
            
            # 3. 格式化结果
            documents = []
            for i, result in enumerate(search_results):
                payload = result.get("payload", {})
                
                # 提取文本和元数据
                text = payload.get("text", "")
                metadata = payload.get("metadata", {})
                
                # 调试：打印payload结构
                if i < 3:  # 只打印前3个
                    logger.debug("检索结果payload", 
                                result_id=result.get("id", ""),
                                score=result.get("score", 0.0),
                                has_text=bool(text),
                                text_length=len(text),
                                payload_keys=list(payload.keys())[:5])
                
                if text:  # 只包含有文本的结果
                    doc = RetrievedDocument(
                        id=str(result.get("id", "")),
                        text=text,
                        score=result.get("score", 0.0),
                        metadata=metadata,
                        source=metadata.get("source", metadata.get("file_name", "未知"))
                    )
                    documents.append(doc)
            
            retrieval_time = time.time() - start_time
            
            logger.info("文档检索完成",
                       query=query.query[:100],
                       collection_name=query.collection_name,
                       use_dynamic_threshold=use_dynamic_threshold,
                       final_threshold=actual_threshold,
                       documents_count=len(documents),
                       retrieval_time=f"{retrieval_time:.3f}s")
            
            return documents
            
        except Exception as e:
            logger.error("文档检索失败",
                        query=query.query[:100],
                        error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查嵌入服务
            embedding_health = await self.embedding_service.health_check()
            
            # 检查向量数据库
            collection_info = await self.vector_manager.get_collection_info(self.collection_name)
            
            return {
                "status": "healthy",
                "embedding": embedding_health,
                "vector_db": {
                    "collection": self.collection_name,
                    "exists": collection_info.get("exists", False),
                    "points_count": collection_info.get("points_count", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "collection": self.collection_name
            }
    
    def _calculate_dynamic_threshold(self, pre_scores: List[float], 
                                   base_threshold: float = 0.3) -> float:
        """
        基于预检索分数分布的最简动态阈值
        
        核心逻辑（适配0.3基准阈值）：
        - 结果质量高（平均分>0.6）：提高阈值
        - 结果质量中等（0.4-0.6）：保持原阈值
        - 结果质量低（平均分<0.4）：降低阈值
        """
        if not pre_scores:
            logger.debug("动态阈值计算: 无预检索分数，返回0.2")
            return 0.2  # 无结果时降低阈值
        
        avg_score = sum(pre_scores) / len(pre_scores)
        
        logger.debug("动态阈值计算",
                   pre_scores_count=len(pre_scores),
                   pre_scores_avg=avg_score,
                   base_threshold=base_threshold)
        
        # 核心逻辑：适配0.3基准阈值
        if avg_score > 0.6:
            result = min(0.7, base_threshold + 0.2)
            logger.debug(f"动态阈值计算: 高质量(>{0.6}) → 提高阈值到{result}")
            return result
        elif avg_score > 0.4:
            logger.debug(f"动态阈值计算: 中等质量({0.4}-{0.6}) → 保持阈值{base_threshold}")
            return base_threshold
        else:
            result = max(0.1, base_threshold - 0.2)
            logger.debug(f"动态阈值计算: 低质量(<{0.4}) → 降低阈值到{result}")
            return result


# 全局检索器实例
_retriever_instance = None

def get_retriever(collection_name: str = "default") -> Retriever:
    """获取检索器实例（单例模式）"""
    global _retriever_instance
    
    if _retriever_instance is None or _retriever_instance.collection_name != collection_name:
        _retriever_instance = Retriever(collection_name)
    
    return _retriever_instance