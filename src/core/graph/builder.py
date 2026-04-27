"""
知识图谱构建器 - 使用neo4j-graphrag的SimpleKGPipeline
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

# 猴子补丁修复neo4j_graphrag的日志问题
import logging
original_log = logging.Logger._log

def patched_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1, **kwargs):
    # 过滤掉merge_strategy等未知参数
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['exc_info', 'extra', 'stack_info', 'stacklevel']}
    return original_log(self, level, msg, args, **filtered_kwargs)

logging.Logger._log = patched_log

# 完全禁用neo4j_graphrag日志
logging.getLogger('neo4j_graphrag').disabled = True

from .deepseek_llm import DeepSeekLLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from .config import GraphConfig
from .client import Neo4jClient

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 使用neo4j-graphrag的SimpleKGPipeline"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        
        # 初始化LLM（使用DeepSeek）
        # 使用自定义的DeepSeekLLM，禁用tools/functions支持
        self.llm = DeepSeekLLM(
            model_name=config.deepseek_model,
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            # OpenAI客户端配置（通过**kwargs传递）
            timeout=60.0,  # 增加超时时间
            max_retries=3   # 增加重试次数
        )
        
        # 初始化实体关系提取器
        self.extractor = LLMEntityRelationExtractor(llm=self.llm)
        
        # 初始化Neo4j客户端
        self.neo4j_client = Neo4jClient(config)
        
        # 初始化SimpleKGPipeline
        driver = self.neo4j_client.driver
        if not driver:
            logger.error("Neo4j driver not available")
            return
        
        # 初始化embedder（使用本地模型路径）
        embedder = SentenceTransformerEmbeddings(
            model="/app/huggingface_cache/bge-small-zh-v1.5"
        )
        
        # 优化SimpleKGPipeline配置，提高成功率
        self.kg_pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=driver,
            embedder=embedder,
            neo4j_database=config.neo4j_database,
            from_pdf=False  # 我们使用纯文本，不是PDF
        )
        
        logger.info("KnowledgeGraphBuilder initialized with SimpleKGPipeline")
    
    async def build_from_documents(self, documents: List[Dict[str, Any]], 
                                  batch_size: int = 10) -> Dict[str, Any]:
        """从文档构建知识图谱"""
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        processed_docs = 0
        total_entities = 0
        total_relations = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
            
            for doc in batch:
                try:
                    doc_id = doc.get("id", str(uuid.uuid4()))
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    filename = doc.get("filename", "")
                    
                    # 使用SimpleKGPipeline处理文档
                    # 如果是PDF文件，需要提供file_path参数
                    kwargs = {
                        "text": content,
                        "document_metadata": metadata
                    }
                    
                    # 如果文件名是PDF，尝试查找文件路径
                    if filename.lower().endswith('.pdf'):
                        # 尝试从上传目录查找文件
                        import os
                        upload_dir = "/app/data/uploads"
                        possible_path = os.path.join(upload_dir, doc_id, filename)
                        if os.path.exists(possible_path):
                            kwargs["file_path"] = possible_path
                        else:
                            # 如果没有找到文件，使用文本模式
                            logger.warning(f"PDF文件未找到: {possible_path}，使用文本模式")
                    
                    result = await self.kg_pipeline.run_async(**kwargs)
                    
                    # 统计结果
                    if result and hasattr(result, 'entities'):
                        entities_count = len(result.entities) if result.entities else 0
                        relations_count = len(result.relations) if result.relations else 0
                        
                        total_entities += entities_count
                        total_relations += relations_count
                        processed_docs += 1
                        
                        logger.debug(f"Document {doc_id}: {entities_count} entities, {relations_count} relations")
                    
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
        
        # 获取统计信息
        stats = self.neo4j_client.get_statistics()
        
        return {
            "total_documents": len(documents),
            "processed_documents": processed_docs,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "method": "SimpleKGPipeline",
            "statistics": stats,
            "config": {
                "model": self.config.deepseek_model,
                "entity_types": self.config.entity_types,
                "relation_types": self.config.relation_types
            }
        }
    
    async def build_from_directory(self, directory_path: str, 
                                  file_extensions: List[str] = None) -> Dict[str, Any]:
        """从目录构建知识图谱"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf', '.docx', '.json']
        
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        documents = []
        
        # 读取所有文件
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in file_extensions:
                try:
                    content = self._read_file(file_path)
                    
                    documents.append({
                        "id": str(file_path.relative_to(dir_path)),
                        "content": content,
                        "metadata": {
                            "path": str(file_path),
                            "filename": file_path.name,
                            "extension": file_path.suffix,
                            "size": file_path.stat().st_size
                        }
                    })
                    
                    logger.debug(f"Loaded document: {file_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
        
        # 构建知识图谱
        return await self.build_from_documents(documents)
    
    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        if file_path.suffix.lower() == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('content', '') or json.dumps(data, ensure_ascii=False)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return self.neo4j_client.get_statistics()
    
    def clear_graph(self) -> bool:
        """清空图谱"""
        try:
            with self.neo4j_client.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.warning("Knowledge graph cleared")
                return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    def close(self):
        """清理资源"""
        self.neo4j_client.close()