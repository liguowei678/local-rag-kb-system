"""
BM25异步管理器
负责BM25索引的异步构建、更新和检索
保持全异步架构，不阻塞事件循环
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BM25Config:
    """BM25配置"""
    use_jieba: bool = True
    remove_stopwords: bool = True
    k1: float = 1.5
    b: float = 0.75
    top_k_multiplier: int = 2  # 获取更多结果用于融合
    score_threshold: float = 0.3  # BM25分数阈值（与语义检索一致）
    cache_ttl_seconds: int = 300  # 索引缓存时间（秒）


class BM25AsyncManager:
    """BM25异步管理器"""
    
    def __init__(self, 
                 vector_manager,  # 向量数据库管理器
                 config: Optional[BM25Config] = None):
        """
        初始化BM25异步管理器
        
        Args:
            vector_manager: 向量数据库管理器（用于获取文档）
            config: BM25配置
        """
        self.vector_manager = vector_manager
        self.config = config or BM25Config()
        
        # BM25索引缓存
        self.bm25_retriever_cache = None
        self.cached_document_ids: Set[str] = set()
        self.last_cache_time: float = 0
        self.cache_lock = asyncio.Lock()
        
        # 线程池（用于执行同步的BM25操作）
        self.thread_pool = None
        
        logger.info("BM25异步管理器初始化",
                   use_jieba=self.config.use_jieba,
                   cache_ttl=self.config.cache_ttl_seconds)
    
    async def initialize(self):
        """异步初始化（可选）"""
        # 可以在这里预加载文档或执行其他初始化
        pass
    
    async def search(self, 
                    query: str, 
                    top_k: int = 10,
                    collection_name: str = "default") -> List[Dict[str, Any]]:
        """
        异步执行BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            collection_name: 集合名称
            
        Returns:
            BM25检索结果列表
        """
        start_time = time.time()
        
        try:
            # 1. 确保有有效的BM25索引
            retriever = await self._get_or_create_index_async(collection_name)
            if not retriever:
                logger.warning("BM25索引不可用，返回空结果")
                return []
            
            # 2. 在线程中执行同步的BM25搜索
            bm25_top_k = top_k * self.config.top_k_multiplier
            
            results = await asyncio.to_thread(
                retriever.search,
                query,
                top_k=bm25_top_k,
                score_threshold=self.config.score_threshold
            )
            
            search_time = time.time() - start_time
            
            logger.debug("BM25检索完成",
                        query=query[:50],
                        collection_name=collection_name,
                        results_count=len(results),
                        search_time=f"{search_time:.3f}s")
            
            # 添加分数详情日志
            if results:
                scores = [float(r.get('score', 0.0)) for r in results]
                logger.info("BM25分数详情",
                            max_score=f"{max(scores):.4f}",
                            min_score=f"{min(scores):.4f}",
                            avg_score=f"{sum(scores)/len(scores):.4f}",
                            threshold=self.config.score_threshold,
                            passed_count=len([s for s in scores if s >= self.config.score_threshold]))
            
            # 3. 格式化结果，确保与语义检索结果格式兼容
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result.get('id', ''),
                    'text': result.get('text', ''),
                    'score': float(result.get('score', 0.0)),
                    'metadata': result.get('metadata', {}),
                    'retrieval_type': 'bm25',
                    'query_coverage': result.get('query_coverage', {}),
                    'bm25_score': float(result.get('score', 0.0)),  # 保留原始BM25分数
                }
                formatted_results.append(formatted_result)
            
            return formatted_results[:top_k]  # 只返回请求的数量
            
        except asyncio.TimeoutError:
            logger.warning("BM25检索超时", query=query[:50])
            return []
        except Exception as e:
            logger.error("BM25检索失败",
                        query=query[:50],
                        error=str(e))
            return []
    
    async def _get_or_create_index_async(self, collection_name: str):
        """
        异步获取或创建BM25索引
        
        Returns:
            BM25Retriever实例或None
        """
        async with self.cache_lock:
            # 检查缓存是否有效
            if self._is_cache_valid():
                return self.bm25_retriever_cache
            
            # 缓存无效或不存在，需要构建新索引
            return await self._build_index_async(collection_name)
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if self.bm25_retriever_cache is None:
            return False
        
        # 检查缓存是否过期
        current_time = time.time()
        cache_age = current_time - self.last_cache_time
        
        if cache_age > self.config.cache_ttl_seconds:
            logger.debug(f"BM25缓存已过期，年龄: {cache_age:.1f}s")
            return False
        
        return True
    
    async def _build_index_async(self, collection_name: str):
        """
        异步构建BM25索引
        
        Returns:
            BM25Retriever实例或None
        """
        logger.info("开始异步构建BM25索引", collection_name=collection_name)
        build_start_time = time.time()
        
        try:
            # 1. 异步获取文档数据
            documents = await self._fetch_documents_async(collection_name)
            if not documents:
                logger.warning("没有获取到文档数据，无法构建BM25索引")
                return None
            
            # 2. 在线程中构建BM25索引
            def build_bm25_index():
                from src.core.retrieval.bm25_retriever import BM25Retriever
                
                retriever = BM25Retriever(
                    use_jieba=self.config.use_jieba,
                    remove_stopwords=self.config.remove_stopwords,
                    k1=self.config.k1,
                    b=self.config.b
                )
                
                retriever.build_index(documents)
                return retriever
            
            retriever = await asyncio.to_thread(build_bm25_index)
            
            # 3. 更新缓存
            self.bm25_retriever_cache = retriever
            self.cached_document_ids = {doc['id'] for doc in documents}
            self.last_cache_time = time.time()
            
            build_time = time.time() - build_start_time
            
            logger.info("BM25索引构建完成",
                       collection_name=collection_name,
                       documents_count=len(documents),
                       build_time=f"{build_time:.3f}s")
            
            return retriever
            
        except ImportError as e:
            logger.error("导入BM25Retriever失败", error=str(e))
            return None
        except Exception as e:
            logger.error("构建BM25索引失败", error=str(e))
            return None
    
    async def _fetch_documents_async(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        异步从向量数据库获取文档数据
        
        Returns:
            文档列表，每个文档包含id、text、metadata
        """
        try:
            # 使用RealQdrantManager实际可用的方法
            # 首先检查集合是否存在
            if not await self.vector_manager.collection_exists(collection_name):
                logger.warning(f"集合 {collection_name} 不存在")
                return []
            
            # 限制获取的文档数量（避免内存问题）
            limit = 1000  # 最多1000个文档
            
            logger.debug(f"从集合 {collection_name} 获取文档", limit=limit)
            
            # 使用Scroll API获取所有文档点，而不是Search API
            # Scroll API更适合批量获取文档
            import requests
            import json
            
            # Qdrant的Scroll API端点
            qdrant_url = "http://qdrant:6333"  # 容器内部地址
            scroll_url = f"{qdrant_url}/collections/{collection_name}/points/scroll"
            
            # Scroll请求参数
            scroll_payload = {
                "limit": limit,
                "with_payload": True,
                "with_vector": False  # 不需要向量
            }
            
            try:
                # 发送Scroll请求
                response = requests.post(
                    scroll_url,
                    json=scroll_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    points_data = result.get("result", {})
                    points = points_data.get("points", [])
                    
                    logger.debug(f"通过Scroll API获取到 {len(points)} 个点")
                    
                    # 转换为search_results格式
                    search_results = []
                    for point in points:
                        search_results.append({
                            "id": point.get("id"),
                            "score": 1.0,  # Scroll没有分数，设为1.0
                            "payload": point.get("payload", {})
                        })
                    
                    logger.debug(f"成功转换 {len(search_results)} 个文档")
                else:
                    logger.error(f"Scroll API请求失败: {response.status_code}", 
                                response_text=response.text[:200])
                    search_results = []
                    
            except Exception as e:
                logger.error(f"Scroll API异常: {str(e)}")
                search_results = []
            
            
            # search_results是字典列表，需要转换为文档格式
            documents = []
            for result in search_results:
                payload = result.get("payload", {})
                text = payload.get("text", "")
                if text:  # 只处理有文本的文档
                    documents.append({
                        "id": result.get("id"),
                        "text": text,
                        "metadata": {
                            "document_id": payload.get("document_id"),
                            "chunk_index": payload.get("chunk_index"),
                            **payload.get("metadata", {})
                        }
                    })
            
            logger.debug(f"从集合 {collection_name} 获取到 {len(documents)} 个文档")
            return documents
            
            # 提取文档数据
            documents = []
            for point in points:
                payload = point.get("payload", {})
                text = payload.get("text", "")
                
                # 只包含有文本内容的文档
                if text and len(text.strip()) > 10:
                    doc = {
                        "id": str(point.get("id", "")),
                        "text": text,
                        "metadata": payload.get("metadata", {})
                    }
                    documents.append(doc)
            
            logger.debug(f"成功获取 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error("获取文档数据失败",
                        collection_name=collection_name,
                        error=str(e))
            return []
    
    async def refresh_index(self, collection_name: str):
        """异步刷新BM25索引（强制重建）"""
        logger.info("刷新BM25索引", collection_name=collection_name)
        
        async with self.cache_lock:
            self.bm25_retriever_cache = None
            self.cached_document_ids = set()
        
        # 异步重建索引
        return await self._build_index_async(collection_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查索引状态
            index_healthy = self.bm25_retriever_cache is not None
            
            # 检查文档数量
            document_count = len(self.cached_document_ids) if self.cached_document_ids else 0
            
            # 检查缓存年龄
            cache_age = time.time() - self.last_cache_time if self.last_cache_time > 0 else 0
            
            return {
                "status": "healthy" if index_healthy else "unhealthy",
                "index_available": index_healthy,
                "cached_documents": document_count,
                "cache_age_seconds": round(cache_age, 1),
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "config": {
                    "use_jieba": self.config.use_jieba,
                    "remove_stopwords": self.config.remove_stopwords,
                    "k1": self.config.k1,
                    "b": self.config.b
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = None
        
        self.bm25_retriever_cache = None
        self.cached_document_ids.clear()
        
        logger.info("BM25异步管理器已关闭")


# 全局实例管理
_bm25_manager_instance = None

async def get_bm25_manager(vector_manager, config: Optional[BM25Config] = None):
    """获取BM25管理器实例（异步单例）"""
    global _bm25_manager_instance
    
    if _bm25_manager_instance is None:
        _bm25_manager_instance = BM25AsyncManager(vector_manager, config)
        await _bm25_manager_instance.initialize()
    
    return _bm25_manager_instance