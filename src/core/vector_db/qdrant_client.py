"""
Qdrant向量数据库客户端封装
"""
import asyncio
import uuid
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.core.logging import get_logger, log_execution_time
from src.core.cache import cached_with_stats

logger = get_logger("openclaw.vector_db")

class DistanceMetric(Enum):
    """距离度量标准"""
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"

class VectorSearchResult:
    """向量搜索结果"""
    
    def __init__(
        self,
        id: Union[str, int],
        score: float,
        payload: Dict[str, Any],
        vector: Optional[List[float]] = None
    ):
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "score": self.score,
            "payload": self.payload,
            "has_vector": self.vector is not None
        }
    
    def __repr__(self):
        return f"VectorSearchResult(id={self.id}, score={self.score:.4f})"

class QdrantManager:
    """Qdrant向量数据库管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._connected = False
        return cls._instance
    
    @log_execution_time("openclaw.vector_db")
    async def connect(self) -> QdrantClient:
        """连接到Qdrant服务器"""
        if self._client is None or not self._connected:
            try:
                self._client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=30,
                    prefer_grpc=True,  # 优先使用gRPC协议，性能更好
                )
                
                # 测试连接
                self._client.get_collections()
                self._connected = True
                
                logger.info(
                    "Qdrant连接成功",
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT
                )
                
            except Exception as e:
                logger.error("Qdrant连接失败", error=str(e))
                raise
        
        return self._client
    
    async def disconnect(self):
        """断开Qdrant连接"""
        if self._client:
            # QdrantClient没有显式的close方法
            self._client = None
            self._connected = False
            logger.info("Qdrant连接已关闭")
    
    async def get_client(self) -> QdrantClient:
        """获取Qdrant客户端实例"""
        return await self.connect()
    
    # Collection管理
    
    @log_execution_time("openclaw.vector_db")
    @cached_with_stats(ttl=60, key_prefix="qdrant_collections")
    async def list_collections(self) -> List[str]:
        """列出所有Collection"""
        client = await self.get_client()
        
        try:
            collections = client.get_collections()
            return [collection.name for collection in collections.collections]
            
        except Exception as e:
            logger.error("获取Collection列表失败", error=str(e))
            return []
    
    @log_execution_time("openclaw.vector_db")
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> bool:
        """
        创建Collection
        
        Args:
            collection_name: Collection名称
            vector_size: 向量维度
            distance: 距离度量标准
            **kwargs: 其他参数
            
        Returns:
            是否创建成功
        """
        client = await self.get_client()
        
        try:
            # 检查Collection是否已存在
            collections = await self.list_collections()
            if collection_name in collections:
                logger.warning("Collection已存在", collection_name=collection_name)
                return True
            
            # 创建Collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance.value
                ),
                **kwargs
            )
            
            logger.info(
                "Collection创建成功",
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance.value
            )
            
            return True
            
        except Exception as e:
            logger.error("Collection创建失败", collection_name=collection_name, error=str(e))
            return False
    
    @log_execution_time("openclaw.vector_db")
    async def delete_collection(self, collection_name: str) -> bool:
        """删除Collection"""
        client = await self.get_client()
        
        try:
            client.delete_collection(collection_name=collection_name)
            
            logger.info("Collection删除成功", collection_name=collection_name)
            return True
            
        except Exception as e:
            logger.error("Collection删除失败", collection_name=collection_name, error=str(e))
            return False
    
    @log_execution_time("openclaw.vector_db")
    @cached_with_stats(ttl=30, key_prefix="qdrant_collection_info")
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """获取Collection信息"""
        client = await self.get_client()
        
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            
            return {
                "name": collection_info.config.params.name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
            }
            
        except Exception as e:
            logger.error("获取Collection信息失败", collection_name=collection_name, error=str(e))
            return None
    
    # 向量操作
    
    @log_execution_time("openclaw.vector_db")
    async def upsert_points(
        self,
        collection_name: str,
        points: List[models.PointStruct],
        wait: bool = True
    ) -> bool:
        """
        插入或更新向量点
        
        Args:
            collection_name: Collection名称
            points: 向量点列表
            wait: 是否等待操作完成
            
        Returns:
            是否操作成功
        """
        client = await self.get_client()
        
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=wait
            )
            
            logger.debug(
                "向量点插入/更新成功",
                collection_name=collection_name,
                points_count=len(points)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "向量点插入/更新失败",
                collection_name=collection_name,
                points_count=len(points),
                error=str(e)
            )
            return False
    
    @log_execution_time("openclaw.vector_db")
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter: Optional[models.Filter] = None
    ) -> List[VectorSearchResult]:
        """
        向量搜索
        
        Args:
            collection_name: Collection名称
            query_vector: 查询向量
            limit: 返回结果数量
            score_threshold: 分数阈值
            with_payload: 是否返回payload
            with_vectors: 是否返回向量
            filter: 过滤条件
            
        Returns:
            搜索结果列表
        """
        client = await self.get_client()
        
        try:
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors,
                query_filter=filter
            )
            
            results = []
            for hit in search_result:
                results.append(VectorSearchResult(
                    id=hit.id,
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=hit.vector if with_vectors else None
                ))
            
            logger.debug(
                "向量搜索完成",
                collection_name=collection_name,
                query_size=len(query_vector),
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "向量搜索失败",
                collection_name=collection_name,
                error=str(e)
            )
            return []
    
    @log_execution_time("openclaw.vector_db")
    async def search_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        **kwargs
    ) -> List[List[VectorSearchResult]]:
        """
        批量向量搜索
        
        Args:
            collection_name: Collection名称
            query_vectors: 查询向量列表
            limit: 每个查询返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            批量搜索结果
        """
        client = await self.get_client()
        
        try:
            search_results = client.search_batch(
                collection_name=collection_name,
                requests=[
                    models.SearchRequest(
                        vector=vector,
                        limit=limit,
                        **kwargs
                    )
                    for vector in query_vectors
                ]
            )
            
            batch_results = []
            for result_list in search_results:
                results = []
                for hit in result_list:
                    results.append(VectorSearchResult(
                        id=hit.id,
                        score=hit.score,
                        payload=hit.payload or {},
                        vector=None
                    ))
                batch_results.append(results)
            
            logger.debug(
                "批量向量搜索完成",
                collection_name=collection_name,
                batch_size=len(query_vectors),
                total_results=sum(len(r) for r in batch_results)
            )
            
            return batch_results
            
        except Exception as e:
            logger.error(
                "批量向量搜索失败",
                collection_name=collection_name,
                batch_size=len(query_vectors),
                error=str(e)
            )
            return []
    
    @log_execution_time("openclaw.vector_db")
    async def recommend(
        self,
        collection_name: str,
        positive: List[Union[str, int]],
        negative: Optional[List[Union[str, int]]] = None,
        limit: int = 10,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        推荐相似向量（基于正例和负例）
        
        Args:
            collection_name: Collection名称
            positive: 正例向量ID列表
            negative: 负例向量ID列表
            limit: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            推荐结果列表
        """
        client = await self.get_client()
        
        try:
            recommend_result = client.recommend(
                collection_name=collection_name,
                positive=positive,
                negative=negative or [],
                limit=limit,
                **kwargs
            )
            
            results = []
            for hit in recommend_result:
                results.append(VectorSearchResult(
                    id=hit.id,
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=None
                ))
            
            logger.debug(
                "向量推荐完成",
                collection_name=collection_name,
                positive_count=len(positive),
                negative_count=len(negative) if negative else 0,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "向量推荐失败",
                collection_name=collection_name,
                error=str(e)
            )
            return []
    
    @log_execution_time("openclaw.vector_db")
    async def delete_points(
        self,
        collection_name: str,
        points_ids: List[Union[str, int]],
        wait: bool = True
    ) -> bool:
        """删除向量点"""
        client = await self.get_client()
        
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=points_ids
                ),
                wait=wait
            )
            
            logger.debug(
                "向量点删除成功",
                collection_name=collection_name,
                points_count=len(points_ids)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "向量点删除失败",
                collection_name=collection_name,
                points_count=len(points_ids),
                error=str(e)
            )
            return False
    
    @log_execution_time("openclaw.vector_db")
    async def delete_by_filter(
        self,
        collection_name: str,
        filter: models.Filter,
        wait: bool = True
    ) -> bool:
        """根据过滤条件删除向量点"""
        client = await self.get_client()
        
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=filter
                ),
                wait=wait
            )
            
            logger.debug(
                "按条件删除向量点成功",
                collection_name=collection_name
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "按条件删除向量点失败",
                collection_name=collection_name,
                error=str(e)
            )
            return False
    
    # 统计和监控
    
    @log_execution_time("openclaw.vector_db")
    async def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """获取Collection统计信息"""
        client = await self.get_client()
        
        try:
            # 获取Collection信息
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info:
                return None
            
            # 获取点数量统计
            count_result = client.count(
                collection_name=collection_name,
                exact=True
            )
            
            return {
                **collection_info,
                "exact_count": count_result.count,
                "estimated_count": collection_info["points_count"],
            }
            
        except Exception as e:
            logger.error(
                "获取Collection统计失败",
                collection_name=collection_name,
                error=str(e)
            )
            return None
    
    @log_execution_time("openclaw.vector_db")
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self.get_client()
            
            # 获取服务状态
            collections = client.get_collections()
            
            return {
                "status": "healthy",
                "qdrant_version": "1.0.0",  # 实际应该从API获取
                "collections_count": len(collections.collections),
                "connected": True,
            }
            
        except Exception as e:
            logger.error("Qdrant健康检查失败", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
            }
    
    # 工具方法
    
    @staticmethod
    def create_point(
        vector: List[float],
        payload: Dict[str, Any],
        point_id: Optional[Union[str, int]] = None
    ) -> models.PointStruct:
        """
        创建向量点
        
        Args:
            vector: 向量
            payload: 附加数据
            point_id: 点ID，None则自动生成
            
        Returns:
            向量点结构
        """
        if point_id is None:
            point_id = str(uuid.uuid4())
        
        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
    
    @staticmethod
    def create_filter(
        must: Optional[List[models.FieldCondition]] = None,
        must_not: Optional[List[models.FieldCondition]] = None,
        should: Optional[List[models.FieldCondition]] = None,
        **kwargs
    ) -> models.Filter:
        """
        创建过滤条件
        
        Args:
            must: 必须满足的条件
            must_not: 必须不满足的条件
            should: 应该满足的条件（至少一个）
            **kwargs: 其他参数
            
        Returns:
            过滤条件
        """
        return models.Filter(
            must=must or [],
            must_not=must_not or [],
            should=should or [],
            **kwargs
        )
    
    @staticmethod
    def create_field_condition(
        key: str,
        match: Optional[Union[str, List[str], models.MatchValue, models.MatchText]] = None,
        range: Optional[models.Range] = None,
        geo_radius: Optional[models.GeoRadius] = None,
        geo_bounding_box: Optional[models.GeoBoundingBox] = None,
        values_count: Optional[models.ValuesCount] = None,
        **kwargs
    ) -> models.FieldCondition:
        """
        创建字段条件
        
        Args:
            key: 字段名
            match: 匹配条件
            range: 范围条件
            geo_radius: 地理半径条件
            geo_bounding_box: 地理边界框条件
            values_count: 值数量条件
            **kwargs: 其他参数
            
        Returns:
            字段条件
        """
        return models.FieldCondition(
            key=key,
            match=match,
            range=range,
            geo_radius=geo_radius,
            geo_bounding_box=geo_bounding_box,
            values_count=values_count,
            **kwargs
        )

# 全局Qdrant管理器实例
qdrant_manager = QdrantManager()