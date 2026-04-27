"""
向量数据库主模块
"""
from src.core.vector_db import (
    # Qdrant客户端
    QdrantManager,
    DistanceMetric,
    VectorSearchResult,
    qdrant_manager,
    
    # Collection管理
    CollectionManager,
    CollectionConfig,
    collection_manager,
    
    # 嵌入服务
    EmbeddingService,
    EmbeddingResult,
    embedding_service,
)

__all__ = [
    "QdrantManager",
    "DistanceMetric",
    "VectorSearchResult",
    "qdrant_manager",
    "CollectionManager",
    "CollectionConfig",
    "collection_manager",
    "EmbeddingService",
    "EmbeddingResult",
    "embedding_service",
]

# 导出常用函数
async def search_similar(
    collection_name: str,
    query_text: str,
    limit: int = 10,
    score_threshold: float = 0.7,
    user_id: str = None
) -> list:
    """
    搜索相似文本（完整流程）
    
    Args:
        collection_name: Collection名称
        query_text: 查询文本
        limit: 返回结果数量
        score_threshold: 分数阈值
        user_id: 用户ID（用于权限检查）
        
    Returns:
        搜索结果列表
    """
    from src.core.logging import get_logger
    logger = get_logger("openclaw.vector_db")
    
    try:
        # 1. 检查权限
        if user_id:
            has_access = await collection_manager.user_has_access(user_id, collection_name)
            if not has_access:
                logger.warning("用户无Collection访问权限", user_id=user_id, collection_name=collection_name)
                return []
        
        # 2. 文本嵌入
        embedding_result = await embedding_service.embed_text(query_text)
        if not embedding_result:
            logger.error("文本嵌入失败", query_text=query_text[:100])
            return []
        
        # 3. 向量搜索
        search_results = await qdrant_manager.search(
            collection_name=collection_name,
            query_vector=embedding_result.vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        # 4. 格式化结果
        formatted_results = []
        for result in search_results:
            if result.score >= score_threshold:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                })
        
        logger.info(
            "相似文本搜索完成",
            collection_name=collection_name,
            query_length=len(query_text),
            results_count=len(formatted_results)
        )
        
        return formatted_results
        
    except Exception as e:
        logger.error("相似文本搜索失败", error=str(e))
        return []

async def add_document_to_collection(
    collection_name: str,
    text: str,
    metadata: dict = None,
    user_id: str = None
) -> bool:
    """
    添加文档到Collection
    
    Args:
        collection_name: Collection名称
        text: 文档文本
        metadata: 文档元数据
        user_id: 用户ID
        
    Returns:
        是否添加成功
    """
    from src.core.logging import get_logger
    logger = get_logger("openclaw.vector_db")
    
    try:
        # 1. 检查权限
        if user_id:
            has_access = await collection_manager.user_has_access(user_id, collection_name)
            if not has_access:
                logger.warning("用户无Collection访问权限", user_id=user_id, collection_name=collection_name)
                return False
        
        # 2. 文本嵌入
        embedding_result = await embedding_service.embed_text(text)
        if not embedding_result:
            logger.error("文本嵌入失败", text=text[:100])
            return False
        
        # 3. 准备向量点
        from src.core.vector_db.qdrant_client import qdrant_manager
        point = qdrant_manager.create_point(
            vector=embedding_result.vector,
            payload={
                "text": text,
                "metadata": metadata or {},
                "embedding_model": embedding_result.model,
                "created_at": embedding_result.created_at.isoformat(),
            }
        )
        
        # 4. 插入向量点
        success = await qdrant_manager.upsert_points(
            collection_name=collection_name,
            points=[point]
        )
        
        if success:
            logger.info(
                "文档添加到Collection成功",
                collection_name=collection_name,
                text_length=len(text),
                vector_dimension=len(embedding_result.vector)
            )
        
        return success
        
    except Exception as e:
        logger.error("添加文档到Collection失败", error=str(e))
        return False

async def create_user_collection(
    user_id: str,
    collection_name: str,
    vector_size: int = 1536,
    description: str = ""
) -> bool:
    """
    为用户创建Collection
    
    Args:
        user_id: 用户ID
        collection_name: Collection名称
        vector_size: 向量维度
        description: 描述
        
    Returns:
        是否创建成功
    """
    return await collection_manager.create_user_collection(
        user_id=user_id,
        collection_name=collection_name,
        vector_size=vector_size,
        description=description
    )

async def get_collection_stats(collection_name: str) -> dict:
    """
    获取Collection统计信息
    
    Args:
        collection_name: Collection名称
        
    Returns:
        统计信息
    """
    stats = await qdrant_manager.get_collection_stats(collection_name)
    if stats:
        return stats
    
    # 如果获取失败，返回基础信息
    info = await qdrant_manager.get_collection_info(collection_name)
    if info:
        return {
            "name": info["name"],
            "points_count": info["points_count"],
            "status": info["status"],
        }
    
    return {}

async def health_check() -> dict:
    """
    向量数据库健康检查
    
    Returns:
        健康状态
    """
    from src.core.logging import get_logger
    logger = get_logger("openclaw.vector_db")
    
    try:
        # 检查Qdrant连接
        qdrant_health = await qdrant_manager.health_check()
        
        # 检查嵌入服务
        embedding_health = await embedding_service.health_check()
        
        # 获取Collection列表
        collections = await qdrant_manager.list_collections()
        
        return {
            "status": "healthy" if qdrant_health.get("connected") and embedding_health.get("status") == "healthy" else "unhealthy",
            "qdrant": qdrant_health,
            "embeddings": embedding_health,
            "collections_count": len(collections),
            "collections": collections[:10],  # 只返回前10个
        }
        
    except Exception as e:
        logger.error("向量数据库健康检查失败", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
        }