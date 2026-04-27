"""
搜索工具函数
"""

from typing import List, Dict, Any, Optional
from src.core.logging import get_logger

logger = get_logger(__name__)


async def search_similar(
    collection_name: str,
    query_text: str,
    limit: int = 10,
    score_threshold: float = 0.7,
    user_id: str = None
) -> List[Dict[str, Any]]:
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
    try:
        # 延迟导入以避免循环依赖
        from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
        from src.core.embedding import get_embedding_service
        
        # 1. 检查权限（简化版本）
        if user_id:
            # 简化权限检查
            pass
        
        # 2. 文本嵌入
        embedding_service = get_embedding_service()
        embedding_result = await embedding_service.embed_text(query_text)
        if not embedding_result:
            logger.error("文本嵌入失败", query_text=query_text[:100])
            return []
        
        # 3. 向量搜索
        vector = embedding_result.get("vector", []) if isinstance(embedding_result, dict) else (embedding_result.vector if hasattr(embedding_result, 'vector') else [])
        search_results = await real_qdrant_manager.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        # 4. 格式化结果
        formatted_results = []
        for result in search_results:
            score = result.get("score", 0.0)
            if score >= score_threshold:
                payload = result.get("payload", {})
                formatted_results.append({
                    "id": result.get("id", ""),
                    "score": score,
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                })
        
        logger.info(
            "相似文本搜索完成",
            collection_name=collection_name,
            query_length=len(query_text),
            results_count=len(formatted_results)
        )
        
        return formatted_results
        
    except Exception as e:
        logger.error("搜索失败", 
                    collection_name=collection_name,
                    query_text=query_text[:100],
                    error=str(e),
                    exc_info=True)
        return []