"""
简单语义检索QA接口 - 用于测试基础异步链路

只保留语义检索，移除BM25和重排序，简化问题
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
import time

from src.core.logging import get_logger
from src.core.embedding import get_embedding_service
from src.core.vector_db.real_qdrant_manager import real_qdrant_manager

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["简单语义检索"])


class SimpleSemanticRetriever:
    """简单语义检索器 - 只做语义检索"""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant_manager = real_qdrant_manager
        logger.info("简单语义检索器初始化完成")
    
    async def search(self, 
                    query: str, 
                    collection_name: str = "default",
                    top_k: int = 5,
                    score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        执行语义检索
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            检索结果列表
        """
        try:
            start_time = time.time()
            
            # 1. 生成查询向量
            logger.info(f"开始生成查询向量: {query[:50]}...")
            embedding_result = await self.embedding_service.embed_text(query)
            query_vector = embedding_result.vector
            logger.info(f"查询向量生成完成，维度: {len(query_vector)}")
            
            # 2. 在Qdrant中搜索
            logger.info(f"在Qdrant中搜索，集合: {collection_name}, top_k: {top_k}")
            search_results = await self.qdrant_manager.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # 3. 格式化结果
            results = []
            for point in search_results:
                # 处理字典或对象
                if isinstance(point, dict):
                    point_id = point.get("id", "")
                    point_score = point.get("score", 0.0)
                    point_payload = point.get("payload", {})
                else:
                    point_id = str(point.id)
                    point_score = point.score
                    point_payload = point.payload
                
                result = {
                    "id": str(point_id),
                    "text": point_payload.get("text", ""),
                    "score": point_score,
                    "metadata": point_payload.get("metadata", {}),
                    "retrieval_type": "semantic",
                    "score_breakdown": {
                        "semantic_score": point_score
                    }
                }
                results.append(result)
            
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(f"语义检索完成，返回 {len(results)} 个结果，耗时: {elapsed_time:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            raise


# 全局检索器实例
_simple_retriever = None


def get_simple_retriever():
    """获取简单语义检索器实例"""
    global _simple_retriever
    
    if _simple_retriever is None:
        try:
            _simple_retriever = SimpleSemanticRetriever()
            logger.info("简单语义检索器创建成功")
        except Exception as e:
            logger.error(f"创建简单语义检索器失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"检索器初始化失败: {str(e)}"
            )
    
    return _simple_retriever


@router.post("/simple/semantic/query")
async def simple_semantic_query(
    request: Dict[str, Any],
    retriever=Depends(get_simple_retriever)
):
    """
    简单语义检索查询接口
    
    请求参数：
    {
        "query": "查询文本",
        "collection_name": "default",  # 可选
        "top_k": 5,                    # 可选
        "score_threshold": 0.3         # 可选
    }
    
    返回结果：
    {
        "success": true,
        "message": "查询成功",
        "data": {
            "contexts": [
                {
                    "id": "文档ID",
                    "text": "文档内容",
                    "score": 0.85,
                    "metadata": {},
                    "retrieval_type": "semantic",
                    "score_breakdown": {
                        "semantic_score": 0.85
                    }
                }
            ],
            "retrieval_method": "semantic",
            "retrieval_time": 123.45,
            "query": "原始查询",
            "collection_name": "default",
            "top_k": 5,
            "score_threshold": 0.3
        }
    }
    """
    try:
        # 解析请求参数
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        collection_name = request.get("collection_name", "default")
        top_k = request.get("top_k", 5)
        score_threshold = request.get("score_threshold", 0.3)
        
        logger.info(f"收到简单语义检索请求: query='{query[:50]}...', collection={collection_name}")
        
        # 执行语义检索
        start_time = time.time()
        contexts = await retriever.search(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold
        )
        retrieval_time = (time.time() - start_time) * 1000
        
        # 构建响应
        response = {
            "success": True,
            "message": "查询成功",
            "data": {
                "contexts": contexts,
                "retrieval_method": "semantic",
                "retrieval_time": retrieval_time,
                "query": query,
                "collection_name": collection_name,
                "top_k": top_k,
                "score_threshold": score_threshold
            }
        }
        
        logger.info(f"简单语义检索成功，返回 {len(contexts)} 个结果")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"简单语义检索接口异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"查询失败: {str(e)}"
        )


@router.get("/simple/semantic/health")
async def simple_semantic_health(retriever=Depends(get_simple_retriever)):
    """
    简单语义检索健康检查
    
    返回结果：
    {
        "status": "healthy",
        "service": "simple-semantic-retriever",
        "components": {
            "embedding_service": "connected",
            "qdrant": "connected"
        }
    }
    """
    try:
        # 测试嵌入服务
        embedding_health = await retriever.embedding_service.health_check()
        
        # 测试Qdrant连接 - 使用简单的方法检查
        try:
            # 尝试获取集合信息来检查连接
            collection_info = await retriever.qdrant_manager.get_collection_info("default")
            qdrant_status = "connected"
        except Exception as e:
            qdrant_status = f"error: {str(e)[:100]}"
        
        return {
            "status": "healthy",
            "service": "simple-semantic-retriever",
            "components": {
                "embedding_service": embedding_health.get("status", "unknown"),
                "qdrant": qdrant_status
            }
        }
        
    except Exception as e:
        logger.error(f"简单语义检索健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "service": "simple-semantic-retriever",
            "error": str(e)
        }