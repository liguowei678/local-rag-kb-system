"""
GraphRAG API端点
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.core.graph.factory import GraphRAGFactory
from src.core.graph.builder import KnowledgeGraphBuilder
from src.core.graph.retriever import GraphRetriever

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graphrag"])  # 去掉前缀，由主路由统一添加


# 请求/响应模型
class GraphRAGStatusResponse(BaseModel):
    """GraphRAG状态响应"""
    enabled: bool
    config: Dict[str, Any]
    dependencies: Dict[str, bool]
    services: Dict[str, str]
    statistics: Optional[Dict[str, Any]] = None


class GraphQueryRequest(BaseModel):
    """图查询请求"""
    query: str = Field(..., description="查询文本")
    max_depth: int = Field(2, description="图搜索深度")
    limit: int = Field(10, description="返回结果数量")


class GraphQueryResponse(BaseModel):
    """图查询响应"""
    entities: List[Dict[str, Any]]
    document_ids: List[str]
    query: str
    search_depth: int


class GraphRetrievalRequest(BaseModel):
    """图检索请求"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(5, description="返回文档数量")
    min_relevance: float = Field(0.5, description="最小相关性")


class GraphRetrievalResponse(BaseModel):
    """图检索响应"""
    documents: List[Dict[str, Any]]
    query: str
    retrieval_stats: Dict[str, Any]


# 后台任务存储
background_tasks = {}


@router.get("/status", response_model=GraphRAGStatusResponse)
async def get_graphrag_status():
    """获取GraphRAG状态"""
    try:
        status = GraphRAGFactory.get_graphrag_status()
        return GraphRAGStatusResponse(**status)
    except Exception as e:
        logger.error(f"获取GraphRAG状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.post("/query", response_model=GraphQueryResponse)
async def graph_query(request: GraphQueryRequest):
    """执行图查询"""
    try:
        # 获取配置
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建图检索器
        retriever = GraphRAGFactory.create_graph_retriever(config)
        if not retriever:
            raise HTTPException(status_code=500, detail="无法创建图检索器")
        
        # 执行图查询
        result = retriever.graph_query(
            query_text=request.query,
            max_depth=request.max_depth,
            limit=request.limit
        )
        
        return GraphQueryResponse(
            entities=result.get("entities", []),
            document_ids=result.get("document_ids", []),
            query=request.query,
            search_depth=request.max_depth
        )
        
    except Exception as e:
        logger.error(f"图查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"图查询失败: {str(e)}")


@router.post("/retrieve", response_model=GraphRetrievalResponse)
async def graph_retrieval(request: GraphRetrievalRequest):
    """执行图检索"""
    try:
        # 获取配置
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建图检索器
        retriever = GraphRAGFactory.create_graph_retriever(config)
        if not retriever:
            raise HTTPException(status_code=500, detail="无法创建图检索器")
        
        # 创建RAG查询对象
        from src.core.rag.types import RAGQuery
        
        rag_query = RAGQuery(
            query=request.query,  # 修复：使用query参数，不是text
            top_k=request.top_k
        )
        
        # 执行检索
        documents = await retriever.retrieve(rag_query)
        
        # 转换为字典格式
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "id": doc.id,
                "content": doc.content,
                "score": doc.score,
                "metadata": doc.metadata
            })
        
        # 获取检索统计
        retrieval_stats = retriever.get_retrieval_stats()
        
        return GraphRetrievalResponse(
            documents=doc_dicts,
            query=request.query,
            retrieval_stats=retrieval_stats
        )
        
    except Exception as e:
        logger.error(f"图检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"图检索失败: {str(e)}")


@router.post("/clear")
async def clear_knowledge_graph():
    """清空知识图谱"""
    try:
        # 获取配置
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建知识图谱构建器
        builder = GraphRAGFactory.create_knowledge_graph_builder(config)
        if not builder:
            raise HTTPException(status_code=500, detail="无法创建知识图谱构建器")
        
        # 清空图谱
        success = builder.clear_graph()
        
        if success:
            return {"success": True, "message": "知识图谱已清空"}
        else:
            raise HTTPException(status_code=500, detail="清空知识图谱失败")
        
    except Exception as e:
        logger.error(f"清空知识图谱失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")


@router.get("/statistics")
async def get_graph_statistics():
    """获取图谱统计信息"""
    try:
        # 获取配置
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建Neo4j客户端
        client = GraphRAGFactory.create_neo4j_client(config)
        
        # 获取统计
        stats = client.get_statistics()
        
        client.close()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"获取图谱统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/test/components")
async def test_components():
    """测试所有组件"""
    try:
        results = GraphRAGFactory.test_components()
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"测试组件失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")