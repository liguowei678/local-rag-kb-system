"""
混合检索问答API端点（支持GraphRAG + 重排序）
提供完整的RAG问答功能，支持语义检索、BM25关键词检索、GraphRAG图检索和Cross-Encoder重排序
"""

import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.rag.pipeline_hybrid_rerank_with_graph import (
    get_hybrid_rag_pipeline_with_graphrag,
    EnhancedHybridConfig,
    RerankConfig
)
from src.core.rag.types import RAGQuery

logger = get_logger(__name__)

router = APIRouter(prefix="/qa-hybrid-graphrag", tags=["混合检索问答（支持GraphRAG）"])


class QueryRequest(BaseModel):
    """查询请求（支持GraphRAG）"""
    query: str = Field(..., description="查询文本", min_length=1, max_length=1000)
    top_k: int = Field(5, description="返回文档数量", ge=1, le=20)
    score_threshold: float = Field(0.6, description="分数阈值", ge=0.0, le=1.0)
    collection_name: str = Field("default", description="集合名称")
    
    # 混合检索权重
    semantic_weight: float = Field(0.5, description="语义检索权重", ge=0.0, le=1.0)
    bm25_weight: float = Field(0.3, description="BM25检索权重", ge=0.0, le=1.0)
    
    # GraphRAG配置
    use_graphrag: bool = Field(True, description="是否使用GraphRAG")
    graph_max_depth: int = Field(2, description="图搜索深度", ge=1, le=5)
    enable_graph_fallback: bool = Field(True, description="是否启用图检索降级")
    
    # 重排序配置
    use_rerank: bool = Field(True, description="是否使用重排序")
    rerank_original_weight: float = Field(0.4, description="重排序原始分数权重", ge=0.0, le=1.0)
    rerank_weight: float = Field(0.6, description="重排序分数权重", ge=0.0, le=1.0)
    top_k_before_rerank: int = Field(12, description="重排序前的文档数量", ge=5, le=50)
    top_k_after_rerank: int = Field(5, description="重排序后的文档数量", ge=1, le=20)
    
    # 生成配置
    max_tokens: int = Field(1000, description="最大token数", ge=100, le=4000)
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)


class BatchQueryRequest(BaseModel):
    """批量查询请求（支持GraphRAG）"""
    queries: List[str] = Field(..., description="查询列表", min_items=1, max_items=10)
    top_k: int = Field(5, description="返回文档数量", ge=1, le=20)
    score_threshold: float = Field(0.6, description="分数阈值", ge=0.0, le=1.0)
    collection_name: str = Field("default", description="集合名称")
    
    # 混合检索权重
    semantic_weight: float = Field(0.5, description="语义检索权重", ge=0.0, le=1.0)
    bm25_weight: float = Field(0.3, description="BM25检索权重", ge=0.0, le=1.0)
    
    # GraphRAG配置
    use_graphrag: bool = Field(True, description="是否使用GraphRAG")
    
    # 重排序配置
    use_rerank: bool = Field(True, description="是否使用重排序")
    rerank_original_weight: float = Field(0.4, description="重排序原始分数权重", ge=0.0, le=1.0)
    rerank_weight: float = Field(0.6, description="重排序分数权重", ge=0.0, le=1.0)
    
    # 生成配置
    max_tokens: int = Field(1000, description="最大token数", ge=100, le=4000)
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_types: List[str] = []


class QueryResponse(BaseModel):
    """查询响应"""
    query: str
    answer: str
    documents: List[DocumentResponse]
    retrieval_time: float
    metadata: Dict[str, Any]
    graphrag_info: Optional[Dict[str, Any]] = None


class BatchQueryResponse(BaseModel):
    """批量查询响应"""
    results: List[QueryResponse]
    total_time: float
    average_time: float


class GraphRAGStatusResponse(BaseModel):
    """GraphRAG状态响应"""
    enabled: bool
    available: bool
    config: Dict[str, Any]
    dependencies: Dict[str, bool]
    statistics: Optional[Dict[str, Any]] = None


@router.post("/query", response_model=QueryResponse)
async def query_with_graphrag(request: QueryRequest = Body(...)):
    """执行RAG查询（支持GraphRAG）"""
    start_time = time.time()
    
    try:
        logger.info(f"GraphRAG查询: query='{request.query[:50]}...', use_graphrag={request.use_graphrag}")
        
        # 创建增强版混合检索配置
        hybrid_config = EnhancedHybridConfig(
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            enable_graph=request.use_graphrag,
            enable_fallback=request.enable_graph_fallback,
            timeout_seconds=5.0  # 增加超时以适应图检索
        )
        
        # 创建重排序配置
        rerank_config = None
        if request.use_rerank:
            rerank_config = RerankConfig(
                enabled=True,
                top_k_before_rerank=request.top_k_before_rerank,
                top_k_after_rerank=request.top_k_after_rerank,
                original_weight=request.rerank_original_weight,
                rerank_weight=request.rerank_weight,
                score_threshold=0.1,
                timeout_seconds=35.0,  # 从原始成功版本继承：35秒超时
                device="cpu",
                use_fp16=False
            )
        
        # 获取RAG管道
        pipeline = get_hybrid_rag_pipeline_with_graphrag(
            collection_name=request.collection_name,
            use_hybrid=True,
            use_graphrag=request.use_graphrag,
            use_rerank=request.use_rerank,
            hybrid_config=hybrid_config,
            rerank_config=rerank_config
        )
        
        # 执行查询
        result = await pipeline.query(
            query=request.query,
            top_k=request.top_k_after_rerank if request.use_rerank else request.top_k,
            score_threshold=request.score_threshold,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k_before_rerank=request.top_k_before_rerank
        )
        
        # 获取GraphRAG信息
        graphrag_info = None
        if request.use_graphrag:
            try:
                from src.core.graph.factory import GraphRAGFactory
                status = GraphRAGFactory.get_graphrag_status()
                graphrag_info = {
                    "enabled": status["enabled"],
                    "available": pipeline.graphrag_available,
                    "config": status["config"],
                    "dependencies": status["dependencies"]
                }
            except Exception as e:
                logger.warning(f"获取GraphRAG信息失败: {e}")
                graphrag_info = {"error": str(e)}
        
        # 转换文档格式
        documents = []
        for doc in result.retrieved_documents:
            documents.append(DocumentResponse(
                id=doc.id,
                text=doc.text,
                score=doc.score,
                metadata=doc.metadata,
                retrieval_types=doc.metadata.get("retrieval_types", [])
            ))
        
        total_time = time.time() - start_time
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            documents=documents,
            retrieval_time=total_time,
            metadata=result.metadata,
            graphrag_info=graphrag_info
        )
        
    except Exception as e:
        logger.error(f"GraphRAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/batch-query", response_model=BatchQueryResponse)
async def batch_query_with_graphrag(request: BatchQueryRequest = Body(...)):
    """批量执行RAG查询（支持GraphRAG）"""
    start_time = time.time()
    
    try:
        logger.info(f"批量GraphRAG查询: queries={len(request.queries)}, use_graphrag={request.use_graphrag}")
        
        # 创建增强版混合检索配置
        hybrid_config = EnhancedHybridConfig(
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            enable_graph=request.use_graphrag,
            enable_fallback=True
        )
        
        # 创建重排序配置
        rerank_config = None
        if request.use_rerank:
            rerank_config = RerankConfig(
                enabled=True,
                top_k_before_rerank=12,  # 批量查询使用固定值
                top_k_after_rerank=request.top_k,
                original_weight=request.rerank_original_weight,
                rerank_weight=request.rerank_weight,
                score_threshold=0.1,
                timeout_seconds=35.0,  # 从原始成功版本继承：35秒超时
                device="cpu",
                use_fp16=False
            )
        
        # 获取RAG管道
        pipeline = get_hybrid_rag_pipeline_with_graphrag(
            collection_name=request.collection_name,
            use_hybrid=True,
            use_graphrag=request.use_graphrag,
            use_rerank=request.use_rerank,
            hybrid_config=hybrid_config,
            rerank_config=rerank_config
        )
        
        # 执行批量查询
        results = await pipeline.batch_query(
            queries=request.queries,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # 转换结果格式
        query_responses = []
        for result in results:
            documents = []
            for doc in result.retrieved_documents:
                documents.append(DocumentResponse(
                    id=doc.id,
                    text=doc.text,
                    score=doc.score,
                    metadata=doc.metadata,
                    retrieval_types=doc.metadata.get("retrieval_types", [])
                ))
            
            query_responses.append(QueryResponse(
                query=result.query,
                answer=result.answer,
                documents=documents,
                retrieval_time=result.retrieval_time,
                metadata=result.metadata,
                graphrag_info=None  # 批量查询不包含GraphRAG信息
            ))
        
        total_time = time.time() - start_time
        
        return BatchQueryResponse(
            results=query_responses,
            total_time=total_time,
            average_time=total_time / len(request.queries)
        )
        
    except Exception as e:
        logger.error(f"批量GraphRAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量查询失败: {str(e)}")


@router.get("/status", response_model=GraphRAGStatusResponse)
async def get_graphrag_status():
    """获取GraphRAG状态"""
    try:
        from src.core.graph.factory import GraphRAGFactory
        
        status = GraphRAGFactory.get_graphrag_status()
        
        # 获取管道实例以检查可用性
        try:
            pipeline = get_hybrid_rag_pipeline_with_graphrag(
                collection_name="default",
                use_graphrag=True
            )
            available = pipeline.graphrag_available
        except:
            available = False
        
        return GraphRAGStatusResponse(
            enabled=status["enabled"],
            available=available,
            config=status["config"],
            dependencies=status["dependencies"],
            statistics=status.get("statistics")
        )
        
    except Exception as e:
        logger.error(f"获取GraphRAG状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/config")
async def get_default_config():
    """获取默认配置"""
    return {
        "hybrid_config": {
            "semantic_weight": 0.5,
            "bm25_weight": 0.3,
            "enable_graph": True,
            "enable_fallback": True,
            "timeout_seconds": 5.0
        },
        "rerank_config": {
            "original_weight": 0.4,
            "rerank_weight": 0.6,
            "top_k_before_rerank": 12,
            "top_k_after_rerank": 5
        },
        "generation_config": {
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "graphrag_config": {
            "graph_max_depth": 2,
            "enable_graph_fallback": True
        }
    }


@router.post("/test")
async def test_graphrag_integration(query: str = "测试GraphRAG集成"):
    """测试GraphRAG集成"""
    try:
        logger.info("测试GraphRAG集成")
        
        # 测试不同配置
        test_cases = [
            {"use_graphrag": False, "description": "无GraphRAG"},
            {"use_graphrag": True, "description": "有GraphRAG"}
        ]
        
        results = []
        
        for test_case in test_cases:
            try:
                # 获取管道
                pipeline = get_hybrid_rag_pipeline_with_graphrag(
                    collection_name="default",
                    use_graphrag=test_case["use_graphrag"]
                )
                
                # 执行查询
                result = await pipeline.query(
                    query=query,
                    top_k=3,
                    score_threshold=0.5
                )
                
                results.append({
                    "description": test_case["description"],
                    "success": True,
                    "retrieved_docs": len(result.retrieved_documents),
                    "graphrag_available": pipeline.graphrag_available,
                    "retrieval_time": result.retrieval_time
                })
                
            except Exception as e:
                results.append({
                    "description": test_case["description"],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "test_query": query,
            "results": results,
            "summary": {
                "total_tests": len(test_cases),
                "successful": len([r for r in results if r["success"]]),
                "failed": len([r for r in results if not r["success"]])
            }
        }
        
    except Exception as e:
        logger.error(f"GraphRAG集成测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"集成测试失败: {str(e)}")


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查GraphRAG状态
        from src.core.graph.factory import GraphRAGFactory
        status = GraphRAGFactory.get_graphrag_status()
        
        # 检查管道
        pipeline = get_hybrid_rag_pipeline_with_graphrag(
            collection_name="default",
            use_graphrag=False  # 简单检查，不使用GraphRAG
        )
        
        return {
            "status": "healthy",
            "graphrag": {
                "enabled": status["enabled"],
                "available": status["dependencies"]["neo4j_available"]
            },
            "pipeline": {
                "initialized": pipeline.hybrid_initialized,
                "collection": pipeline.collection_name
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }