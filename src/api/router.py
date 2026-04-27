"""
API路由定义
"""
from fastapi import APIRouter

from .endpoints import system, documents, embeddings
from .routes import simple_semantic_qa
# 导入GraphRAG增强问答端点
from .endpoints import qa_hybrid_rerank_with_graph
# 导入GraphRAG端点
from .routers import graphrag, graphrag_incremental
# 导入GraphRAG文档上传端点
from .endpoints import documents_graphrag

api_router = APIRouter()

# 注册系统端点
api_router.include_router(
    system.router,
    prefix="/system",
    tags=["系统"]
)

# 注册文档端点
api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["文档"]
)

# 注册嵌入端点
api_router.include_router(
    embeddings.router,
    prefix="/embeddings",
    tags=["嵌入"]
)

# 注册GraphRAG增强问答端点（完整功能：混合检索+重排序+GraphRAG）
api_router.include_router(
    qa_hybrid_rerank_with_graph.router,
    prefix="",  # 路由已经在router中定义了前缀
    tags=["问答（完整功能）"]
)

# 注册简单语义检索端点
api_router.include_router(
    simple_semantic_qa.router,
    prefix="/qa",
    tags=["简单语义检索"]
)

# 注册GraphRAG端点
api_router.include_router(
    graphrag.router,
    prefix="/graphrag",  # 添加/graphrag前缀
    tags=["GraphRAG"]
)

# 注册增量GraphRAG端点（使用独立前缀避免冲突）
api_router.include_router(
    graphrag_incremental.router,
    prefix="/graphrag/incremental",  # 使用独立前缀
    tags=["GraphRAG-增量"]
)

# 注册GraphRAG文档上传端点（使用不同前缀避免冲突）
api_router.include_router(
    documents_graphrag.router,
    prefix="/documents/graphrag",  # 使用/documents/graphrag避免冲突
    tags=["文档-GraphRAG"]
)

@api_router.get("/")
async def root():
    """API根端点"""
    return {
        "service": "OpenClaw本地RAG知识库",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "system": "/api/v1/system",
            "documents": "/api/v1/documents",
            "documents_graphrag": "/api/v1/documents/graphrag",
            "embeddings": "/api/v1/embeddings",
            "qa": "/api/v1/qa",
            "qa_full": "/api/v1/qa-hybrid-graphrag",  # 完整功能端点
            "graphrag": "/api/v1/graphrag",
            "graphrag_incremental": "/api/v1/graphrag/incremental",
            "health": "/health"
        }
    }