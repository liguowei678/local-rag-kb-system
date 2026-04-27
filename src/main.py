"""
OpenClaw本地RAG知识库 - 主应用入口
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.core.middleware.logging_middleware import setup_middlewares
from src.api.router import api_router
from src.core.task_processor import init_task_processor, shutdown_task_processor
from src.core.storage.document_store import document_store
from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
from src.core.embedding import get_embedding_service
from src.core.llm import get_llm_service
from src.core.rag.retriever import get_retriever

# 设置日志
setup_logging(log_level=settings.LOG_LEVEL)

app = FastAPI(
    title="OpenClaw本地RAG知识库",
    description="企业级本地RAG知识库系统",
    version="0.1.0",
    docs_url="/docs" if settings.APP_ENV != "production" else None,
    redoc_url="/redoc" if settings.APP_ENV != "production" else None,
)

# 设置中间件
app = setup_middlewares(app)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "openclaw-local-rag"}

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    # 初始化所有服务
    print("🚀 正在初始化 RAG 系统服务...")
    
    # 1. 初始化任务处理器
    await init_task_processor()
    print("✅ 任务处理器已启动")
    
    # 2. 测试 Qdrant 连接
    try:
        qdrant_status = await real_qdrant_manager.collection_exists("test_connection")
        print(f"✅ Qdrant 连接正常 (测试集合存在: {qdrant_status})")
    except Exception as e:
        print(f"❌ Qdrant 连接失败: {e}")
    
    # 3. 测试 BGE 嵌入服务
    try:
        embedding_service = get_embedding_service()
        health = await embedding_service.health_check()
        print(f"✅ BGE 嵌入服务正常: {health.get('status')} (模型: {embedding_service.model_name})")
    except Exception as e:
        print(f"❌ BGE 嵌入服务失败: {e}")
    
    # 4. 测试 DeepSeek LLM 服务
    try:
        llm_service = get_llm_service()
        health = await llm_service.health_check()
        print(f"✅ DeepSeek LLM 服务正常: {health.get('status')} (模型: {llm_service.model})")
    except Exception as e:
        print(f"❌ DeepSeek LLM 服务失败: {e}")
    
    # 5. 测试 Qdrant 向量检索
    try:
        retriever = get_retriever("default")
        health = await retriever.health_check()
        print(f"✅ Qdrant 向量检索正常: {health.get('status')}")
    except Exception as e:
        print(f"❌ Qdrant 向量检索失败: {e}")
    
    # 4. 测试 PostgreSQL 连接
    try:
        stats = document_store.get_statistics()
        print(f"✅ PostgreSQL 连接正常 (文档总数: {stats.get('total', 0)})")
    except Exception as e:
        print(f"❌ PostgreSQL 连接失败: {e}")
    
    # 6. 启动监控（OpenTelemetry + Prometheus）
    try:
        from src.core.monitoring import init_monitoring
        init_monitoring(app, jaeger_endpoint="http://jaeger:4317")
        print(f"✅ 监控已启用 (Prometheus 端口 8001)")
    except Exception as e:
        print(f"❌ 监控启动失败: {e}")
    
    print("🎉 RAG 系统所有服务初始化完成！")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    # 关闭所有服务
    print("🛑 正在关闭 RAG 系统服务...")
    
    # 1. 关闭任务处理器
    await shutdown_task_processor()
    print("✅ 任务处理器已关闭")
    
    # 2. 关闭 LLM 服务
    llm_service = get_llm_service()
    await llm_service.close()
    print("✅ LLM 服务已关闭")
    
    print("👋 RAG 系统所有服务已安全关闭")

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.APP_ENV == "development",
    )