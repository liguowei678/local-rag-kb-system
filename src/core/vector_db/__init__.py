"""
向量数据库模块
"""
from src.core.vector_db.qdrant_client import (
    QdrantManager,
    DistanceMetric,
    VectorSearchResult,
    qdrant_manager
)
from src.core.vector_db.collection_manager import (
    CollectionManager,
    CollectionConfig,
    collection_manager
)
from src.core.vector_db.embedding_service import (
    EmbeddingService,
    EmbeddingResult,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_service
)

from src.core.vector_db.search_utils import search_similar

__all__ = [
    # Qdrant客户端
    "QdrantManager",
    "DistanceMetric",
    "VectorSearchResult",
    "qdrant_manager",
    
    # Collection管理
    "CollectionManager",
    "CollectionConfig",
    "collection_manager",
    
    # 嵌入服务
    "EmbeddingService",
    "EmbeddingResult",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "embedding_service",
    
    # 常用函数
    "search_similar",
]

# 初始化向量数据库服务
def init_vector_db():
    """初始化向量数据库服务"""
    logger = get_logger("openclaw.vector_db")
    
    # 这里可以添加初始化逻辑
    # 例如：创建默认Collection、检查连接等
    
    logger.info("向量数据库服务已初始化")
    
    return {
        "qdrant": qdrant_manager,
        "collections": collection_manager,
        "embeddings": embedding_service,
    }

# 自动初始化
try:
    from src.core.logging import get_logger
    init_vector_db()
except Exception as e:
    # 初始化失败不影响导入
    pass