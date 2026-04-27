"""
核心服务模块 - 文档处理、向量数据库、权限管理、文件监控
"""

# 核心服务模块 - 文档处理、向量数据库、权限管理、文件监控
# 注意：某些模块可能有依赖，导入时进行错误处理

try:
    from src.core.cache import cache_manager, CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    cache_manager = None
    CacheManager = None

try:
    from src.core.config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    settings = None

try:
    from src.core.database import database, DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    database = None
    DatabaseManager = None

# 文档处理器（无外部依赖）
from src.core.document_processor import document_processor, DocumentProcessor

# 日志（无外部依赖）
from src.core.logging import get_logger, setup_logging

try:
    from src.core.models import Base, User, Document, Permission
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    Base = User = Document = Permission = None

try:
    from src.core.vector_db import (
        QdrantManager,
        CollectionManager,
        EmbeddingService,
        embedding_service
    )
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    QdrantManager = CollectionManager = EmbeddingService = embedding_service = None

__all__ = [
    # 配置
    "settings",
    
    # 缓存
    "cache_manager",
    "CacheManager",
    
    # 数据库
    "database",
    "DatabaseManager",
    
    # 文档处理
    "document_processor",
    "DocumentProcessor",
    
    # 日志
    "get_logger",
    "setup_logging",
    
    # 模型
    "Base",
    "User",
    "Document",
    "Permission",
    
    # 向量数据库
    "QdrantManager",
    "CollectionManager",
    "EmbeddingService",
    "embedding_service",
]