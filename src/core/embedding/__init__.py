"""
嵌入服务模块
提供统一的文本嵌入接口
"""

from .bge_embedding import BGEEmbeddingService, bge_embedding_service
from .embedding_factory import EmbeddingFactory, get_embedding_service
from .types import EmbeddingResult, BatchEmbeddingResult

__all__ = [
    "BGEEmbeddingService",
    "bge_embedding_service",
    "EmbeddingFactory",
    "get_embedding_service",
    "EmbeddingResult",
    "BatchEmbeddingResult"
]