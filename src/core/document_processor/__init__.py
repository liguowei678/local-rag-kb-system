# src/core/document_processor/__init__.py
"""
文档处理流水线模块

提供多格式文档解析、智能分块和预处理功能
"""

from src.core.document_processor.models import (
    DocumentFormat,
    ChunkingStrategy,
    DocumentMetadata,
    DocumentChunk,
    ProcessingResult
)

from src.core.document_processor.processor import (
    DocumentProcessor,
    document_processor
)

__all__ = [
    # 模型
    "DocumentFormat",
    "ChunkingStrategy",
    "DocumentMetadata",
    "DocumentChunk",
    "ProcessingResult",
    
    # 处理器
    "DocumentProcessor",
    "document_processor",
]