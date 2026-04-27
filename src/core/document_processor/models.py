# src/core/document_processor/models.py
"""
文档处理模型定义
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentFormat(str, Enum):
    """文档格式枚举"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    
    @classmethod
    def from_extension(cls, extension: str) -> "DocumentFormat":
        """从文件扩展名获取文档格式"""
        extension = extension.lower()
        if extension == ".pdf":
            return cls.PDF
        elif extension == ".docx" or extension == ".doc":
            return cls.DOCX
        elif extension == ".txt" or extension == ".text":
            return cls.TXT
        elif extension == ".md" or extension == ".markdown":
            return cls.MARKDOWN
        else:
            # 默认返回TXT格式
            return cls.TXT


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    FIXED = "fixed"      # 固定大小分块
    SEMANTIC = "semantic" # 语义分块
    OVERLAP = "overlap"  # 重叠分块
    HIERARCHICAL = "hierarchical"  # 分层分块（按标题层级）


class DocumentMetadata(BaseModel):
    """文档元数据"""
    filename: str = Field(..., description="文件名")
    format: DocumentFormat = Field(..., description="文档格式")
    size: int = Field(..., description="文件大小（字节）")
    pages: Optional[int] = Field(None, description="页数（如果适用）")
    language: Optional[str] = Field(None, description="语言")
    author: Optional[str] = Field(None, description="作者")
    created_at: Optional[str] = Field(None, description="创建时间")
    modified_at: Optional[str] = Field(None, description="修改时间")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="自定义元数据")
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """文档分块"""
    text: str = Field(..., description="分块文本")
    chunk_id: str = Field(..., description="分块ID")
    index: int = Field(..., description="分块索引")
    start_char: int = Field(..., description="起始字符位置")
    end_char: int = Field(..., description="结束字符位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分块元数据")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata
        }


class ProcessingResult(BaseModel):
    """处理结果"""
    chunks: List[DocumentChunk] = Field(..., description="文档分块列表")
    metadata: DocumentMetadata = Field(..., description="文档元数据")
    processing_time: float = Field(..., description="处理时间（秒）")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="处理统计信息")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata.model_dump(),
            "processing_time": self.processing_time,
            "statistics": self.statistics
        }