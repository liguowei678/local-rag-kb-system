"""
文档模型
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship

from src.core.models.base import Base

class Document(Base):
    """文档表"""
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    file_path = Column(String(500), unique=True, nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, etc.
    file_size = Column(Integer, nullable=False)  # 字节数
    
    # 元数据
    document_metadata = Column(JSON, default=dict, nullable=False)
    
    # 向量相关信息
    vector_id = Column(String(100), unique=True, index=True)  # Qdrant中的向量ID
    embedding_model = Column(String(100))  # 使用的嵌入模型
    embedding_dimension = Column(Integer)  # 向量维度
    
    # 处理状态
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    processed_at = Column(DateTime, nullable=True)
    
    # 所有者
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", backref="documents")
    
    # 权限关系
    permissions = relationship("PermissionAssignment", back_populates="document", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        {"comment": "文档存储表"},
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:30]}...', owner_id={self.owner_id})>"