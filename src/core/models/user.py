"""
用户模型
"""
from sqlalchemy import Column, String, Boolean, Text, JSON
from sqlalchemy.orm import relationship

from src.core.models.base import Base

class User(Base):
    """用户表"""
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # 用户属性（用于ABAC）
    attributes = Column(JSON, default=dict, nullable=False)
    
    # 关系
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    permissions = relationship("PermissionAssignment", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"