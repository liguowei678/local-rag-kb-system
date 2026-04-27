"""
数据库模型定义
"""
from src.core.models.base import Base
from src.core.models.user import User
from src.core.models.document import Document
from src.core.models.permission import Permission, Role, UserRole, PermissionAssignment

__all__ = [
    "Base",
    "User",
    "Document", 
    "Permission",
    "Role",
    "UserRole",
    "PermissionAssignment",
]