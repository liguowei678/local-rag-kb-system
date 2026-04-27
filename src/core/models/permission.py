"""
权限模型（RBAC + ABAC）
"""
from sqlalchemy import Column, String, Text, Boolean, Integer, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.core.models.base import Base

class Permission(Base):
    """权限定义表"""
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    resource_type = Column(String(50), nullable=False)  # document, namespace, system, etc.
    action = Column(String(50), nullable=False)  # read, write, delete, admin, etc.
    
    # ABAC属性条件
    conditions = Column(JSON, default=dict, nullable=False)  # 属性条件
    
    def __repr__(self):
        return f"<Permission(id={self.id}, name='{self.name}', action='{self.action}')>"

class Role(Base):
    """角色表"""
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    is_system = Column(Boolean, default=False)  # 是否为系统角色
    
    # 关系
    permissions = relationship("PermissionAssignment", back_populates="role", cascade="all, delete-orphan")
    users = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"

class UserRole(Base):
    """用户-角色关联表"""
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    assigned_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)  # 角色过期时间
    
    # 关系
    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")
    
    def __repr__(self):
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"

class PermissionAssignment(Base):
    """权限分配表（支持直接分配和通过角色分配）"""
    permission_id = Column(Integer, ForeignKey("permissions.id"), primary_key=True)
    
    # 分配目标（三选一）
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=True)
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=True)  # 预留群组支持
    
    # 资源目标
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(Integer, nullable=True)  # 具体资源ID，null表示所有该类型资源
    
    # 分配属性
    assigned_at = Column(DateTime, default=func.now(), nullable=False)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=True)  # 分配者
    expires_at = Column(DateTime, nullable=True)  # 权限过期时间
    
    # 覆盖条件（可以覆盖权限本身的conditions）
    override_conditions = Column(JSON, default=dict, nullable=False)
    
    # 关系
    permission = relationship("Permission")
    user = relationship("User", foreign_keys=[user_id], back_populates="permissions")
    role = relationship("Role", foreign_keys=[role_id], back_populates="permissions")
    document = relationship("Document", foreign_keys=[resource_id], 
                          primaryjoin="and_(PermissionAssignment.resource_type=='document', "
                                     "PermissionAssignment.resource_id==Document.id)",
                          back_populates="permissions")
    
    # 复合索引
    __table_args__ = (
        {"comment": "权限分配表，支持RBAC和ABAC混合模型"},
    )
    
    def __repr__(self):
        target = f"user:{self.user_id}" if self.user_id else f"role:{self.role_id}"
        resource = f"{self.resource_type}:{self.resource_id}" if self.resource_id else f"{self.resource_type}:all"
        return f"<PermissionAssignment(permission={self.permission_id}, target={target}, resource={resource})>"