"""
Collection管理器 - 多租户Collection管理
"""
import hashlib
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from src.core.config import settings
from src.core.logging import get_logger, log_execution_time
from src.core.cache import cached_with_stats

from .qdrant_client import QdrantManager, DistanceMetric

logger = get_logger("openclaw.vector_db")

class CollectionConfig:
    """Collection配置"""
    
    def __init__(
        self,
        name: str,
        vector_size: int = 1536,  # OpenAI兼容维度
        distance: DistanceMetric = DistanceMetric.COSINE,
        description: str = "",
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.vector_size = vector_size
        self.distance = distance
        self.description = description
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "distance": self.distance.value,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionConfig":
        """从字典创建"""
        config = cls(
            name=data["name"],
            vector_size=data["vector_size"],
            distance=DistanceMetric(data["distance"]),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        config.created_at = datetime.fromisoformat(data["created_at"])
        config.updated_at = datetime.fromisoformat(data["updated_at"])
        return config

class CollectionManager:
    """Collection管理器（支持多租户）"""
    
    def __init__(self):
        self.qdrant = QdrantManager()
        self._user_collections: Dict[str, Set[str]] = {}  # 用户-Collections映射
        self._collection_configs: Dict[str, CollectionConfig] = {}  # Collection配置缓存
    
    # 用户-Collection映射
    
    @log_execution_time("openclaw.vector_db")
    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查Collection是否存在
        
        Args:
            collection_name: Collection名称
            
        Returns:
            Collection是否存在
        """
        try:
            collections = await self.qdrant.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error("检查Collection存在性失败", 
                        collection_name=collection_name, 
                        error=str(e))
            return False
    
    @log_execution_time("openclaw.vector_db")
    async def get_user_collections(self, user_id: str) -> List[str]:
        """
        获取用户的Collections
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户的Collection名称列表
        """
        if user_id not in self._user_collections:
            # 从数据库或缓存加载（这里简化处理）
            self._user_collections[user_id] = set()
        
        return list(self._user_collections[user_id])
    
    @log_execution_time("openclaw.vector_db")
    async def add_user_to_collection(self, user_id: str, collection_name: str) -> bool:
        """
        将用户添加到Collection
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            
        Returns:
            是否添加成功
        """
        if user_id not in self._user_collections:
            self._user_collections[user_id] = set()
        
        self._user_collections[user_id].add(collection_name)
        
        logger.info(
            "用户添加到Collection",
            user_id=user_id,
            collection_name=collection_name
        )
        
        return True
    
    @log_execution_time("openclaw.vector_db")
    async def remove_user_from_collection(self, user_id: str, collection_name: str) -> bool:
        """
        将用户从Collection移除
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            
        Returns:
            是否移除成功
        """
        if user_id in self._user_collections:
            self._user_collections[user_id].discard(collection_name)
            
            logger.info(
                "用户从Collection移除",
                user_id=user_id,
                collection_name=collection_name
            )
        
        return True
    
    @log_execution_time("openclaw.vector_db")
    async def user_has_access(self, user_id: str, collection_name: str) -> bool:
        """
        检查用户是否有Collection访问权限
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            
        Returns:
            是否有访问权限
        """
        user_collections = await self.get_user_collections(user_id)
        return collection_name in user_collections
    
    # Collection生命周期管理
    
    @log_execution_time("openclaw.vector_db")
    async def create_user_collection(
        self,
        user_id: str,
        collection_name: str,
        vector_size: int = 1536,
        distance: DistanceMetric = DistanceMetric.COSINE,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        为用户创建Collection
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            vector_size: 向量维度
            distance: 距离度量标准
            description: 描述
            metadata: 元数据
            
        Returns:
            是否创建成功
        """
        # 生成唯一的Collection名称（避免冲突）
        unique_name = self._generate_collection_name(user_id, collection_name)
        
        # 创建Collection配置
        config = CollectionConfig(
            name=unique_name,
            vector_size=vector_size,
            distance=distance,
            description=description,
            metadata=metadata or {}
        )
        
        # 在Qdrant中创建Collection
        success = await self.qdrant.create_collection(
            collection_name=unique_name,
            vector_size=vector_size,
            distance=distance
        )
        
        if success:
            # 保存配置
            self._collection_configs[unique_name] = config
            
            # 添加用户映射
            await self.add_user_to_collection(user_id, unique_name)
            
            logger.info(
                "用户Collection创建成功",
                user_id=user_id,
                collection_name=unique_name,
                vector_size=vector_size
            )
        
        return success
    
    @log_execution_time("openclaw.vector_db")
    async def delete_user_collection(self, user_id: str, collection_name: str) -> bool:
        """
        删除用户的Collection
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            
        Returns:
            是否删除成功
        """
        # 获取完整的Collection名称
        full_collection_name = self._generate_collection_name(user_id, collection_name)
        
        # 检查用户是否有权限
        if not await self.user_has_access(user_id, full_collection_name):
            logger.warning(
                "用户无权限删除Collection",
                user_id=user_id,
                collection_name=full_collection_name
            )
            return False
        
        # 删除Collection
        success = await self.qdrant.delete_collection(full_collection_name)
        
        if success:
            # 清理缓存
            self._collection_configs.pop(full_collection_name, None)
            await self.remove_user_from_collection(user_id, full_collection_name)
            
            logger.info(
                "用户Collection删除成功",
                user_id=user_id,
                collection_name=full_collection_name
            )
        
        return success
    
    @log_execution_time("openclaw.vector_db")
    @cached_with_stats(ttl=30, key_prefix="user_collections")
    async def list_user_collections(self, user_id: str) -> List[Dict[str, Any]]:
        """
        列出用户的所有Collections（带统计信息）
        
        Args:
            user_id: 用户ID
            
        Returns:
            Collection信息列表
        """
        collections = []
        user_collection_names = await self.get_user_collections(user_id)
        
        for collection_name in user_collection_names:
            # 获取Collection信息
            info = await self.qdrant.get_collection_info(collection_name)
            if info:
                # 获取配置
                config = self._collection_configs.get(collection_name)
                if config:
                    collection_data = config.to_dict()
                    collection_data.update(info)
                    collections.append(collection_data)
        
        return collections
    
    @log_execution_time("openclaw.vector_db")
    async def get_collection_config(self, collection_name: str) -> Optional[CollectionConfig]:
        """
        获取Collection配置
        
        Args:
            collection_name: Collection名称
            
        Returns:
            Collection配置
        """
        return self._collection_configs.get(collection_name)
    
    @log_execution_time("openclaw.vector_db")
    async def update_collection_config(
        self,
        collection_name: str,
        **updates
    ) -> bool:
        """
        更新Collection配置
        
        Args:
            collection_name: Collection名称
            **updates: 更新字段
            
        Returns:
            是否更新成功
        """
        if collection_name not in self._collection_configs:
            return False
        
        config = self._collection_configs[collection_name]
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = datetime.now()
        
        logger.info(
            "Collection配置已更新",
            collection_name=collection_name,
            updates=updates
        )
        
        return True
    
    # 系统Collection管理
    
    @log_execution_time("openclaw.vector_db")
    async def create_system_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: DistanceMetric = DistanceMetric.COSINE,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        创建系统Collection（所有用户可访问）
        
        Args:
            collection_name: Collection名称
            vector_size: 向量维度
            distance: 距离度量标准
            description: 描述
            metadata: 元数据
            
        Returns:
            是否创建成功
        """
        # 创建Collection配置
        config = CollectionConfig(
            name=collection_name,
            vector_size=vector_size,
            distance=distance,
            description=description,
            metadata=metadata or {}
        )
        
        # 在Qdrant中创建Collection
        success = await self.qdrant.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance
        )
        
        if success:
            # 保存配置
            self._collection_configs[collection_name] = config
            
            logger.info(
                "系统Collection创建成功",
                collection_name=collection_name,
                vector_size=vector_size
            )
        
        return success
    
    @log_execution_time("openclaw.vector_db")
    async def grant_collection_access(
        self,
        collection_name: str,
        user_ids: List[str]
    ) -> bool:
        """
        授予用户Collection访问权限
        
        Args:
            collection_name: Collection名称
            user_ids: 用户ID列表
            
        Returns:
            是否授权成功
        """
        success = True
        
        for user_id in user_ids:
            if await self.add_user_to_collection(user_id, collection_name):
                logger.debug(
                    "用户Collection权限已授予",
                    user_id=user_id,
                    collection_name=collection_name
                )
            else:
                success = False
                logger.error(
                    "用户Collection权限授予失败",
                    user_id=user_id,
                    collection_name=collection_name
                )
        
        return success
    
    @log_execution_time("openclaw.vector_db")
    async def revoke_collection_access(
        self,
        collection_name: str,
        user_ids: List[str]
    ) -> bool:
        """
        撤销用户Collection访问权限
        
        Args:
            collection_name: Collection名称
            user_ids: 用户ID列表
            
        Returns:
            是否撤销成功
        """
        success = True
        
        for user_id in user_ids:
            if await self.remove_user_from_collection(user_id, collection_name):
                logger.debug(
                    "用户Collection权限已撤销",
                    user_id=user_id,
                    collection_name=collection_name
                )
            else:
                success = False
                logger.error(
                    "用户Collection权限撤销失败",
                    user_id=user_id,
                    collection_name=collection_name
                )
        
        return success
    
    # 工具方法
    
    def _generate_collection_name(self, user_id: str, collection_name: str) -> str:
        """
        生成唯一的Collection名称
        
        Args:
            user_id: 用户ID
            collection_name: Collection名称
            
        Returns:
            唯一的Collection名称
        """
        # 使用用户ID和Collection名称生成哈希
        name_data = f"{user_id}:{collection_name}"
        name_hash = hashlib.md5(name_data.encode()).hexdigest()[:8]
        
        return f"user_{user_id}_{collection_name}_{name_hash}"
    
    def _validate_collection_name(self, name: str) -> bool:
        """
        验证Collection名称是否有效
        
        Args:
            name: Collection名称
            
        Returns:
            是否有效
        """
        # 基本验证规则
        if not name or len(name) > 100:
            return False
        
        # 只允许字母、数字、下划线、连字符
        import re
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, name))
    
    # 批量操作
    
    @log_execution_time("openclaw.vector_db")
    async def batch_create_collections(
        self,
        collections: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        批量创建Collections
        
        Args:
            collections: Collection配置列表
            
        Returns:
            创建结果字典
        """
        results = {}
        
        for config in collections:
            collection_name = config.get("name")
            user_id = config.get("user_id")
            
            if user_id:
                # 用户Collection
                success = await self.create_user_collection(
                    user_id=user_id,
                    collection_name=collection_name,
                    vector_size=config.get("vector_size", 1536),
                    distance=DistanceMetric(config.get("distance", "Cosine")),
                    description=config.get("description", ""),
                    metadata=config.get("metadata", {})
                )
            else:
                # 系统Collection
                success = await self.create_system_collection(
                    collection_name=collection_name,
                    vector_size=config.get("vector_size", 1536),
                    distance=DistanceMetric(config.get("distance", "Cosine")),
                    description=config.get("description", ""),
                    metadata=config.get("metadata", {})
                )
            
            results[collection_name] = success
        
        return results
    
    @log_execution_time("openclaw.vector_db")
    async def cleanup_orphaned_collections(self) -> int:
        """
        清理孤立的Collections（无用户关联）
        
        Returns:
            清理的Collection数量
        """
        # 获取所有Collection
        all_collections = await self.qdrant.list_collections()
        cleaned_count = 0
        
        for collection_name in all_collections:
            # 检查是否有用户关联
            has_users = False
            for user_id, collections in self._user_collections.items():
                if collection_name in collections:
                    has_users = True
                    break
            
            # 如果是用户Collection且无用户关联，则删除
            if collection_name.startswith("user_") and not has_users:
                success = await self.qdrant.delete_collection(collection_name)
                if success:
                    self._collection_configs.pop(collection_name, None)
                    cleaned_count += 1
                    
                    logger.info(
                        "孤立Collection已清理",
                        collection_name=collection_name
                    )
        
        return cleaned_count

# 全局Collection管理器实例
collection_manager = CollectionManager()