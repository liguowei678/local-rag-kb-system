"""
缓存策略定义
"""
import asyncio
import time
from typing import Any, Optional, Callable, Dict, List, Tuple, Union
from datetime import timedelta
from enum import Enum
import hashlib
import json

from src.core.logging import get_logger

logger = get_logger("openclaw.cache")

class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最不经常使用
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 生存时间
    WRITE_THROUGH = "write_through"  # 直写
    WRITE_BACK = "write_back"  # 回写

class CacheKeyGenerator:
    """缓存键生成器"""
    
    @staticmethod
    def generate_key(
        prefix: str,
        func_name: str,
        args: tuple,
        kwargs: dict,
        version: str = "v1"
    ) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 键前缀
            func_name: 函数名
            args: 位置参数
            kwargs: 关键字参数
            version: 缓存版本
            
        Returns:
            缓存键
        """
        # 序列化参数
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        
        # 生成哈希
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"{prefix}:{version}:{key_hash}"
    
    @staticmethod
    def generate_pattern(prefix: str, version: str = "v1") -> str:
        """
        生成键模式（用于批量删除）
        
        Args:
            prefix: 键前缀
            version: 缓存版本
            
        Returns:
            键模式
        """
        return f"{prefix}:{version}:*"

class CacheStats:
    """缓存统计"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
    
    def hit(self):
        """记录命中"""
        self.hits += 1
    
    def miss(self):
        """记录未命中"""
        self.misses += 1
    
    def set(self):
        """记录设置"""
        self.sets += 1
    
    def delete(self):
        """记录删除"""
        self.deletes += 1
    
    def error(self):
        """记录错误"""
        self.errors += 1
    
    @property
    def total(self) -> int:
        """总请求数"""
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total == 0:
            return 0.0
        return self.hits / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total": self.total,
            "hit_rate": self.hit_rate,
        }

class BaseCacheStrategy:
    """基础缓存策略"""
    
    def __init__(
        self,
        ttl: Optional[Union[int, timedelta]] = None,
        max_size: Optional[int] = None,
        namespace: str = "default"
    ):
        """
        初始化
        
        Args:
            ttl: 生存时间（秒或timedelta）
            max_size: 最大缓存大小
            namespace: 命名空间
        """
        self.ttl = ttl
        self.max_size = max_size
        self.namespace = namespace
        self.stats = CacheStats()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        raise NotImplementedError
    
    async def set(self, key: str, value: Any) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            是否设置成功
        """
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        raise NotImplementedError
    
    async def clear(self, pattern: str = "*") -> int:
        """
        清除缓存
        
        Args:
            pattern: 键模式
            
        Returns:
            清除的键数量
        """
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.to_dict()

class TTLCacheStrategy(BaseCacheStrategy):
    """生存时间缓存策略"""
    
    def __init__(
        self,
        ttl: Union[int, timedelta] = 300,  # 默认5分钟
        max_size: Optional[int] = 1000,
        namespace: str = "ttl"
    ):
        super().__init__(ttl=ttl, max_size=max_size, namespace=namespace)
        
        from src.core.cache.redis_client import redis_client
        self.redis = redis_client
    
    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        try:
            full_key = f"{self.namespace}:{key}"
            value = await self.redis.get(full_key, default)
            
            if value is not default:
                self.stats.hit()
            else:
                self.stats.miss()
            
            return value
            
        except Exception as e:
            self.stats.error()
            logger.error("TTL缓存获取失败", key=key, error=str(e))
            return default
    
    async def set(self, key: str, value: Any) -> bool:
        """设置缓存值"""
        try:
            full_key = f"{self.namespace}:{key}"
            success = await self.redis.set(full_key, value, expire=self.ttl)
            
            if success:
                self.stats.set()
            
            return success
            
        except Exception as e:
            self.stats.error()
            logger.error("TTL缓存设置失败", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            full_key = f"{self.namespace}:{key}"
            deleted = await self.redis.delete(full_key) > 0
            
            if deleted:
                self.stats.delete()
            
            return deleted
            
        except Exception as e:
            self.stats.error()
            logger.error("TTL缓存删除失败", key=key, error=str(e))
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """清除缓存"""
        # 注意：Redis的KEYS命令在生产环境慎用
        # 这里使用SCAN命令更安全，但为简化使用模式匹配
        try:
            from src.core.cache.redis_client import redis_client
            client = await redis_client.get_client()
            
            # 构建完整模式
            full_pattern = f"{self.namespace}:{pattern}"
            
            # 获取匹配的键
            keys = []
            cursor = 0
            while True:
                cursor, found_keys = await client.scan(
                    cursor=cursor, match=full_pattern, count=100
                )
                keys.extend(found_keys)
                if cursor == 0:
                    break
            
            # 删除键
            if keys:
                deleted = await client.delete(*keys)
                self.stats.deletes += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            self.stats.error()
            logger.error("TTL缓存清除失败", pattern=pattern, error=str(e))
            return 0

class MemoryCacheStrategy(BaseCacheStrategy):
    """内存缓存策略（LRU实现）"""
    
    def __init__(
        self,
        ttl: Optional[Union[int, timedelta]] = None,
        max_size: int = 1000,
        namespace: str = "memory"
    ):
        super().__init__(ttl=ttl, max_size=max_size, namespace=namespace)
        
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        
        # 转换为秒数
        self._ttl_seconds = None
        if isinstance(ttl, timedelta):
            self._ttl_seconds = ttl.total_seconds()
        elif isinstance(ttl, int):
            self._ttl_seconds = ttl
    
    def _clean_expired(self):
        """清理过期缓存"""
        if self._ttl_seconds is None:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self._ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def _evict_if_needed(self):
        """如果需要则驱逐缓存（LRU）"""
        if self.max_size is not None and len(self._cache) > self.max_size:
            # 找到最久未访问的键
            if self._access_times:
                lru_key = min(self._access_times, key=self._access_times.get)
                self._cache.pop(lru_key, None)
                self._access_times.pop(lru_key, None)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        self._clean_expired()
        
        full_key = f"{self.namespace}:{key}"
        
        if full_key in self._cache:
            value, _ = self._cache[full_key]
            self._access_times[full_key] = time.time()
            self.stats.hit()
            return value
        else:
            self.stats.miss()
            return default
    
    async def set(self, key: str, value: Any) -> bool:
        """设置缓存值"""
        self._clean_expired()
        
        full_key = f"{self.namespace}:{key}"
        self._cache[full_key] = (value, time.time())
        self._access_times[full_key] = time.time()
        
        self._evict_if_needed()
        self.stats.set()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        full_key = f"{self.namespace}:{key}"
        
        if full_key in self._cache:
            self._cache.pop(full_key, None)
            self._access_times.pop(full_key, None)
            self.stats.delete()
            return True
        
        return False
    
    async def clear(self, pattern: str = "*") -> int:
        """清除缓存"""
        if pattern == "*":
            deleted = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self.stats.deletes += deleted
            return deleted
        else:
            # 简单的模式匹配（只支持*通配符）
            deleted = 0
            full_pattern = f"{self.namespace}:{pattern}"
            
            keys_to_delete = []
            for key in list(self._cache.keys()):
                if self._match_pattern(key, full_pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                deleted += 1
            
            self.stats.deletes += deleted
            return deleted
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """简单模式匹配"""
        if pattern == "*":
            return True
        
        # 将模式转换为正则表达式
        import re
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(regex_pattern, key))

class CacheManager:
    """缓存管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies: Dict[str, BaseCacheStrategy] = {}
            cls._instance._default_strategy = "ttl"
        return cls._instance
    
    def register_strategy(
        self,
        name: str,
        strategy: BaseCacheStrategy,
        set_as_default: bool = False
    ):
        """
        注册缓存策略
        
        Args:
            name: 策略名称
            strategy: 策略实例
            set_as_default: 是否设置为默认策略
        """
        self._strategies[name] = strategy
        
        if set_as_default:
            self._default_strategy = name
        
        logger.info(f"缓存策略已注册: {name}")
    
    def get_strategy(self, name: Optional[str] = None) -> BaseCacheStrategy:
        """
        获取缓存策略
        
        Args:
            name: 策略名称，None表示使用默认策略
            
        Returns:
            缓存策略实例
        """
        if name is None:
            name = self._default_strategy
        
        if name not in self._strategies:
            raise ValueError(f"缓存策略未找到: {name}")
        
        return self._strategies[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略的统计信息"""
        return {
            name: strategy.get_stats()
            for name, strategy in self._strategies.items()
        }

# 全局缓存管理器实例
cache_manager = CacheManager()