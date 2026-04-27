"""
缓存模块
"""
from src.core.cache.redis_client import RedisClient, redis_client
from src.core.cache.strategies import (
    CacheStrategy,
    CacheKeyGenerator,
    CacheStats,
    BaseCacheStrategy,
    TTLCacheStrategy,
    MemoryCacheStrategy,
    CacheManager,
    cache_manager
)
from src.core.cache.decorators import (
    cached,
    cache_invalidate,
    cache_update,
    cache_ttl,
    cache_memory,
    cache_session,
    cache_long_term,
    cache_never_expire,
    cached_with_stats
)

__all__ = [
    # Redis客户端
    "RedisClient",
    "redis_client",
    
    # 策略相关
    "CacheStrategy",
    "CacheKeyGenerator",
    "CacheStats",
    "BaseCacheStrategy",
    "TTLCacheStrategy",
    "MemoryCacheStrategy",
    "CacheManager",
    "cache_manager",
    
    # 装饰器
    "cached",
    "cache_invalidate",
    "cache_update",
    "cache_ttl",
    "cache_memory",
    "cache_session",
    "cache_long_term",
    "cache_never_expire",
    "cached_with_stats",
]

# 初始化默认缓存策略
def init_cache_strategies():
    """初始化默认缓存策略"""
    # TTL策略（使用Redis）
    ttl_strategy = TTLCacheStrategy(
        ttl=300,  # 5分钟
        max_size=10000,
        namespace="app"
    )
    
    # 内存策略（LRU）
    memory_strategy = MemoryCacheStrategy(
        ttl=60,  # 1分钟
        max_size=1000,
        namespace="memory"
    )
    
    # 注册策略
    cache_manager.register_strategy("ttl", ttl_strategy, set_as_default=True)
    cache_manager.register_strategy("memory", memory_strategy)
    
    return cache_manager

# 自动初始化
init_cache_strategies()