"""
缓存服务主模块
"""
from src.core.cache import (
    # Redis客户端
    RedisClient,
    redis_client,
    
    # 策略相关
    CacheStrategy,
    CacheKeyGenerator,
    CacheStats,
    BaseCacheStrategy,
    TTLCacheStrategy,
    MemoryCacheStrategy,
    CacheManager,
    cache_manager,
    
    # 装饰器
    cached,
    cache_invalidate,
    cache_update,
    cache_ttl,
    cache_memory,
    cache_session,
    cache_long_term,
    cache_never_expire,
    cached_with_stats,
)

__all__ = [
    "RedisClient",
    "redis_client",
    "CacheStrategy",
    "CacheKeyGenerator",
    "CacheStats",
    "BaseCacheStrategy",
    "TTLCacheStrategy",
    "MemoryCacheStrategy",
    "CacheManager",
    "cache_manager",
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

# 导出常用函数
async def get_cache_stats() -> dict:
    """获取缓存统计信息"""
    return cache_manager.get_all_stats()

async def clear_cache(pattern: str = "*", strategy: str = "ttl") -> int:
    """
    清除缓存
    
    Args:
        pattern: 键模式
        strategy: 缓存策略
        
    Returns:
        清除的键数量
    """
    cache_strategy = cache_manager.get_strategy(strategy)
    return await cache_strategy.clear(pattern)

async def get_cache_value(key: str, default=None, strategy: str = "ttl") -> any:
    """
    获取缓存值
    
    Args:
        key: 缓存键
        default: 默认值
        strategy: 缓存策略
        
    Returns:
        缓存值
    """
    cache_strategy = cache_manager.get_strategy(strategy)
    return await cache_strategy.get(key, default)

async def set_cache_value(
    key: str,
    value: any,
    ttl: int = None,
    strategy: str = "ttl"
) -> bool:
    """
    设置缓存值
    
    Args:
        key: 缓存键
        value: 缓存值
        ttl: 生存时间（秒）
        strategy: 缓存策略
        
    Returns:
        是否设置成功
    """
    cache_strategy = cache_manager.get_strategy(strategy)
    
    # 如果指定了TTL，创建临时策略
    if ttl is not None:
        if strategy == "ttl":
            # 对于TTL策略，可以直接设置过期时间
            return await redis_client.set(key, value, expire=ttl)
        else:
            # 对于其他策略，需要特殊处理
            # 这里简化处理，直接使用原策略
            pass
    
    return await cache_strategy.set(key, value)