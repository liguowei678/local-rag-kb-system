"""
缓存装饰器
"""
import asyncio
import functools
import inspect
from typing import Any, Callable, Optional, Union, Dict, List
from datetime import timedelta

from src.core.cache.strategies import (
    CacheKeyGenerator,
    cache_manager,
    BaseCacheStrategy,
    TTLCacheStrategy,
    MemoryCacheStrategy
)
from src.core.logging import get_logger, log_execution_time

logger = get_logger("openclaw.cache")

def cached(
    ttl: Optional[Union[int, timedelta]] = 300,
    key_prefix: str = "cache",
    strategy: str = "ttl",
    cache_condition: Optional[Callable[[Any], bool]] = None,
    cache_exceptions: bool = False
):
    """
    缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒或timedelta）
        key_prefix: 缓存键前缀
        strategy: 缓存策略名称
        cache_condition: 缓存条件函数，返回True则缓存
        cache_exceptions: 是否缓存异常结果
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        # 判断是否是异步函数
        is_async = inspect.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate_key(
                prefix=key_prefix,
                func_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            # 获取缓存策略
            cache_strategy = cache_manager.get_strategy(strategy)
            
            # 尝试从缓存获取
            try:
                cached_value = await cache_strategy.get(cache_key)
                if cached_value is not None:
                    logger.debug(
                        "缓存命中",
                        function=func.__name__,
                        cache_key=cache_key,
                        strategy=strategy
                    )
                    return cached_value
            except Exception as e:
                logger.warning(
                    "缓存获取失败，继续执行函数",
                    function=func.__name__,
                    cache_key=cache_key,
                    error=str(e)
                )
            
            # 缓存未命中，执行函数
            logger.debug(
                "缓存未命中，执行函数",
                function=func.__name__,
                cache_key=cache_key
            )
            
            try:
                result = await func(*args, **kwargs)
                
                # 检查缓存条件
                should_cache = True
                if cache_condition is not None:
                    should_cache = cache_condition(result)
                
                if should_cache:
                    # 设置缓存
                    try:
                        await cache_strategy.set(cache_key, result)
                        logger.debug(
                            "结果已缓存",
                            function=func.__name__,
                            cache_key=cache_key,
                            ttl=ttl
                        )
                    except Exception as e:
                        logger.warning(
                            "缓存设置失败",
                            function=func.__name__,
                            cache_key=cache_key,
                            error=str(e)
                        )
                
                return result
                
            except Exception as e:
                logger.error(
                    "函数执行失败",
                    function=func.__name__,
                    error=str(e)
                )
                
                if cache_exceptions:
                    # 缓存异常结果
                    try:
                        await cache_strategy.set(cache_key, e)
                    except Exception as cache_error:
                        logger.warning(
                            "异常结果缓存失败",
                            function=func.__name__,
                            error=str(cache_error)
                        )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate_key(
                prefix=key_prefix,
                func_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            # 获取缓存策略
            cache_strategy = cache_manager.get_strategy(strategy)
            
            # 尝试从缓存获取
            try:
                cached_value = asyncio.run(cache_strategy.get(cache_key))
                if cached_value is not None:
                    logger.debug(
                        "缓存命中",
                        function=func.__name__,
                        cache_key=cache_key,
                        strategy=strategy
                    )
                    return cached_value
            except Exception as e:
                logger.warning(
                    "缓存获取失败，继续执行函数",
                    function=func.__name__,
                    cache_key=cache_key,
                    error=str(e)
                )
            
            # 缓存未命中，执行函数
            logger.debug(
                "缓存未命中，执行函数",
                function=func.__name__,
                cache_key=cache_key
            )
            
            try:
                result = func(*args, **kwargs)
                
                # 检查缓存条件
                should_cache = True
                if cache_condition is not None:
                    should_cache = cache_condition(result)
                
                if should_cache:
                    # 设置缓存
                    try:
                        asyncio.run(cache_strategy.set(cache_key, result))
                        logger.debug(
                            "结果已缓存",
                            function=func.__name__,
                            cache_key=cache_key,
                            ttl=ttl
                        )
                    except Exception as e:
                        logger.warning(
                            "缓存设置失败",
                            function=func.__name__,
                            cache_key=cache_key,
                            error=str(e)
                        )
                
                return result
                
            except Exception as e:
                logger.error(
                    "函数执行失败",
                    function=func.__name__,
                    error=str(e)
                )
                
                if cache_exceptions:
                    # 缓存异常结果
                    try:
                        asyncio.run(cache_strategy.set(cache_key, e))
                    except Exception as cache_error:
                        logger.warning(
                            "异常结果缓存失败",
                            function=func.__name__,
                            error=str(cache_error)
                        )
                
                raise
        
        if is_async:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_invalidate(
    key_prefix: str,
    strategy: str = "ttl",
    pattern: str = "*"
):
    """
    缓存失效装饰器
    
    Args:
        key_prefix: 缓存键前缀
        strategy: 缓存策略名称
        pattern: 键模式（用于批量删除）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 使缓存失效
            try:
                cache_strategy = cache_manager.get_strategy(strategy)
                cache_pattern = CacheKeyGenerator.generate_pattern(key_prefix)
                
                deleted = await cache_strategy.clear(cache_pattern)
                
                if deleted > 0:
                    logger.info(
                        "缓存已失效",
                        function=func.__name__,
                        pattern=cache_pattern,
                        deleted_count=deleted
                    )
                    
            except Exception as e:
                logger.warning(
                    "缓存失效失败",
                    function=func.__name__,
                    error=str(e)
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 执行函数
            result = func(*args, **kwargs)
            
            # 使缓存失效
            try:
                cache_strategy = cache_manager.get_strategy(strategy)
                cache_pattern = CacheKeyGenerator.generate_pattern(key_prefix)
                
                deleted = asyncio.run(cache_strategy.clear(cache_pattern))
                
                if deleted > 0:
                    logger.info(
                        "缓存已失效",
                        function=func.__name__,
                        pattern=cache_pattern,
                        deleted_count=deleted
                    )
                    
            except Exception as e:
                logger.warning(
                    "缓存失效失败",
                    function=func.__name__,
                    error=str(e)
                )
            
            return result
        
        if is_async:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_update(
    key_prefix: str,
    strategy: str = "ttl",
    ttl: Optional[Union[int, timedelta]] = None
):
    """
    缓存更新装饰器（先更新缓存，再执行函数）
    
    Args:
        key_prefix: 缓存键前缀
        strategy: 缓存策略名称
        ttl: 新的生存时间
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate_key(
                prefix=key_prefix,
                func_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 更新缓存
            try:
                cache_strategy = cache_manager.get_strategy(strategy)
                
                # 删除旧缓存
                await cache_strategy.delete(cache_key)
                
                # 设置新缓存
                await cache_strategy.set(cache_key, result)
                
                logger.debug(
                    "缓存已更新",
                    function=func.__name__,
                    cache_key=cache_key
                )
                
            except Exception as e:
                logger.warning(
                    "缓存更新失败",
                    function=func.__name__,
                    error=str(e)
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = CacheKeyGenerator.generate_key(
                prefix=key_prefix,
                func_name=func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 更新缓存
            try:
                cache_strategy = cache_manager.get_strategy(strategy)
                
                # 删除旧缓存
                asyncio.run(cache_strategy.delete(cache_key))
                
                # 设置新缓存
                asyncio.run(cache_strategy.set(cache_key, result))
                
                logger.debug(
                    "缓存已更新",
                    function=func.__name__,
                    cache_key=cache_key
                )
                
            except Exception as e:
                logger.warning(
                    "缓存更新失败",
                    function=func.__name__,
                    error=str(e)
                )
            
            return result
        
        if is_async:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# 常用缓存装饰器快捷方式

def cache_ttl(ttl: Union[int, timedelta] = 300):
    """TTL缓存装饰器快捷方式"""
    return cached(ttl=ttl, strategy="ttl")

def cache_memory(ttl: Optional[Union[int, timedelta]] = None):
    """内存缓存装饰器快捷方式"""
    return cached(ttl=ttl, strategy="memory")

def cache_session():
    """会话缓存装饰器（短时间缓存）"""
    return cached(ttl=60, strategy="ttl", key_prefix="session")

def cache_long_term():
    """长期缓存装饰器"""
    return cached(ttl=3600, strategy="ttl", key_prefix="long_term")

def cache_never_expire():
    """永不过期缓存装饰器"""
    return cached(ttl=None, strategy="ttl", key_prefix="permanent")

# 性能监控缓存装饰器

def cached_with_stats(
    ttl: Optional[Union[int, timedelta]] = 300,
    key_prefix: str = "cache",
    strategy: str = "ttl"
):
    """
    带统计信息的缓存装饰器
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        cached_func = cached(
            ttl=ttl,
            key_prefix=key_prefix,
            strategy=strategy
        )(func)
        
        return log_execution_time("openclaw.cache")(cached_func)
    
    return decorator