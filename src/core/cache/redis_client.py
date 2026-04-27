"""
Redis客户端封装
"""
import json
import pickle
from typing import Any, Optional, Union, Dict, List, Tuple
from datetime import timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("openclaw.cache")

class RedisClient:
    """Redis客户端封装类"""
    
    def __init__(self):
        self._client: Optional[Redis] = None
        self._connected = False
    
    async def connect(self) -> Redis:
        """连接到Redis服务器"""
        if self._client is None or not self._connected:
            try:
                self._client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD or None,
                    db=0,
                    decode_responses=False,  # 不自动解码，支持二进制数据
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    max_connections=50,
                )
                
                # 测试连接
                await self._client.ping()
                self._connected = True
                
                logger.info(
                    "Redis连接成功",
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT
                )
                
            except RedisError as e:
                logger.error("Redis连接失败", error=str(e))
                raise
        
        return self._client
    
    async def disconnect(self):
        """断开Redis连接"""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Redis连接已关闭")
    
    async def get_client(self) -> Redis:
        """获取Redis客户端实例"""
        return await self.connect()
    
    # 基础操作
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[Union[int, timedelta]] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        设置键值对
        
        Args:
            key: 键名
            value: 值（会自动序列化）
            expire: 过期时间（秒或timedelta）
            nx: 仅当键不存在时设置
            xx: 仅当键存在时设置
            
        Returns:
            是否设置成功
        """
        client = await self.get_client()
        
        try:
            # 序列化值
            serialized_value = self._serialize(value)
            
            # 设置过期时间
            expire_seconds = None
            if isinstance(expire, timedelta):
                expire_seconds = int(expire.total_seconds())
            elif isinstance(expire, int):
                expire_seconds = expire
            
            # 执行设置
            if nx:
                result = await client.set(
                    key, serialized_value, ex=expire_seconds, nx=True
                )
            elif xx:
                result = await client.set(
                    key, serialized_value, ex=expire_seconds, xx=True
                )
            else:
                result = await client.set(
                    key, serialized_value, ex=expire_seconds
                )
            
            return result is True
            
        except RedisError as e:
            logger.error("Redis设置失败", key=key, error=str(e))
            return False
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        获取键值
        
        Args:
            key: 键名
            default: 默认值
            
        Returns:
            反序列化的值或默认值
        """
        client = await self.get_client()
        
        try:
            value = await client.get(key)
            if value is None:
                return default
            
            return self._deserialize(value)
            
        except RedisError as e:
            logger.error("Redis获取失败", key=key, error=str(e))
            return default
    
    async def delete(self, *keys: str) -> int:
        """
        删除键
        
        Args:
            *keys: 要删除的键名
            
        Returns:
            删除的键数量
        """
        client = await self.get_client()
        
        try:
            return await client.delete(*keys)
            
        except RedisError as e:
            logger.error("Redis删除失败", keys=keys, error=str(e))
            return 0
    
    async def exists(self, *keys: str) -> int:
        """
        检查键是否存在
        
        Args:
            *keys: 要检查的键名
            
        Returns:
            存在的键数量
        """
        client = await self.get_client()
        
        try:
            return await client.exists(*keys)
            
        except RedisError as e:
            logger.error("Redis检查存在失败", keys=keys, error=str(e))
            return 0
    
    async def expire(self, key: str, time: Union[int, timedelta]) -> bool:
        """
        设置键的过期时间
        
        Args:
            key: 键名
            time: 过期时间（秒或timedelta）
            
        Returns:
            是否设置成功
        """
        client = await self.get_client()
        
        try:
            if isinstance(time, timedelta):
                time = int(time.total_seconds())
            
            return await client.expire(key, time)
            
        except RedisError as e:
            logger.error("Redis设置过期时间失败", key=key, error=str(e))
            return False
    
    async def ttl(self, key: str) -> int:
        """
        获取键的剩余生存时间
        
        Args:
            key: 键名
            
        Returns:
            剩余秒数，-1表示没有设置过期时间，-2表示键不存在
        """
        client = await self.get_client()
        
        try:
            return await client.ttl(key)
            
        except RedisError as e:
            logger.error("Redis获取TTL失败", key=key, error=str(e))
            return -2
    
    # 高级操作
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """
        递增键的值
        
        Args:
            key: 键名
            amount: 递增数量
            
        Returns:
            递增后的值
        """
        client = await self.get_client()
        
        try:
            if amount == 1:
                return await client.incr(key)
            else:
                return await client.incrby(key, amount)
                
        except RedisError as e:
            logger.error("Redis递增失败", key=key, error=str(e))
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """
        递减键的值
        
        Args:
            key: 键名
            amount: 递减数量
            
        Returns:
            递减后的值
        """
        client = await self.get_client()
        
        try:
            if amount == 1:
                return await client.decr(key)
            else:
                return await client.decrby(key, amount)
                
        except RedisError as e:
            logger.error("Redis递减失败", key=key, error=str(e))
            return 0
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """
        设置哈希字段
        
        Args:
            key: 哈希键名
            field: 字段名
            value: 字段值
            
        Returns:
            是否设置成功
        """
        client = await self.get_client()
        
        try:
            serialized_value = self._serialize(value)
            return await client.hset(key, field, serialized_value) > 0
            
        except RedisError as e:
            logger.error("Redis哈希设置失败", key=key, field=field, error=str(e))
            return False
    
    async def hget(self, key: str, field: str, default: Any = None) -> Any:
        """
        获取哈希字段
        
        Args:
            key: 哈希键名
            field: 字段名
            default: 默认值
            
        Returns:
            字段值或默认值
        """
        client = await self.get_client()
        
        try:
            value = await client.hget(key, field)
            if value is None:
                return default
            
            return self._deserialize(value)
            
        except RedisError as e:
            logger.error("Redis哈希获取失败", key=key, field=field, error=str(e))
            return default
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        获取哈希所有字段
        
        Args:
            key: 哈希键名
            
        Returns:
            字段字典
        """
        client = await self.get_client()
        
        try:
            result = await client.hgetall(key)
            return {
                field.decode(): self._deserialize(value)
                for field, value in result.items()
            }
            
        except RedisError as e:
            logger.error("Redis哈希获取全部失败", key=key, error=str(e))
            return {}
    
    # 列表操作
    
    async def lpush(self, key: str, *values: Any) -> int:
        """
        从左侧推入列表
        
        Args:
            key: 列表键名
            *values: 要推入的值
            
        Returns:
            推入后列表的长度
        """
        client = await self.get_client()
        
        try:
            serialized_values = [self._serialize(v) for v in values]
            return await client.lpush(key, *serialized_values)
            
        except RedisError as e:
            logger.error("Redis列表左侧推入失败", key=key, error=str(e))
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """
        从右侧推入列表
        
        Args:
            key: 列表键名
            *values: 要推入的值
            
        Returns:
            推入后列表的长度
        """
        client = await self.get_client()
        
        try:
            serialized_values = [self._serialize(v) for v in values]
            return await client.rpush(key, *serialized_values)
            
        except RedisError as e:
            logger.error("Redis列表右侧推入失败", key=key, error=str(e))
            return 0
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        获取列表范围
        
        Args:
            key: 列表键名
            start: 起始索引
            end: 结束索引
            
        Returns:
            列表值
        """
        client = await self.get_client()
        
        try:
            values = await client.lrange(key, start, end)
            return [self._deserialize(v) for v in values]
            
        except RedisError as e:
            logger.error("Redis列表获取范围失败", key=key, error=str(e))
            return []
    
    # 集合操作
    
    async def sadd(self, key: str, *values: Any) -> int:
        """
        添加集合元素
        
        Args:
            key: 集合键名
            *values: 要添加的值
            
        Returns:
            添加的元素数量
        """
        client = await self.get_client()
        
        try:
            serialized_values = [self._serialize(v) for v in values]
            return await client.sadd(key, *serialized_values)
            
        except RedisError as e:
            logger.error("Redis集合添加失败", key=key, error=str(e))
            return 0
    
    async def smembers(self, key: str) -> List[Any]:
        """
        获取集合所有元素
        
        Args:
            key: 集合键名
            
        Returns:
            集合元素列表
        """
        client = await self.get_client()
        
        try:
            values = await client.smembers(key)
            return [self._deserialize(v) for v in values]
            
        except RedisError as e:
            logger.error("Redis集合获取失败", key=key, error=str(e))
            return []
    
    # 序列化/反序列化
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if isinstance(value, (str, int, float, bool, type(None))):
            # 基本类型使用JSON
            return json.dumps(value).encode('utf-8')
        else:
            # 复杂类型使用pickle
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """反序列化值"""
        try:
            # 先尝试JSON
            return json.loads(value.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 失败则尝试pickle
            try:
                return pickle.loads(value)
            except pickle.PickleError:
                # 如果都失败，返回原始字节
                return value
    
    # 上下文管理器支持
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# 全局Redis客户端实例
redis_client = RedisClient()