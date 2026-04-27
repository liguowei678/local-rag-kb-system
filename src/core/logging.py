"""
日志配置和管理
"""
import logging
import logging.config
import os
import yaml
from typing import Optional, Dict, Any
from datetime import datetime

from src.core.config import settings

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str = "openclaw"):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """设置日志上下文"""
        self.context.update(kwargs)
        return self
    
    def clear_context(self):
        """清除日志上下文"""
        self.context.clear()
    
    def _format_message(self, message: str) -> str:
        """格式化消息，添加上下文"""
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} [{context_str}]"
        return message
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.set_context(**kwargs)
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.set_context(**kwargs)
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.set_context(**kwargs)
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.set_context(**kwargs)
        self.logger.error(self._format_message(message))
    
    def exception(self, message: str, **kwargs):
        """记录异常日志（包含堆栈跟踪）"""
        self.set_context(**kwargs)
        self.logger.exception(self._format_message(message))
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.set_context(**kwargs)
        self.logger.critical(self._format_message(message))

def setup_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_json: bool = False
):
    """
    设置日志配置
    
    Args:
        config_path: 日志配置文件路径
        log_level: 日志级别（覆盖配置文件）
        enable_json: 是否启用JSON格式日志
    """
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 默认配置文件路径
    if config_path is None:
        config_path = "config/logging.yaml"
    
    try:
        # 加载YAML配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 根据环境调整配置
        if settings.APP_ENV == "production":
            # 生产环境：禁用控制台日志，只保留文件日志
            if "handlers" in config and "console" in config["handlers"]:
                config["handlers"]["console"]["level"] = "WARNING"
            
            # 启用JSON日志
            if enable_json:
                config["root"]["handlers"] = ["json_file"]
                for logger in config["loggers"].values():
                    if "handlers" in logger and "file" in logger["handlers"]:
                        logger["handlers"].remove("file")
                        logger["handlers"].append("json_file")
        
        elif settings.APP_ENV == "development":
            # 开发环境：启用详细控制台日志
            if "handlers" in config and "console" in config["handlers"]:
                config["handlers"]["console"]["level"] = "DEBUG"
        
        # 覆盖日志级别
        if log_level:
            config["root"]["level"] = log_level
            for logger in config["loggers"].values():
                logger["level"] = log_level
        
        # 应用配置
        logging.config.dictConfig(config)
        
        # 设置uvicorn日志级别
        logging.getLogger("uvicorn").setLevel(log_level or "INFO")
        logging.getLogger("uvicorn.access").setLevel(log_level or "INFO")
        
        # 记录日志配置完成
        logger = logging.getLogger("openclaw")
        logger.info(f"日志配置完成 - 环境: {settings.APP_ENV}, 级别: {log_level or '默认'}")
        
    except Exception as e:
        # 如果配置文件加载失败，使用基础配置
        logging.basicConfig(
            level=log_level or "INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("openclaw").warning(f"无法加载日志配置文件: {e}，使用基础配置")

def get_logger(name: str = "openclaw") -> StructuredLogger:
    """
    获取结构化日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        StructuredLogger实例
    """
    return StructuredLogger(name)

# 创建常用日志记录器
logger = get_logger()
api_logger = get_logger("openclaw.api")
db_logger = get_logger("openclaw.db")
cache_logger = get_logger("openclaw.cache")
access_logger = get_logger("openclaw.access")

# 性能监控装饰器
def log_execution_time(logger_name: str = "openclaw"):
    """
    记录函数执行时间的装饰器
    
    Args:
        logger_name: 使用的日志记录器名称
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger = get_logger(logger_name)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    f"函数 {func.__name__} 执行完成",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.error(
                    f"函数 {func.__name__} 执行失败",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e),
                    status="error"
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger = get_logger(logger_name)
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    f"函数 {func.__name__} 执行完成",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.error(
                    f"函数 {func.__name__} 执行失败",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e),
                    status="error"
                )
                raise
        
        if hasattr(func, "__code__") and hasattr(func.__code__, "co_flags"):
            # 检查是否是异步函数
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
        return sync_wrapper
    
    return decorator