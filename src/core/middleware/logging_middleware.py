"""
日志中间件 - 记录HTTP请求和响应
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.logging import get_logger

class LoggingMiddleware(BaseHTTPMiddleware):
    """HTTP请求日志中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("openclaw.access")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并记录日志"""
        start_time = time.time()
        
        # 获取客户端信息
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("user-agent", "")
        
        # 记录请求开始
        self.logger.info(
            "请求开始",
            client_host=client_host,
            method=method,
            url=url,
            user_agent=user_agent[:100]  # 限制长度
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算响应时间
            process_time = time.time() - start_time
            
            # 记录响应
            self.logger.info(
                "请求完成",
                client_host=client_host,
                method=method,
                url=url,
                status_code=response.status_code,
                response_time=f"{process_time:.3f}s",
                content_length=response.headers.get("content-length", "0")
            )
            
            # 添加响应时间头
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 计算错误处理时间
            process_time = time.time() - start_time
            
            # 记录错误
            self.logger.error(
                "请求处理失败",
                client_host=client_host,
                method=method,
                url=url,
                error=str(e),
                response_time=f"{process_time:.3f}s"
            )
            
            raise

class RequestIDMiddleware(BaseHTTPMiddleware):
    """请求ID中间件 - 为每个请求生成唯一ID"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        import uuid
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """为请求添加唯一ID"""
        import uuid
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 添加到请求状态
        request.state.request_id = request_id
        
        # 处理请求
        response = await call_next(request)
        
        # 添加请求ID到响应头
        response.headers["X-Request-ID"] = request_id
        
        return response

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """关联ID中间件 - 支持跨服务追踪"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理关联ID"""
        # 从请求头获取关联ID，如果没有则生成新的
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            import uuid
            correlation_id = str(uuid.uuid4())
        
        # 添加到请求状态
        request.state.correlation_id = correlation_id
        
        # 处理请求
        response = await call_next(request)
        
        # 添加关联ID到响应头
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response

def setup_middlewares(app: ASGIApp):
    """
    设置所有中间件
    
    Args:
        app: FastAPI应用实例
    """
    # 注意中间件顺序很重要
    app.add_middleware(CorrelationIDMiddleware)  # 第一层：关联ID
    app.add_middleware(RequestIDMiddleware)      # 第二层：请求ID
    app.add_middleware(LoggingMiddleware)        # 第三层：日志记录
    
    return app