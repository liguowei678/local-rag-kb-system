"""
系统信息端点
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import platform

from src.core.config import settings

router = APIRouter()

@router.get("/info")
async def get_system_info():
    """获取系统信息"""
    # 获取系统信息
    system_info = {
        "name": "OpenClaw本地RAG知识库",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
        "start_time": datetime.now().isoformat(),
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        },
        "resources": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        },
        "services": {
            "database": "PostgreSQL",
            "vector_db": "Qdrant",
            "cache": "Redis",
            "embedding": "DeepSeek"
        },
        "config": {
            "log_level": settings.LOG_LEVEL,
            "max_upload_size": settings.MAX_UPLOAD_SIZE,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS
        }
    }
    
    return system_info

@router.get("/status")
async def get_service_status():
    """获取服务状态"""
    # 这里可以添加检查数据库、Redis、Qdrant连接状态
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "api": "healthy",
            "database": "connected",  # 需要实际检查
            "vector_db": "connected",  # 需要实际检查
            "cache": "connected"      # 需要实际检查
        }
    }

@router.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
            "frequency": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else 0
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        },
        "disk": {
            "total": psutil.disk_usage('/').total if hasattr(psutil, 'disk_usage') else 0,
            "used": psutil.disk_usage('/').used if hasattr(psutil, 'disk_usage') else 0,
            "free": psutil.disk_usage('/').free if hasattr(psutil, 'disk_usage') else 0,
            "percent": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        },
        "network": {
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv
        }
    }