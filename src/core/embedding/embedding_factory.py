"""
嵌入服务工厂
根据配置创建和管理嵌入服务实例
"""

from typing import Dict, Any, Optional, Type
from src.core.config import settings
from src.core.logging import get_logger
from .bge_embedding import BGEEmbeddingService
from .tei_embedding import TEIEmbeddingService
from .fallback_embedding import FallbackEmbeddingService
from .types import EmbeddingConfig

logger = get_logger(__name__)


class EmbeddingFactory:
    """嵌入服务工厂"""
    
    # 注册的嵌入服务类型
    _services = {
        "tei": TEIEmbeddingService,
        "bge": BGEEmbeddingService,
        "fallback": FallbackEmbeddingService,
        "BAAI/bge-small-zh-v1.5": TEIEmbeddingService,  # TEI服务使用BGE模型
        "BAAI/bge-base-zh-v1.5": TEIEmbeddingService,
        "BAAI/bge-large-zh-v1.5": TEIEmbeddingService,
    }
    
    @classmethod
    def create(cls, model_name: Optional[str] = None, 
               config: Optional[EmbeddingConfig] = None) -> BGEEmbeddingService:
        """创建嵌入服务实例"""
        # 确定模型名称
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL
        
        # 确定配置
        if config is None:
            config = {
                "model_name": model_name,
                "device": getattr(settings, "EMBEDDING_DEVICE", "cpu"),
                "batch_size": getattr(settings, "EMBEDDING_BATCH_SIZE", 32),
                "normalize": getattr(settings, "EMBEDDING_NORMALIZE", True),
                "show_progress_bar": getattr(settings, "EMBEDDING_SHOW_PROGRESS", False)
            }
        else:
            # 确保配置中包含模型名称
            config["model_name"] = model_name
        
        # 根据配置确定提供商
        provider = getattr(settings, "EMBEDDING_PROVIDER", "tei").lower()
        
        # 查找对应的服务类
        service_class = None
        
        # 首先按提供商查找
        if provider in cls._services:
            service_class = cls._services[provider]
        else:
            # 然后按模型名查找
            for key, cls_type in cls._services.items():
                if key in model_name.lower():
                    service_class = cls_type
                    break
        
        if service_class is None:
            logger.warning("未找到匹配的嵌入服务，使用TEI作为默认",
                          requested_model=model_name,
                          provider=provider)
            service_class = TEIEmbeddingService
        
        # 创建实例
        try:
            instance = service_class(config)
            logger.info("嵌入服务创建成功",
                       model_name=model_name,
                       service_class=service_class.__name__)
            return instance
        except Exception as e:
            logger.error("创建嵌入服务失败",
                        model_name=model_name,
                        error=str(e))
            # 返回默认TEI服务
            return TEIEmbeddingService(config)
    
    @classmethod
    def register_service(cls, name: str, service_class: Type):
        """注册新的嵌入服务类型"""
        cls._services[name] = service_class
        logger.info("注册新的嵌入服务", name=name, class_name=service_class.__name__)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """列出可用的嵌入模型"""
        return {
            "tei+bge-small-zh": "TEI服务 + BAAI/bge-small-zh-v1.5 (512维)",
            "tei+bge-base-zh": "TEI服务 + BAAI/bge-base-zh-v1.5 (768维)",
            "tei+bge-large-zh": "TEI服务 + BAAI/bge-large-zh-v1.5 (1024维)",
            "bge-small-zh": "BAAI/bge-small-zh-v1.5 - 本地轻量级中文嵌入模型 (512维)",
            "fallback": "回退嵌入服务 - 无网络依赖，质量有限",
        }
    
    @classmethod
    def get_default_config(cls, model_name: str) -> EmbeddingConfig:
        """获取指定模型的默认配置"""
        configs = {
            "tei": {
                "model_name": "BAAI/bge-small-zh-v1.5",
                "base_url": "http://tei-bge:80",
                "timeout": 30,
                "max_retries": 3
            },
            "BAAI/bge-small-zh-v1.5": {
                "model_name": "BAAI/bge-small-zh-v1.5",
                "base_url": "http://tei-bge:80",
                "timeout": 30,
                "max_retries": 3
            },
            "fallback": {
                "model_name": "fallback-embedding",
                "vector_size": 512
            }
        }
        
        return configs.get(model_name, configs["tei"])


# 全局函数：获取嵌入服务
def get_embedding_service(model_name: Optional[str] = None) -> BGEEmbeddingService:
    """获取嵌入服务实例（单例模式）"""
    if not hasattr(get_embedding_service, "_instance") or get_embedding_service._instance is None:
        get_embedding_service._instance = EmbeddingFactory.create(model_name)
    
    # 如果请求的模型与当前实例不同，重新创建
    if get_embedding_service._instance:
        current_model = get_embedding_service._instance.model_name
        requested_model = model_name or settings.EMBEDDING_MODEL
        
        if current_model != requested_model:
            logger.info("切换嵌入模型",
                       current=current_model,
                       requested=requested_model)
            get_embedding_service._instance = EmbeddingFactory.create(model_name)
    
    return get_embedding_service._instance


# 初始化单例
get_embedding_service._instance = None