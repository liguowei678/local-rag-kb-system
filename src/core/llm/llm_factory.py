"""
LLM服务工厂
根据配置创建和管理LLM服务实例
"""

from typing import Dict, Any, Optional, Type
from src.core.config import settings
from src.core.logging import get_logger
from .deepseek_llm import DeepSeekLLMService
from .types import LLMConfig, LLMProvider

logger = get_logger(__name__)


class LLMFactory:
    """LLM服务工厂"""

    # 注册的LLM服务类型
    _services = {
        LLMProvider.DEEPSEEK: DeepSeekLLMService,
        "deepseek": DeepSeekLLMService,
        "deepseek-chat": DeepSeekLLMService,
    }

    @classmethod
    def create(cls, provider: Optional[str] = None,
               config: Optional[LLMConfig] = None) -> DeepSeekLLMService:
        """创建LLM服务实例"""
        # 确定提供商
        if provider is None:
            provider = "deepseek"  # 默认使用DeepSeek

        # 确定配置
        if config is None:
            config = {
                "api_key": settings.DEEPSEEK_API_KEY,
                "base_url": settings.DEEPSEEK_BASE_URL,
                "model": settings.DEEPSEEK_MODEL,
                "temperature": settings.DEEPSEEK_TEMPERATURE,
                "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
                "timeout": 30,
                "max_retries": 3
            }
        else:
            # 确保配置中包含必要字段
            if "api_key" not in config:
                config["api_key"] = settings.DEEPSEEK_API_KEY
            if "model" not in config:
                config["model"] = settings.DEEPSEEK_MODEL

        # 查找对应的服务类
        provider_lower = provider.lower()
        service_class = None

        for key, cls_type in cls._services.items():
            if key.lower() == provider_lower:
                service_class = cls_type
                break

        if service_class is None:
            logger.warning("未找到匹配的LLM服务，使用DeepSeek作为默认",
                          requested_provider=provider)
            service_class = DeepSeekLLMService

        # 创建实例
        try:
            instance = service_class(config)
            logger.info("LLM服务创建成功",
                       provider=provider,
                       model=config.get("model"),
                       service_class=service_class.__name__)
            return instance
        except Exception as e:
            logger.error("创建LLM服务失败",
                        provider=provider,
                        error=str(e))
            # 返回默认DeepSeek服务
            return DeepSeekLLMService(config)

    @classmethod
    def register_service(cls, name: str, service_class: Type):
        """注册新的LLM服务类型"""
        cls._services[name] = service_class
        logger.info("注册新的LLM服务", name=name, class_name=service_class.__name__)

    @classmethod
    def list_available_providers(cls) -> Dict[str, str]:
        """列出可用的LLM提供商"""
        return {
            "deepseek": "DeepSeek - 深度求索AI，强大的中文LLM",
            "openai": "OpenAI - GPT系列模型",
            "anthropic": "Anthropic - Claude系列模型",
            "local": "本地模型 - 如Llama、Qwen等本地部署模型",
        }

    @classmethod
    def get_default_config(cls, provider: str) -> LLMConfig:
        """获取指定提供商的默认配置"""
        configs = {
            "deepseek": {
                "api_key": settings.DEEPSEEK_API_KEY,
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 30,
                "max_retries": 3
            },
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30,
                "max_retries": 3
            }
        }

        return configs.get(provider.lower(), configs["deepseek"])


# 全局函数：获取LLM服务
def get_llm_service(provider: Optional[str] = None) -> DeepSeekLLMService:
    """获取LLM服务实例（单例模式）"""
    # 确保_instance属性存在
    if not hasattr(get_llm_service, "_instance"):
        get_llm_service._instance = None

    # 如果实例不存在或需要重新创建
    requested_provider = provider or "deepseek"

    if get_llm_service._instance is None:
        get_llm_service._instance = LLMFactory.create(requested_provider)

    return get_llm_service._instance
