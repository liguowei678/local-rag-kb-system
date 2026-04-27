"""
LLM服务类型定义
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """聊天消息"""
    role: Role
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMConfig(TypedDict, total=False):
    """LLM服务配置"""
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


class LLMProvider(str, Enum):
    """LLM提供商"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"