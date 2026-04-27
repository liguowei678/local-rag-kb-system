"""
DeepSeek LLM服务
提供DeepSeek API的LLM功能
"""

import httpx
import json
import random
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import time as time_module

from src.core.config import settings
from src.core.logging import get_logger
from .types import LLMResponse, ChatMessage, Role, LLMConfig

logger = get_logger(__name__)

# ── 熔断器常量 ──
_CIRCUIT_THRESHOLD = 3          # 连续失败多少次后熔断
_CIRCUIT_RECOVERY_SECONDS = 30  # 熔断后等多久尝试恢复
_CIRCUIT_TIMEOUT_SECONDS = 12   # 单次调用超时（秒）
_CIRCUIT_BACKOFF_BASE = 2       # 重试退避基数


class DeepSeekLLMService:
    """DeepSeek LLM服务"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key", settings.DEEPSEEK_API_KEY)
        self.base_url = self.config.get("base_url", settings.DEEPSEEK_BASE_URL)
        self.model = self.config.get("model", settings.DEEPSEEK_MODEL)
        self.temperature = self.config.get("temperature", settings.DEEPSEEK_TEMPERATURE)
        self.max_tokens = self.config.get("max_tokens", settings.DEEPSEEK_MAX_TOKENS)
        self.timeout = self.config.get("timeout", _CIRCUIT_TIMEOUT_SECONDS)
        self.max_retries = self.config.get("max_retries", 3)
        
        self._client = None
        
        # ── 熔断器状态 ──
        self._cb_state = "closed"        # closed / open / half-open
        self._cb_fail_count = 0          # 连续失败计数
        self._cb_last_open_time = 0.0    # 上次熔断开启的时间戳
        
        logger.info("DeepSeek LLM服务初始化",
                   model=self.model,
                   base_url=self.base_url)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端（延迟创建）"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_CIRCUIT_TIMEOUT_SECONDS,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client

    # ── 熔断器逻辑 ──
    def _check_circuit_breaker(self) -> bool:
        """
        检查熔断器状态。
        返回 True 表示熔断中（不调 API），False 表示正常。
        """
        if self._cb_state == "closed":
            return False  # 正常
        
        if self._cb_state == "open":
            elapsed = time_module.time() - self._cb_last_open_time
            if elapsed >= _CIRCUIT_RECOVERY_SECONDS:
                # 恢复时间到了，尝试半开
                self._cb_state = "half-open"
                logger.info("熔断器进入 half-open 状态，允许一次试探")
                return False
            return True  # 熔断中
        
        # half-open 状态：只放一次试探请求
        return False

    def _record_success(self):
        """调用成功时重置熔断器"""
        if self._cb_state != "closed":
            logger.info("熔断器已恢复，回到 closed 状态")
        self._cb_state = "closed"
        self._cb_fail_count = 0

    def _record_failure(self):
        """调用失败时累计，达到阈值则熔断"""
        self._cb_fail_count += 1
        if self._cb_fail_count >= _CIRCUIT_THRESHOLD:
            self._cb_state = "open"
            self._cb_last_open_time = time_module.time()
            logger.warning(f"连续失败 {self._cb_fail_count} 次，熔断器打开")
        elif self._cb_state == "half-open":
            # half-open 状态试探失败，立即回到 open
            self._cb_state = "open"
            self._cb_last_open_time = time_module.time()
            logger.warning("half-open 试探失败，熔断器重新打开")

    # ── 熔断器逻辑结束 ──

    async def chat(self, messages: List[ChatMessage]) -> LLMResponse:
        """聊天对话"""
        # ── 熔断器检查 ──
        circuit_open = self._check_circuit_breaker()
        if circuit_open:
            logger.warning("熔断器打开，跳过API调用，返回降级响应")
            return self._create_error_response("AI服务暂时不可用，已降级")
        
        try:
            client = await self._get_client()
            
            # 准备请求数据
            data = {
                "model": self.model,
                "messages": [msg.to_dict() for msg in messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            # 重试逻辑
            for attempt in range(self.max_retries):
                try:
                    logger.debug("发送DeepSeek请求",
                                url=url,
                                model=self.model,
                                messages_count=len(messages))
                    
                    response = await client.post(url, json=data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # 解析响应
                        choice = result["choices"][0]
                        message = choice["message"]
                        
                        llm_response = LLMResponse(
                            content=message["content"],
                            model=result["model"],
                            usage=result.get("usage", {}),
                            finish_reason=choice.get("finish_reason", "stop"),
                            metadata={
                                "id": result.get("id"),
                                "created": result.get("created"),
                                "provider": "deepseek"
                            }
                        )
                        
                        self._record_success()
                        logger.info("DeepSeek响应成功",
                                   model=self.model,
                                   tokens_used=llm_response.usage.get("total_tokens", 0))
                        return llm_response
                    
                    elif response.status_code == 429:
                        # 速率限制，等待后重试
                        wait_time = 2 ** attempt  # 指数退避
                        logger.warning("DeepSeek速率限制，等待重试",
                                      attempt=attempt + 1,
                                      wait_time=wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        self._record_failure()
                        error_text = response.text[:200]
                        logger.error("DeepSeek请求失败",
                                    status_code=response.status_code,
                                    error=error_text)
                        
                        # 如果是认证错误，不重试
                        if response.status_code in [401, 403]:
                            break
                        
                        # 其他错误，等待后重试
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        
                        # 最后一次尝试失败
                        return self._create_error_response(f"API错误: {response.status_code}")
                        
                except httpx.TimeoutException:
                    logger.warning("DeepSeek请求超时", attempt=attempt + 1)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    self._record_failure()
                    return self._create_error_response("请求超时")
                    
                except Exception as e:
                    logger.error("DeepSeek请求异常",
                                attempt=attempt + 1,
                                error=str(e))
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    self._record_failure()
                    return self._create_error_response(f"请求异常: {str(e)}")
            
            # 所有重试都失败
            self._record_failure()
            return self._create_error_response("所有重试都失败")
            
        except Exception as e:
            logger.error("DeepSeek聊天处理失败", error=str(e))
            return self._create_error_response(f"处理失败: {str(e)}")
    
    async def generate(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """生成文本（简化接口）"""
        messages = []
        
        # 添加上下文（如果有）
        if context:
            messages.append(ChatMessage(
                role=Role.SYSTEM,
                content=f"基于以下上下文回答问题：\n\n{context}"
            ))
        
        # 添加用户提示
        messages.append(ChatMessage(
            role=Role.USER,
            content=prompt
        ))
        
        return await self.chat(messages)
    
    async def chat_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """流式聊天对话"""
        try:
            client = await self._get_client()
            
            # 准备请求数据
            data = {
                "model": self.model,
                "messages": [msg.to_dict() for msg in messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            logger.debug("发送DeepSeek流式请求",
                        url=url,
                        model=self.model,
                        messages_count=len(messages))
            
            async with client.stream("POST", url, json=data, headers=headers) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_line = line[6:].strip()
                            if data_line == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data_line)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.aread()
                    logger.error("DeepSeek流式请求失败",
                                status_code=response.status_code,
                                error=error_text[:200])
                    yield f"[错误: {response.status_code}]"
                    
        except Exception as e:
            logger.error("DeepSeek流式聊天失败", error=str(e))
            yield f"[错误: {str(e)}]"
    
    def _create_error_response(self, error_message: str) -> LLMResponse:
        """创建错误响应"""
        return LLMResponse(
            content=f"抱歉，我遇到了一个错误：{error_message}",
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="error",
            metadata={
                "error": error_message,
                "provider": "deepseek",
                "fallback": True
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 发送一个简单的测试请求
            test_messages = [
                ChatMessage(role=Role.USER, content="你好")
            ]
            
            response = await self.chat(test_messages)
            
            if response.metadata.get("fallback"):
                return {
                    "status": "degraded",
                    "model": self.model,
                    "error": "API调用失败，使用回退响应"
                }
            
            return {
                "status": "healthy",
                "model": self.model,
                "provider": "deepseek"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model,
                "error": str(e)
            }
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("DeepSeek客户端已关闭")


# 全局实例
deepseek_llm_service = DeepSeekLLMService()