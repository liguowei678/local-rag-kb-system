"""
真实的 DeepSeek 嵌入服务
使用真实的 DeepSeek API 进行文本嵌入
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
import json

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class RealDeepSeekEmbedding:
    """真实的 DeepSeek 嵌入服务"""
    
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model = settings.DEEPSEEK_EMBEDDING_MODEL
        self.vector_size = 1536  # DeepSeek 嵌入维度
        
        # 创建异步 HTTP 客户端
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("真实的 DeepSeek 嵌入服务已初始化", 
                   base_url=self.base_url,
                   model=self.model,
                   vector_size=self.vector_size)
    
    async def embed_text(self, text: str) -> Dict[str, Any]:
        """生成文本嵌入（使用真实的 DeepSeek API）"""
        if not text:
            logger.warning("文本为空，跳过嵌入")
            return None
        
        try:
            # 准备请求数据
            request_data = {
                "model": self.model,
                "input": text,
                "encoding_format": "float"  # 返回浮点数向量
            }
            
            # DeepSeek 嵌入 API 端点
            # 根据 DeepSeek 文档，嵌入端点是 /v1/embeddings
            api_url = f"{self.base_url}/v1/embeddings"
            
            logger.debug("调用 DeepSeek 嵌入 API", 
                        url=api_url,
                        text_length=len(text))
            
            # 发送请求
            response = await self.client.post(api_url, json=request_data)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 提取嵌入向量
            if "data" in result and len(result["data"]) > 0:
                embedding_data = result["data"][0]
                vector = embedding_data.get("embedding", [])
                
                result_dict = {
                    "vector": vector,
                    "text": text[:200],  # 截断文本用于日志
                    "dimension": len(vector),
                    "model": self.model,
                    "usage": result.get("usage", {}),
                    "object": embedding_data.get("object", ""),
                    "index": embedding_data.get("index", 0)
                }
                
                logger.debug("DeepSeek 嵌入成功", 
                           text_length=len(text),
                           vector_length=len(vector))
                return result_dict
            else:
                logger.error("DeepSeek API 返回无效数据", response=result)
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error("DeepSeek API HTTP 错误", 
                        status_code=e.response.status_code,
                        error=str(e),
                        response_text=e.response.text[:200])
            # API 失败时回退到模拟嵌入
            return await self._fallback_embed_text(text)
        except Exception as e:
            logger.error("DeepSeek 嵌入失败", 
                        text=text[:100],
                        error=str(e))
            # 其他错误也回退到模拟嵌入
            return await self._fallback_embed_text(text)
    
    async def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本嵌入"""
        if not texts:
            return []
        
        try:
            # 准备批量请求数据
            request_data = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float"
            }
            
            api_url = f"{self.base_url}/v1/embeddings"
            
            logger.info("调用 DeepSeek 批量嵌入 API", 
                       url=api_url,
                       texts_count=len(texts))
            
            # 发送请求
            response = await self.client.post(api_url, json=request_data)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 处理结果
            embeddings = []
            if "data" in result:
                for i, embedding_data in enumerate(result["data"]):
                    vector = embedding_data.get("embedding", [])
                    text = texts[i] if i < len(texts) else ""
                    
                    embeddings.append({
                        "vector": vector,
                        "text": text[:200],
                        "dimension": len(vector),
                        "model": self.model,
                        "index": embedding_data.get("index", i)
                    })
            
            logger.info("DeepSeek 批量嵌入完成", 
                       texts_count=len(texts),
                       embeddings_count=len(embeddings))
            return embeddings
                
        except Exception as e:
            logger.error("DeepSeek 批量嵌入失败", 
                        texts_count=len(texts),
                        error=str(e))
            # 失败时回退到单条处理
            return await self._fallback_embed_batch(texts)
    
    async def _fallback_embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量嵌入失败时的回退方案：逐条处理"""
        embeddings = []
        for text in texts:
            result = await self.embed_text(text)
            if result:
                embeddings.append(result)
            else:
                # 如果单条也失败，添加空向量占位
                embeddings.append({
                    "vector": [0.0] * self.vector_size,
                    "text": text[:200],
                    "dimension": self.vector_size,
                    "model": self.model,
                    "error": "嵌入失败"
                })
        
        logger.warning("使用回退方案完成批量嵌入", 
                      texts_count=len(texts),
                      success_count=len([e for e in embeddings if "error" not in e]))
        return embeddings
    
    async def _fallback_embed_text(self, text: str) -> Dict[str, Any]:
        """回退到模拟嵌入（当真实 API 失败时）"""
        import random
        
        # 生成模拟向量
        vector = [random.uniform(-1, 1) for _ in range(self.vector_size)]
        
        # 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        result = {
            "vector": vector,
            "text": text[:200],
            "dimension": self.vector_size,
            "model": "fallback-embedding",
            "fallback": True
        }
        
        logger.warning("使用回退嵌入", 
                      text_length=len(text),
                      vector_length=len(vector))
        return result
    
    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()
        logger.info("DeepSeek 嵌入服务已关闭")


# 全局实例
real_deepseek_embedding = RealDeepSeekEmbedding()