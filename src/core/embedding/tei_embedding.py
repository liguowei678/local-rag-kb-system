"""
TEI (Text Embeddings Inference) 嵌入服务
使用独立的TEI服务提供BGE嵌入
"""

import httpx
import numpy as np
from typing import List, Dict, Any, Optional
import json

from src.core.config import settings
from src.core.logging import get_logger
from .types import EmbeddingResult, BatchEmbeddingResult, EmbeddingConfig

logger = get_logger(__name__)


class TEIEmbeddingService:
    """TEI嵌入服务"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or {}
        self.base_url = self.config.get("base_url", settings.TEI_BASE_URL)
        self.model_name = self.config.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.vector_size = 512  # BGE-small的维度
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        self._client = None
        
        logger.info("TEI嵌入服务初始化",
                   base_url=self.base_url,
                   model_name=self.model_name,
                   vector_size=self.vector_size)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client
    
    async def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """生成单个文本嵌入"""
        if not text or not text.strip():
            logger.warning("文本为空，跳过嵌入")
            return None
        
        try:
            client = await self._get_client()
            url = f"{self.base_url}/embed"
            
            # TEI API格式
            data = {
                "inputs": [text],
                "truncate": True
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # 重试逻辑
            for attempt in range(self.max_retries):
                try:
                    logger.debug("发送TEI嵌入请求",
                                url=url,
                                text_length=len(text))
                    
                    response = await client.post(url, json=data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # TEI返回格式: [[向量1], [向量2], ...]
                        if isinstance(result, list) and len(result) > 0:
                            vector = result[0]
                            
                            embedding_result = EmbeddingResult(
                                vector=vector,
                                text=text[:200],
                                dimension=len(vector),
                                model=self.model_name,
                                metadata={
                                    "provider": "tei",
                                    "base_url": self.base_url,
                                    "truncated": True
                                }
                            )
                            
                            logger.debug("TEI嵌入完成",
                                        text_length=len(text),
                                        vector_length=len(vector))
                            return embedding_result
                        else:
                            logger.error("TEI响应格式错误", response=result)
                            break
                    
                    elif response.status_code == 429:
                        # 速率限制
                        wait_time = 2 ** attempt
                        logger.warning("TEI速率限制，等待重试",
                                      attempt=attempt + 1,
                                      wait_time=wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = response.text[:200]
                        logger.error("TEI请求失败",
                                    status_code=response.status_code,
                                    error=error_text)
                        
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        break
                        
                except httpx.TimeoutException:
                    logger.warning("TEI请求超时", attempt=attempt + 1)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    break
                    
                except Exception as e:
                    logger.error("TEI请求异常",
                                attempt=attempt + 1,
                                error=str(e))
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    break
            
            # 所有尝试都失败，使用回退
            logger.warning("TEI嵌入失败，使用回退方案", text=text[:100])
            return await self._fallback_embed_text(text)
            
        except Exception as e:
            logger.error("TEI嵌入处理失败",
                        text=text[:100],
                        error=str(e))
            return await self._fallback_embed_text(text)
    
    async def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """批量生成文本嵌入"""
        if not texts:
            return BatchEmbeddingResult(
                vectors=[],
                texts=[],
                dimension=self.vector_size,
                model=self.model_name
            )
        
        try:
            client = await self._get_client()
            url = f"{self.base_url}/embed"
            
            # 过滤空文本
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                logger.warning("所有文本都为空，跳过批量嵌入")
                return BatchEmbeddingResult(
                    vectors=[],
                    texts=[],
                    dimension=self.vector_size,
                    model=self.model_name
                )
            
            # TEI批量API
            data = {
                "inputs": valid_texts,
                "truncate": True
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.debug("发送TEI批量嵌入请求",
                       url=url,
                       texts_count=len(valid_texts))
            
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) == len(valid_texts):
                    vectors = result
                    
                    batch_result = BatchEmbeddingResult(
                        vectors=vectors,
                        texts=[t[:200] for t in valid_texts],
                        dimension=self.vector_size,
                        model=self.model_name,
                        metadata=[
                            {
                                "provider": "tei",
                                "base_url": self.base_url,
                                "truncated": True,
                                "original_length": len(t)
                            }
                            for t in valid_texts
                        ]
                    )
                    
                    logger.info("TEI批量嵌入完成",
                               texts_count=len(texts),
                               valid_count=len(valid_texts),
                               vectors_count=len(vectors))
                    return batch_result
                else:
                    logger.error("TEI批量响应格式错误",
                               expected=len(valid_texts),
                               actual=len(result) if isinstance(result, list) else "not list")
            
            else:
                error_text = response.text[:200]
                logger.error("TEI批量请求失败",
                           status_code=response.status_code,
                           error=error_text)
                
        except Exception as e:
            logger.error("TEI批量嵌入失败",
                        texts_count=len(texts),
                        error=str(e))
        
        # 失败时回退到逐条处理
        return await self._fallback_embed_batch(texts)
    
    async def _fallback_embed_text(self, text: str) -> EmbeddingResult:
        """TEI失败时的回退方案"""
        logger.warning("使用回退方案生成嵌入", text_length=len(text))
        
        # 基于文本哈希生成确定性向量
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        
        np.random.seed(hash_int % (2**32))
        vector = np.random.randn(self.vector_size).tolist()
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()
        
        return EmbeddingResult(
            vector=vector,
            text=text[:200],
            dimension=len(vector),
            model=self.model_name,
            metadata={
                "method": "fallback-hash",
                "provider": "tei-fallback",
                "error": "TEI服务不可用",
                "fallback": True
            }
        )
    
    async def _fallback_embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """批量嵌入失败时的回退方案"""
        logger.warning("使用回退方案进行批量嵌入", texts_count=len(texts))
        
        vectors = []
        metadata_list = []
        valid_texts = []
        
        for text in texts:
            if text and text.strip():
                result = await self._fallback_embed_text(text)
                vectors.append(result.vector)
                metadata_list.append(result.metadata)
                valid_texts.append(result.text)
        
        return BatchEmbeddingResult(
            vectors=vectors,
            texts=valid_texts,
            dimension=self.vector_size,
            model=self.model_name,
            metadata=metadata_list
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "vector_size": self.vector_size,
            "provider": "tei",
            "base_url": self.base_url,
            "description": "Text Embeddings Inference服务 + BGE模型"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self._get_client()
            url = f"{self.base_url}/health"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                except:
                    # TEI的/health端点可能返回空响应
                    health_data = {"status": "ok", "message": "TEI服务运行正常"}
                
                return {
                    "status": "healthy",
                    "model": self.model_name,
                    "provider": "tei",
                    "tei_health": health_data
                }
            else:
                return {
                    "status": "unhealthy",
                    "model": self.model_name,
                    "error": f"TEI健康检查失败: {response.status_code}",
                    "fallback": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e),
                "fallback": True
            }
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("TEI客户端已关闭")


# 全局实例
tei_embedding_service = TEIEmbeddingService()