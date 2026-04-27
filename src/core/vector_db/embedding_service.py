"""
嵌入服务 - 文本向量化
"""
import asyncio
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logging import get_logger, log_execution_time
from src.core.cache import cached_with_stats

logger = get_logger("openclaw.embeddings")

class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    input: List[str] = Field(..., description="要嵌入的文本列表")
    model: str = Field(default="deepseek-embedding", description="模型名称")
    encoding_format: str = Field(default="float", description="编码格式")

class EmbeddingResponse(BaseModel):
    """嵌入响应"""
    data: List[Dict[str, Any]] = Field(..., description="嵌入数据")
    model: str = Field(..., description="模型名称")
    usage: Dict[str, int] = Field(..., description="使用量统计")

class EmbeddingResult:
    """嵌入结果"""
    
    def __init__(
        self,
        text: str,
        vector: List[float],
        model: str,
        token_count: int = 0
    ):
        self.text = text
        self.vector = vector
        self.model = model
        self.token_count = token_count
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "vector_length": len(self.vector),
            "model": self.model,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
        }
    
    def __repr__(self):
        return f"EmbeddingResult(text='{self.text[:50]}...', vector_dim={len(self.vector)})"

class EmbeddingService:
    """嵌入服务"""
    
    def __init__(self):
        self._client = None
        self._model_cache: Dict[str, List[float]] = {}  # 文本哈希 -> 向量缓存
    
    async def get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @log_execution_time("openclaw.embeddings")
    @cached_with_stats(ttl=3600, key_prefix="embeddings")  # 缓存1小时
    async def embed_text(
        self,
        text: str,
        model: str = None,
        use_cache: bool = True
    ) -> Optional[EmbeddingResult]:
        """
        嵌入单个文本
        
        Args:
            text: 要嵌入的文本
            model: 模型名称，None则使用默认模型
            use_cache: 是否使用缓存
            
        Returns:
            嵌入结果
        """
        if not text or not text.strip():
            logger.warning("嵌入文本为空")
            return None
        
        # 检查缓存
        if use_cache:
            text_hash = self._get_text_hash(text, model)
            if text_hash in self._model_cache:
                logger.debug("嵌入缓存命中", text_hash=text_hash[:16])
                return EmbeddingResult(
                    text=text,
                    vector=self._model_cache[text_hash],
                    model=model or settings.DEEPSEEK_EMBEDDING_MODEL,
                    token_count=len(text.split())  # 简单估算
                )
        
        # 批量嵌入（即使只有一个文本，也使用批量接口）
        results = await self.embed_batch([text], model=model, use_cache=use_cache)
        
        if results and len(results) > 0:
            return results[0]
        
        return None
    
    @log_execution_time("openclaw.embeddings")
    async def embed_batch(
        self,
        texts: List[str],
        model: str = None,
        use_cache: bool = True,
        batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
            model: 模型名称
            use_cache: 是否使用缓存
            batch_size: 批处理大小
            
        Returns:
            嵌入结果列表
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        
        # 设置模型
        model_name = model or settings.DEEPSEEK_EMBEDDING_MODEL
        
        # 分批处理
        all_results = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_results = await self._embed_batch_internal(batch, model_name, use_cache)
            all_results.extend(batch_results)
        
        logger.info(
            "批量嵌入完成",
            total_texts=len(texts),
            valid_texts=len(valid_texts),
            results_count=len(all_results),
            model=model_name
        )
        
        return all_results
    
    async def _embed_batch_internal(
        self,
        texts: List[str],
        model: str,
        use_cache: bool
    ) -> List[EmbeddingResult]:
        """内部批量嵌入实现"""
        # 检查缓存
        cached_results = []
        uncached_texts = []
        text_to_index = {}
        
        for idx, text in enumerate(texts):
            if use_cache:
                text_hash = self._get_text_hash(text, model)
                if text_hash in self._model_cache:
                    # 使用缓存
                    cached_results.append(EmbeddingResult(
                        text=text,
                        vector=self._model_cache[text_hash],
                        model=model,
                        token_count=len(text.split())
                    ))
                    continue
            
            uncached_texts.append(text)
            text_to_index[idx] = len(uncached_texts) - 1
        
        # 如果没有需要嵌入的文本，直接返回缓存结果
        if not uncached_texts:
            return cached_results
        
        # 调用DeepSeek API
        try:
            embeddings = await self._call_deepseek_api(uncached_texts, model)
            
            # 处理结果
            new_results = []
            for idx, (text, embedding_data) in enumerate(zip(uncached_texts, embeddings)):
                if embedding_data and "embedding" in embedding_data:
                    vector = embedding_data["embedding"]
                    
                    # 缓存结果
                    if use_cache:
                        text_hash = self._get_text_hash(text, model)
                        self._model_cache[text_hash] = vector
                    
                    result = EmbeddingResult(
                        text=text,
                        vector=vector,
                        model=model,
                        token_count=embedding_data.get("token_count", len(text.split()))
                    )
                    new_results.append(result)
            
            # 合并缓存结果和新结果
            all_results = cached_results + new_results
            
            # 按原始顺序排序
            final_results = []
            result_map = {r.text: r for r in all_results}
            for text in texts:
                if text in result_map:
                    final_results.append(result_map[text])
            
            return final_results
            
        except Exception as e:
            logger.error("批量嵌入失败", error=str(e), texts_count=len(uncached_texts))
            
            # 返回缓存结果（如果有）
            return cached_results
    
    async def _call_deepseek_api(
        self,
        texts: List[str],
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        调用DeepSeek API（带重试机制）
        
        Args:
            texts: 文本列表
            model: 模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            API响应
        """
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API密钥未配置")
        
        client = await self.get_client()
        
        # 准备请求
        request_data = EmbeddingRequest(
            input=texts,
            model=model,
            encoding_format="float"
        )
        
        headers = {
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{settings.DEEPSEEK_BASE_URL}/embeddings",
                    json=request_data.model_dump(),
                    headers=headers
                )
                
                response.raise_for_status()
                
                # 解析响应
                response_data = response.json()
                embedding_response = EmbeddingResponse(**response_data)
                
                logger.debug(
                    "DeepSeek API调用成功",
                    texts_count=len(texts),
                    model=model,
                    usage=embedding_response.usage,
                    attempt=attempt + 1
                )
                
                return embedding_response.data
                
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                
                # 根据状态码决定是否重试
                if status_code in [429, 500, 502, 503, 504]:  # 可重试的错误
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # 指数退避
                        logger.warning(
                            "DeepSeek API调用失败，准备重试",
                            status_code=status_code,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_time=wait_time
                        )
                        await asyncio.sleep(wait_time)
                        continue
                
                # 不可重试的错误或达到最大重试次数
                logger.error(
                    "DeepSeek API HTTP错误",
                    status_code=status_code,
                    error=str(e),
                    attempt=attempt + 1
                )
                raise
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "DeepSeek API网络错误，准备重试",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "DeepSeek API网络错误",
                        error=str(e),
                        attempt=attempt + 1
                    )
                    raise
                
            except Exception as e:
                last_error = e
                logger.error(
                    "DeepSeek API调用异常",
                    error=str(e),
                    attempt=attempt + 1
                )
                raise
        
        # 理论上不会执行到这里
        raise last_error if last_error else Exception("未知错误")
    
    async def _embed_with_fallback(
        self,
        text: str,
        model: str,
        fallback_models: List[str] = None
    ) -> Optional[EmbeddingResult]:
        """
        带降级的嵌入（主模型失败时尝试备用模型）
        
        Args:
            text: 文本
            model: 主模型名称
            fallback_models: 备用模型列表
            
        Returns:
            嵌入结果
        """
        if fallback_models is None:
            fallback_models = ["text-embedding-ada-002", "text-embedding-3-small"]
        
        all_models = [model] + fallback_models
        
        for current_model in all_models:
            try:
                result = await self.embed_text(text, model=current_model, use_cache=True)
                if result:
                    logger.info(
                        "嵌入成功（使用备用模型）",
                        original_model=model,
                        used_model=current_model,
                        text_length=len(text)
                    )
                    return result
            except Exception as e:
                logger.warning(
                    "嵌入失败，尝试备用模型",
                    model=current_model,
                    error=str(e)
                )
                continue
        
        logger.error("所有嵌入模型都失败", text_length=len(text))
        return None
    
    @log_execution_time("openclaw.embeddings")
    async def get_embedding_dimension(self, model: str = None) -> int:
        """
        获取嵌入维度
        
        Args:
            model: 模型名称
            
        Returns:
            向量维度
        """
        model_name = model or settings.DEEPSEEK_EMBEDDING_MODEL
        
        # 常见模型的维度（这里简化处理，实际应该从API获取）
        dimension_map = {
            "deepseek-embedding": 1536,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        return dimension_map.get(model_name, 1536)
    
    @log_execution_time("openclaw.embeddings")
    async def validate_embedding(self, vector: List[float], expected_dimension: int = None) -> bool:
        """
        验证嵌入向量
        
        Args:
            vector: 向量
            expected_dimension: 期望维度
            
        Returns:
            是否有效
        """
        if not vector:
            return False
        
        # 检查维度
        if expected_dimension and len(vector) != expected_dimension:
            logger.warning(
                "向量维度不匹配",
                actual_dimension=len(vector),
                expected_dimension=expected_dimension
            )
            return False
        
        # 检查是否为有效浮点数
        try:
            for value in vector:
                float(value)
        except (ValueError, TypeError):
            return False
        
        # 检查向量是否全为零或NaN
        import math
        if all(v == 0 for v in vector) or any(math.isnan(v) for v in vector):
            logger.warning("向量无效（全零或包含NaN）")
            return False
        
        # 检查向量是否包含Inf
        if any(math.isinf(v) for v in vector):
            logger.warning("向量无效（包含Inf）")
            return False
        
        return True
    
    @log_execution_time("openclaw.embeddings")
    async def assess_embedding_quality(self, vector: List[float]) -> Dict[str, Any]:
        """
        评估嵌入向量质量
        
        Args:
            vector: 向量
            
        Returns:
            质量评估结果
        """
        import math
        import numpy as np
        
        if not vector:
            return {
                "valid": False,
                "error": "空向量",
                "score": 0.0
            }
        
        try:
            v = np.array(vector, dtype=np.float32)
            
            # 计算质量指标
            norm = np.linalg.norm(v)
            mean = np.mean(v)
            std = np.std(v)
            
            # 检查问题
            issues = []
            
            if norm == 0:
                issues.append("零向量")
            
            if np.any(np.isnan(v)):
                issues.append("包含NaN")
            
            if np.any(np.isinf(v)):
                issues.append("包含Inf")
            
            # 计算质量分数（0-1）
            if norm == 0:
                quality_score = 0.0
            else:
                # 基于范数和标准差的质量评分
                # 理想情况：范数适中，标准差适中
                norm_score = min(norm / 10.0, 1.0)  # 假设理想范数在10左右
                std_score = 1.0 - min(abs(std - 0.1) / 0.2, 1.0)  # 理想标准差在0.1左右
                
                quality_score = (norm_score + std_score) / 2.0
            
            return {
                "valid": len(issues) == 0,
                "dimension": len(vector),
                "norm": float(norm),
                "mean": float(mean),
                "std": float(std),
                "issues": issues,
                "score": float(quality_score),
                "quality": "good" if quality_score > 0.7 else "fair" if quality_score > 0.4 else "poor"
            }
            
        except Exception as e:
            logger.error("嵌入质量评估失败", error=str(e))
            return {
                "valid": False,
                "error": str(e),
                "score": 0.0
            }
    
    @log_execution_time("openclaw.embeddings")
    async def compute_similarity(
        self,
        vector1: List[float],
        vector2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        计算向量相似度
        
        Args:
            vector1: 向量1
            vector2: 向量2
            metric: 相似度度量（cosine, dot, euclidean）
            
        Returns:
            相似度分数
        """
        if not await self.validate_embedding(vector1) or not await self.validate_embedding(vector2):
            return 0.0
        
        if len(vector1) != len(vector2):
            logger.warning("向量维度不匹配，无法计算相似度")
            return 0.0
        
        import numpy as np
        
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == "cosine":
            # 余弦相似度
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        elif metric == "dot":
            # 点积相似度
            return float(np.dot(v1, v2))
        
        elif metric == "euclidean":
            # 欧几里得距离（转换为相似度）
            distance = np.linalg.norm(v1 - v2)
            # 将距离转换为相似度（距离越小，相似度越高）
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
    
    # 缓存管理
    
    def _get_text_hash(self, text: str, model: str) -> str:
        """获取文本哈希（用于缓存键）"""
        # 预处理文本以获得更好的缓存命中率
        processed_text = self.preprocess_text(text)
        data = f"{model}:{processed_text}"
        return hashlib.md5(data.encode()).hexdigest()
    
    async def clear_cache(self):
        """清空缓存"""
        cache_size = len(self._model_cache)
        self._model_cache.clear()
        
        logger.info("嵌入缓存已清空", cache_size=cache_size)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self._model_cache),
            "cache_hits": 0,  # 实际应该从缓存装饰器获取
            "cache_misses": 0,
        }
    
    async def optimize_cache(self, max_size: int = 10000) -> Dict[str, Any]:
        """
        优化缓存（LRU策略）
        
        Args:
            max_size: 最大缓存大小
            
        Returns:
            优化结果
        """
        current_size = len(self._model_cache)
        
        if current_size <= max_size:
            return {
                "action": "none",
                "current_size": current_size,
                "max_size": max_size,
                "removed": 0
            }
        
        # 简单实现：移除超过最大大小的部分
        # 在实际项目中，应该实现LRU或LFU策略
        keys_to_remove = list(self._model_cache.keys())[max_size:]
        removed_count = len(keys_to_remove)
        
        for key in keys_to_remove:
            self._model_cache.pop(key, None)
        
        logger.info(
            "缓存已优化",
            original_size=current_size,
            new_size=len(self._model_cache),
            removed=removed_count
        )
        
        return {
            "action": "trimmed",
            "original_size": current_size,
            "new_size": len(self._model_cache),
            "removed": removed_count,
            "max_size": max_size
        }
    
    async def get_cache_effectiveness(self) -> Dict[str, Any]:
        """
        获取缓存效果分析
        
        Returns:
            缓存效果分析
        """
        # 这里简化实现，实际应该跟踪缓存命中率
        cache_size = len(self._model_cache)
        
        # 估算缓存节省的API调用
        # 假设每个嵌入调用平均消耗0.1秒和0.001美元
        estimated_saved_calls = cache_size * 10  # 假设每个缓存条目平均被调用10次
        estimated_saved_time = estimated_saved_calls * 0.1  # 秒
        estimated_saved_cost = estimated_saved_calls * 0.001  # 美元
        
        return {
            "cache_size": cache_size,
            "estimated_saved_calls": estimated_saved_calls,
            "estimated_saved_time_seconds": estimated_saved_time,
            "estimated_saved_cost_usd": estimated_saved_cost,
            "effectiveness": "high" if cache_size > 1000 else "medium" if cache_size > 100 else "low"
        }
    
    # 文本预处理
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 去除多余空白
        text = " ".join(text.split())
        
        # 去除特殊字符（保留中文、英文、数字、基本标点）
        import re
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'"-]', ' ', text)
        
        # 再次去除多余空白
        text = " ".join(text.split())
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_tokens: int = 8192) -> str:
        """
        截断文本（按token估算）
        
        Args:
            text: 原始文本
            max_tokens: 最大token数
            
        Returns:
            截断后的文本
        """
        # 简单按空格分割估算token数
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        # 截断到最大token数
        truncated_words = words[:max_tokens]
        return " ".join(truncated_words)
    
    # 嵌入质量分析
    
    @log_execution_time("openclaw.embeddings")
    async def analyze_embeddings(self, vectors: List[List[float]]) -> Dict[str, Any]:
        """
        分析嵌入向量集合
        
        Args:
            vectors: 向量列表
            
        Returns:
            分析结果
        """
        import numpy as np
        
        if not vectors:
            return {"error": "空向量列表"}
        
        try:
            # 转换为numpy数组
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # 基本统计
            dimensions = vectors_array.shape[1] if len(vectors_array.shape) > 1 else 1
            count = len(vectors)
            
            # 计算质量分数
            quality_scores = []
            valid_count = 0
            
            for vector in vectors:
                quality = await self.assess_embedding_quality(vector)
                if quality["valid"]:
                    valid_count += 1
                    quality_scores.append(quality["score"])
            
            # 计算统计信息
            if quality_scores:
                quality_scores_array = np.array(quality_scores)
                mean_quality = float(np.mean(quality_scores_array))
                std_quality = float(np.std(quality_scores_array))
                min_quality = float(np.min(quality_scores_array))
                max_quality = float(np.max(quality_scores_array))
            else:
                mean_quality = std_quality = min_quality = max_quality = 0.0
            
            # 计算向量间的平均相似度
            avg_similarity = 0.0
            if len(vectors) > 1:
                similarities = []
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        sim = await self.compute_similarity(vectors[i], vectors[j], metric="cosine")
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = float(np.mean(similarities))
            
            return {
                "count": count,
                "valid_count": valid_count,
                "valid_ratio": valid_count / count if count > 0 else 0.0,
                "dimensions": dimensions,
                "quality": {
                    "mean": mean_quality,
                    "std": std_quality,
                    "min": min_quality,
                    "max": max_quality,
                },
                "avg_similarity": avg_similarity,
                "status": "good" if valid_count == count and mean_quality > 0.7 else "warning" if valid_count > count * 0.8 else "poor"
            }
            
        except Exception as e:
            logger.error("嵌入分析失败", error=str(e))
            return {
                "error": str(e),
                "status": "error"
            }
    
    # 健康检查
    
    @log_execution_time("openclaw.embeddings")
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试嵌入一个小文本
            test_text = "健康检查测试"
            result = await self.embed_text(test_text, use_cache=False)
            
            if result:
                # 验证向量
                is_valid = await self.validate_embedding(result.vector)
                
                # 评估质量
                quality = await self.assess_embedding_quality(result.vector)
                
                if is_valid:
                    return {
                        "status": "healthy",
                        "model": result.model,
                        "dimension": len(result.vector),
                        "cache_size": len(self._model_cache),
                        "quality": quality["quality"],
                        "quality_score": quality["score"],
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": "嵌入测试失败",
                        "quality_issues": quality.get("issues", []),
                    }
            else:
                return {
                    "status": "unhealthy",
                    "error": "嵌入结果为空",
                }
                
        except Exception as e:
            logger.error("嵌入服务健康检查失败", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }

# 全局嵌入服务实例
embedding_service = EmbeddingService()