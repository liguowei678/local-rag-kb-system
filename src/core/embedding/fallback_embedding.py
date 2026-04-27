"""
回退嵌入服务
当BGE模型无法加载时使用
"""

import numpy as np
import hashlib
from typing import List, Dict, Any, Optional

from src.core.config import settings
from src.core.logging import get_logger
from .types import EmbeddingResult, BatchEmbeddingResult, EmbeddingConfig

logger = get_logger(__name__)


class FallbackEmbeddingService:
    """回退嵌入服务（无网络依赖）"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or {}
        self.model_name = "fallback-embedding"
        self.vector_size = settings.EMBEDDING_DIMENSION
        
        logger.info("回退嵌入服务初始化",
                   model_name=self.model_name,
                   vector_size=self.vector_size)
    
    async def initialize(self) -> bool:
        """初始化（总是成功）"""
        logger.info("回退嵌入服务已就绪", model_name=self.model_name)
        return True
    
    def _text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量"""
        if not text:
            return [0.0] * self.vector_size
        
        # 使用文本哈希生成确定性向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        
        np.random.seed(hash_int % (2**32))
        vector = np.random.randn(self.vector_size).tolist()
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()
        
        return vector
    
    async def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """生成单个文本嵌入"""
        if not text or not text.strip():
            logger.warning("文本为空，跳过嵌入")
            return None
        
        try:
            vector = self._text_to_vector(text)
            
            result = EmbeddingResult(
                vector=vector,
                text=text[:200],
                dimension=len(vector),
                model=self.model_name,
                metadata={
                    "method": "fallback-hash",
                    "fallback": True,
                    "warning": "使用回退嵌入服务，语义搜索质量有限"
                }
            )
            
            logger.debug("回退嵌入完成",
                        text_length=len(text),
                        vector_length=len(vector))
            return result
            
        except Exception as e:
            logger.error("回退嵌入失败",
                        text=text[:100],
                        error=str(e))
            
            # 最终回退：零向量
            vector = [0.0] * self.vector_size
            return EmbeddingResult(
                vector=vector,
                text=text[:200],
                dimension=len(vector),
                model=self.model_name,
                metadata={
                    "method": "zero-vector",
                    "fallback": True,
                    "error": str(e)
                }
            )
    
    async def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """批量生成文本嵌入"""
        if not texts:
            return BatchEmbeddingResult(
                vectors=[],
                texts=[],
                dimension=self.vector_size,
                model=self.model_name
            )
        
        vectors = []
        metadata_list = []
        valid_texts = []
        
        for text in texts:
            if text and text.strip():
                result = await self.embed_text(text)
                if result:
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
            "type": "fallback",
            "description": "回退嵌入服务，基于文本哈希生成向量",
            "warning": "语义搜索质量有限，建议安装BGE模型"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "model": self.model_name,
            "vector_size": self.vector_size,
            "fallback": True,
            "warning": "使用回退嵌入服务，语义搜索质量有限"
        }


# 全局实例
fallback_embedding_service = FallbackEmbeddingService()