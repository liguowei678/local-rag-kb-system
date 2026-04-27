"""
BGE (BAAI General Embedding) 本地嵌入服务
专门为中文优化的语义嵌入模型
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib

from src.core.config import settings
from src.core.logging import get_logger
from .types import EmbeddingResult, BatchEmbeddingResult, EmbeddingConfig

logger = get_logger(__name__)


class BGEEmbeddingService:
    """BGE本地嵌入服务"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or {}
        self.model_name = self.config.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        self.normalize = self.config.get("normalize", True)
        self.show_progress_bar = self.config.get("show_progress_bar", False)
        
        self.model = None
        self.vector_size = 512  # BGE-small的维度
        self._initialized = False
        
        logger.info("BGE嵌入服务初始化",
                   model_name=self.model_name,
                   device=self.device,
                   vector_size=self.vector_size)
    
    async def initialize(self) -> bool:
        """初始化模型（延迟加载）"""
        if self._initialized:
            return True
        
        try:
            # 延迟导入，避免不必要的依赖
            from sentence_transformers import SentenceTransformer
            
            logger.info("正在加载BGE模型", model_name=self.model_name)
            
            # 加载模型
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device
            )
            
            # 获取实际维度
            test_embedding = self.model.encode(["测试文本"])
            self.vector_size = len(test_embedding[0])
            
            self._initialized = True
            
            logger.info("BGE模型加载成功",
                       model_name=self.model_name,
                       actual_dimension=self.vector_size,
                       device=self.device)
            return True
            
        except ImportError:
            logger.error("sentence-transformers未安装，请运行: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error("加载BGE模型失败",
                        model_name=self.model_name,
                        error=str(e))
            
            # 尝试使用更小的模型或本地缓存
            logger.warning("尝试使用回退嵌入方案")
            return False
    
    async def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """生成单个文本嵌入"""
        if not text or not text.strip():
            logger.warning("文本为空，跳过嵌入")
            return None
        
        try:
            # 确保模型已初始化
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return await self._fallback_embed_text(text)
            
            # 生成嵌入
            embedding = self.model.encode(
                [text],
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )[0]
            
            vector = embedding.tolist()
            
            result = EmbeddingResult(
                vector=vector,
                text=text[:200],  # 截断用于日志
                dimension=len(vector),
                model=self.model_name,
                metadata={
                    "method": "bge",
                    "device": self.device,
                    "normalized": self.normalize
                }
            )
            
            logger.debug("BGE嵌入完成",
                        text_length=len(text),
                        vector_length=len(vector))
            return result
            
        except Exception as e:
            logger.error("BGE嵌入失败",
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
            # 确保模型已初始化
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return await self._fallback_embed_batch(texts)
            
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
            
            # 批量生成嵌入
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=self.show_progress_bar
            )
            
            vectors = [emb.tolist() for emb in embeddings]
            
            result = BatchEmbeddingResult(
                vectors=vectors,
                texts=[t[:200] for t in valid_texts],  # 截断用于日志
                dimension=self.vector_size,
                model=self.model_name,
                metadata=[
                    {
                        "method": "bge",
                        "device": self.device,
                        "normalized": self.normalize,
                        "original_length": len(t)
                    }
                    for t in valid_texts
                ]
            )
            
            logger.info("批量BGE嵌入完成",
                       texts_count=len(texts),
                       valid_count=len(valid_texts),
                       vectors_count=len(vectors))
            return result
            
        except Exception as e:
            logger.error("批量BGE嵌入失败",
                        texts_count=len(texts),
                        error=str(e))
            return await self._fallback_embed_batch(texts)
    
    async def _fallback_embed_text(self, text: str) -> EmbeddingResult:
        """BGE嵌入失败时的回退方案"""
        logger.warning("使用回退方案生成嵌入", text_length=len(text))
        
        # 基于文本哈希生成确定性向量
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
                "error": "BGE模型加载失败",
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
            "device": self.device,
            "initialized": self._initialized,
            "type": "bge",
            "description": "BAAI General Embedding - 专门为中文优化的语义嵌入模型"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return {
                        "status": "unhealthy",
                        "error": "模型初始化失败",
                        "model": self.model_name
                    }
            
            # 测试嵌入
            test_result = await self.embed_text("健康检查测试文本")
            if test_result and len(test_result.vector) == self.vector_size:
                return {
                    "status": "healthy",
                    "model": self.model_name,
                    "vector_size": self.vector_size,
                    "device": self.device
                }
            else:
                return {
                    "status": "degraded",
                    "error": "测试嵌入失败",
                    "model": self.model_name,
                    "fallback": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name
            }


# 全局实例
bge_embedding_service = BGEEmbeddingService()