"""
本地嵌入服务
使用 sentence-transformers 进行本地文本嵌入
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingService:
    """本地嵌入服务"""
    
    def __init__(self):
        from src.core.config import settings
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self.vector_size = settings.EMBEDDING_DIMENSION
        self.model = None
        self._initialized = False
        
        logger.info("本地嵌入服务初始化", 
                   model_name=self.model_name,
                   vector_size=self.vector_size)
    
    async def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return True
        
        try:
            # 延迟导入，避免不必要的依赖
            from sentence_transformers import SentenceTransformer
            
            logger.info("正在加载本地嵌入模型", model_name=self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            
            # 测试模型
            test_embedding = self.model.encode(["测试文本"])
            actual_dimension = len(test_embedding[0])
            
            logger.info("本地嵌入模型加载成功", 
                       model_name=self.model_name,
                       actual_dimension=actual_dimension)
            return True
            
        except ImportError:
            logger.error("sentence-transformers 未安装，请运行: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error("加载本地嵌入模型失败", 
                        model_name=self.model_name,
                        error=str(e))
            return False
    
    async def embed_text(self, text: str) -> Dict[str, Any]:
        """生成文本嵌入"""
        if not text:
            logger.warning("文本为空，跳过嵌入")
            return None
        
        try:
            # 确保模型已初始化
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return None
            
            # 生成嵌入
            embedding = self.model.encode([text])
            vector = embedding[0].tolist()  # 转换为列表
            
            result = {
                "vector": vector,
                "text": text[:200],  # 截断用于日志
                "dimension": len(vector),
                "model": self.model_name,
                "local": True
            }
            
            logger.debug("本地嵌入完成", 
                        text_length=len(text),
                        vector_length=len(vector))
            return result
            
        except Exception as e:
            logger.error("本地嵌入失败", 
                        text=text[:100],
                        error=str(e))
            return None
    
    async def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本嵌入"""
        if not texts:
            return []
        
        try:
            # 确保模型已初始化
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return []
            
            # 批量生成嵌入
            embeddings = self.model.encode(texts)
            
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vector = embedding.tolist()
                results.append({
                    "vector": vector,
                    "text": text[:200],
                    "dimension": len(vector),
                    "model": self.model_name,
                    "local": True,
                    "index": i
                })
            
            logger.info("批量本地嵌入完成", 
                       texts_count=len(texts),
                       results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("批量本地嵌入失败", 
                        texts_count=len(texts),
                        error=str(e))
            
            # 失败时回退到逐条处理
            return await self._fallback_embed_batch(texts)
    
    async def _fallback_embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量嵌入失败时的回退方案"""
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
                    "model": self.model_name,
                    "local": True,
                    "error": "嵌入失败"
                })
        
        logger.warning("使用回退方案完成批量嵌入", 
                      texts_count=len(texts),
                      success_count=len([e for e in embeddings if "error" not in e]))
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "vector_size": self.vector_size,
            "initialized": self._initialized,
            "type": "local"
        }


# 全局实例
local_embedding_service = LocalEmbeddingService()