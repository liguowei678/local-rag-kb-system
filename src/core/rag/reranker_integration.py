"""
重排序集成模块

将Cross-Encoder重排序器集成到RAG管道中，支持：
1. 异步重排序（不阻塞事件循环）
2. 权重配置（原始分数 vs 重排序分数）
3. 降级处理（模型不可用时自动跳过）
4. 性能监控

设计原则：
- 保持现有混合检索架构不变
- 重排序作为可选增强层
- 全异步执行，避免阻塞
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.core.logging import get_logger
from .types import RetrievedDocument

logger = get_logger(__name__)


@dataclass
class RerankConfig:
    """重排序配置"""
    enabled: bool = True  # 是否启用重排序
    top_k_before_rerank: int = 20  # 重排序前的文档数量
    top_k_after_rerank: int = 5  # 重排序后的文档数量
    original_weight: float = 0.4  # 原始分数权重
    rerank_weight: float = 0.6  # 重排序分数权重
    score_threshold: float = 0.1  # 重排序分数阈值
    timeout_seconds: float = 35.0  # 重排序超时时间（从20秒增加到35秒）
    batch_size: int = 8  # 批量处理大小
    device: str = "cpu"  # 设备（cpu/cuda）
    use_fp16: bool = False  # 是否使用半精度


class RerankerIntegration:
    """重排序集成管理器"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        """
        初始化重排序集成
        
        Args:
            config: 重排序配置
        """
        self.config = config or RerankConfig()
        self.reranker = None
        self.initialized = False
        
        logger.info("重排序集成初始化",
                   enabled=self.config.enabled,
                   top_k_before=self.config.top_k_before_rerank,
                   top_k_after=self.config.top_k_after_rerank)
    
    async def initialize(self):
        """异步初始化重排序器"""
        if not self.config.enabled:
            logger.info("重排序功能已禁用")
            self.initialized = True
            return
        
        try:
            # 延迟导入，避免启动时加载模型
            from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker
            
            # 在线程中初始化重排序器（避免阻塞事件循环）
            def init_reranker():
                try:
                    reranker = CrossEncoderReranker(
                        device=self.config.device,
                        batch_size=self.config.batch_size,
                        use_fp16=self.config.use_fp16
                    )
                    return reranker
                except Exception as e:
                    logger.error("初始化重排序器失败", error=str(e))
                    return None
            
            self.reranker = await asyncio.to_thread(init_reranker)
            
            if self.reranker and self.reranker.model is not None:
                logger.info("重排序器初始化成功")
                self.initialized = True
            else:
                logger.warning("重排序器初始化失败，将禁用重排序功能")
                self.config.enabled = False
                self.initialized = True
                
        except ImportError as e:
            logger.error("导入重排序器失败", error=str(e))
            self.config.enabled = False
            self.initialized = True
        except Exception as e:
            logger.error("初始化重排序器异常", error=str(e))
            self.config.enabled = False
            self.initialized = True
    
    async def rerank_documents(self,
                              query: str,
                              documents: List[RetrievedDocument],
                              top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """
        对检索文档进行重排序
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表
            top_k: 返回的文档数量（默认使用配置）
            
        Returns:
            重排序后的文档列表
        """
        if not self.config.enabled or not self.initialized:
            logger.debug("重排序功能未启用或未初始化，返回原始结果")
            return documents[:top_k or self.config.top_k_after_rerank]
        
        if not documents:
            return []
        
        # 确定要处理的文档数量
        process_k = min(len(documents), self.config.top_k_before_rerank)
        target_k = top_k or self.config.top_k_after_rerank
        
        if process_k <= target_k:
            logger.debug(f"文档数量不足，跳过重排序: {process_k} <= {target_k}")
            return documents[:target_k]
        
        # 准备要重排序的文档
        docs_to_rerank = documents[:process_k]
        
        try:
            # 在线程中执行重排序（避免阻塞事件循环）
            def run_rerank():
                try:
                    # 转换为重排序器期望的格式
                    doc_dicts = []
                    for doc in docs_to_rerank:
                        doc_dict = {
                            'id': doc.id,
                            'text': doc.text,
                            'score': doc.score,
                            'metadata': doc.metadata,
                            'retrieval_type': getattr(doc, 'retrieval_type', 'unknown')
                        }
                        doc_dicts.append(doc_dict)
                    
                    # 执行重排序
                    reranked_results = self.reranker.rerank(
                        query=query,
                        documents=doc_dicts,
                        top_k=target_k,
                        score_threshold=self.config.score_threshold,
                        original_weight=self.config.original_weight,
                        rerank_weight=self.config.rerank_weight
                    )
                    
                    # 转换回RetrievedDocument格式
                    reranked_docs = []
                    for result in reranked_results:
                        # 更新metadata中的重排序信息
                        metadata = result.metadata.copy()
                        metadata.update({
                            'rerank_score': result.rerank_score,
                            'hybrid_score': result.hybrid_score,
                            'original_score': result.original_score,
                            'rerank_weight': self.config.rerank_weight,
                            'original_weight': self.config.original_weight
                        })
                        
                        # 创建新的RetrievedDocument
                        reranked_doc = RetrievedDocument(
                            id=str(result.id),  # 确保ID是字符串
                            text=result.text,
                            score=result.hybrid_score,  # 使用混合分数作为新分数
                            metadata=metadata,
                            source=getattr(doc, 'source', 'reranked')
                        )
                        
                        # 保留原始检索类型
                        if hasattr(doc, 'retrieval_type'):
                            reranked_doc.retrieval_type = result.retrieval_type
                        
                        reranked_docs.append(reranked_doc)
                    
                    return reranked_docs
                    
                except Exception as e:
                    logger.error("重排序执行失败", error=str(e))
                    return None
            
            # 设置超时
            reranked_docs = await asyncio.wait_for(
                asyncio.to_thread(run_rerank),
                timeout=self.config.timeout_seconds
            )
            
            if reranked_docs is None:
                logger.warning("重排序返回None，返回原始结果")
                return documents[:target_k]
            
            logger.info("重排序完成",
                       query=query[:50],
                       before_count=len(docs_to_rerank),
                       after_count=len(reranked_docs))
            
            return reranked_docs
            
        except asyncio.TimeoutError:
            logger.warning("重排序超时，返回原始结果", timeout=self.config.timeout_seconds)
            return documents[:target_k]
        except Exception as e:
            logger.error("重排序过程异常", error=str(e))
            return documents[:target_k]
    
    def get_status(self) -> Dict[str, Any]:
        """获取重排序器状态"""
        return {
            'enabled': self.config.enabled,
            'initialized': self.initialized,
            'reranker_available': self.reranker is not None and self.reranker.model is not None,
            'config': {
                'top_k_before_rerank': self.config.top_k_before_rerank,
                'top_k_after_rerank': self.config.top_k_after_rerank,
                'original_weight': self.config.original_weight,
                'rerank_weight': self.config.rerank_weight,
                'score_threshold': self.config.score_threshold,
                'device': self.config.device
            }
        }


# 全局重排序器实例
_global_reranker: Optional[RerankerIntegration] = None


async def get_reranker(config: Optional[RerankConfig] = None) -> RerankerIntegration:
    """
    获取全局重排序器实例（单例模式）
    
    Args:
        config: 重排序配置
        
    Returns:
        重排序集成管理器实例
    """
    global _global_reranker
    
    if _global_reranker is None:
        _global_reranker = RerankerIntegration(config)
        await _global_reranker.initialize()
    
    return _global_reranker


async def rerank_documents(query: str,
                          documents: List[RetrievedDocument],
                          top_k: Optional[int] = None,
                          config: Optional[RerankConfig] = None) -> List[RetrievedDocument]:
    """
    重排序文档的便捷函数
    
    Args:
        query: 查询文本
        documents: 检索到的文档列表
        top_k: 返回的文档数量
        config: 重排序配置
        
    Returns:
        重排序后的文档列表
    """
    reranker = await get_reranker(config)
    return await reranker.rerank_documents(query, documents, top_k)