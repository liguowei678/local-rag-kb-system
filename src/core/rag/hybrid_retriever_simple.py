"""
混合检索器（简化版）
全异步混合检索，集成语义检索和BM25
保持简洁，只实现核心功能
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.logging import get_logger
from .types import RAGQuery, RetrievedDocument
from src.core.monitoring import span, get_meter

logger = get_logger(__name__)


@dataclass
class HybridConfig:
    """混合检索配置（简化）"""
    semantic_weight: float = 0.7  # 语义检索权重
    bm25_weight: float = 0.3  # BM25检索权重
    timeout_seconds: float = 3.0  # 超时时间
    enable_fallback: bool = True  # 启用降级


class HybridRetriever:
    """混合检索器（简化版，全异步）"""
    
    def __init__(self, 
                 semantic_retriever,
                 bm25_manager,
                 config: Optional[HybridConfig] = None):
        """
        初始化混合检索器
        
        Args:
            semantic_retriever: 语义检索器实例
            bm25_manager: BM25异步管理器实例
            config: 混合检索配置
        """
        self.semantic_retriever = semantic_retriever
        self.bm25_manager = bm25_manager
        self.config = config or HybridConfig()
        
        logger.info("混合检索器初始化",
                   semantic_weight=self.config.semantic_weight,
                   bm25_weight=self.config.bm25_weight)
    
    async def retrieve(self, query: RAGQuery) -> List[RetrievedDocument]:
        """
        执行混合检索（全异步）
        
        Args:
            query: RAG查询参数
            
        Returns:
            检索到的文档列表
        """
        start_time = time.time()
        
        try:
            # 并发执行两种检索（带独立 span）
            try:
                with span("semantic_search"):
                    semantic_results = await self._retrieve_semantic_async(query)
            except Exception as e:
                logger.warning("语义检索异常", error=str(e))
                semantic_results = []
            
            try:
                with span("bm25_search"):
                    bm25_results = await self._retrieve_bm25_async(query)
            except Exception as e:
                logger.warning("BM25检索异常", error=str(e))
                bm25_results = []
            
            # 记录各渠道召回文档数到 Prometheus
            _record_retrieval_counts(
                len(semantic_results) if isinstance(semantic_results, list) else 0,
                len(bm25_results) if isinstance(bm25_results, list) else 0
            )
            
            # 融合结果
            final_results = await self._fuse_results_async(
                semantic_results, bm25_results, query.top_k
            )
            
            total_time = time.time() - start_time
            
            logger.debug("混合检索完成",
                        query=query.query[:50],
                        semantic_results=len(semantic_results),
                        bm25_results=len(bm25_results),
                        final_results=len(final_results),
                        total_time=f"{total_time:.3f}s")
            
            return final_results
            
        except asyncio.TimeoutError:
            logger.warning("混合检索超时", query=query.query[:50])
            
            # 降级到语义检索
            if self.config.enable_fallback:
                return await self._fallback_to_semantic(query)
            else:
                return []
                
        except Exception as e:
            logger.error("混合检索失败", query=query.query[:50], error=str(e))
            
            # 降级到语义检索
            if self.config.enable_fallback:
                return await self._fallback_to_semantic(query)
            else:
                return []
    
    async def _retrieve_semantic_async(self, query: RAGQuery) -> List[RetrievedDocument]:
        """异步执行语义检索"""
        try:
            # 创建新的查询对象，降低分数阈值（从0.7降到0.3）
            # 因为语义检索分数通常在0.4-0.5范围，0.7阈值会过滤掉所有结果
            from .types import RAGQuery
            low_threshold_query = RAGQuery(
                query=query.query,
                top_k=query.top_k,
                score_threshold=0.3,  # 降低阈值，让更多语义结果通过
                collection_name=query.collection_name,
                metadata_filter=query.metadata_filter
            )
            
            with span("qdrant_semantic_search"):
                return await asyncio.wait_for(
                    self.semantic_retriever.retrieve(low_threshold_query),
                    timeout=self.config.timeout_seconds
                )
        except Exception as e:
            logger.error("语义检索失败", query=query.query[:50], error=str(e))
            raise
    
    async def _retrieve_bm25_async(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """异步执行BM25检索"""
        try:
            with span("bm25_search_internal"):
                return await asyncio.wait_for(
                    self.bm25_manager.search(
                        query=query.query,
                        top_k=query.top_k * 2,  # 获取更多结果用于融合
                        collection_name=query.collection_name
                    ),
                    timeout=self.config.timeout_seconds
                )
        except Exception as e:
            logger.error("BM25检索失败", query=query.query[:50], error=str(e))
            raise


# ── Prometheus 指标（延迟初始化） ──
_semantic_doc_count = None
_bm25_doc_count = None


def _init_hybrid_metrics():
    global _semantic_doc_count, _bm25_doc_count
    if _semantic_doc_count is not None:
        return
    try:
        meter = get_meter()
        _semantic_doc_count = meter.create_histogram(
            name="rag_semantic_doc_count",
            description="语义检索每次召回的文档片段数",
            unit="1",
        )
        _bm25_doc_count = meter.create_histogram(
            name="rag_bm25_doc_count",
            description="BM25关键词检索每次召回的文档片段数",
            unit="1",
        )
    except Exception:
        pass


def _record_retrieval_counts(semantic_count: int, bm25_count: int):
    _init_hybrid_metrics()
    if _semantic_doc_count:
        _semantic_doc_count.record(semantic_count)
    if _bm25_doc_count:
        _bm25_doc_count.record(bm25_count)


# ── 类方法（在类外定义，通过猴子补丁注入） ──

async def _fuse_results_async(self,
                             semantic_results: List[RetrievedDocument],
                             bm25_results: List[Dict[str, Any]],
                             top_k: int) -> List[RetrievedDocument]:
    """异步融合结果"""
    # 如果没有结果，直接返回
    if not semantic_results and not bm25_results:
        return []
    
    # 如果只有一种结果，直接返回
    if not semantic_results:
        return await self._convert_bm25_results(bm25_results[:top_k])
    
    if not bm25_results:
        return semantic_results[:top_k]
    
    # 在线程中执行融合计算
    return await asyncio.to_thread(
        self._fuse_results_sync,
        semantic_results,
        bm25_results,
        top_k
    )


def _fuse_results_sync(self,
                      semantic_results: List[RetrievedDocument],
                      bm25_results: List[Dict[str, Any]],
                      top_k: int) -> List[RetrievedDocument]:
    """同步融合结果（在线程中执行）"""
    # 转换BM25结果
    bm25_docs = self._convert_bm25_results_sync(bm25_results)
    
    # 创建文档映射
    doc_map = {}
    
    # 处理语义结果
    for doc in semantic_results:
        doc_id = doc.id
        doc_map[doc_id] = {
            'doc': doc,
            'semantic_score': doc.score,
            'bm25_score': 0.0,
        }
    
    # 处理BM25结果
    for doc in bm25_docs:
        doc_id = doc.id
        bm25_score = doc.metadata.get('bm25_score', doc.score)
        
        if doc_id in doc_map:
            # 更新现有文档
            doc_map[doc_id]['bm25_score'] = bm25_score
            doc_map[doc_id]['doc'].metadata.update(doc.metadata)
        else:
            # 添加新文档
            doc_map[doc_id] = {
                'doc': doc,
                'semantic_score': 0.0,
                'bm25_score': bm25_score,
            }
    
    # 计算加权分数
    all_bm25_scores = [data['bm25_score'] for data in doc_map.values() if data['bm25_score'] > 0]
    max_bm25 = max(all_bm25_scores) if all_bm25_scores else 1.0
    
    for data in doc_map.values():
        semantic_score = data['semantic_score']
        bm25_score = data['bm25_score']
        
        # 语义分数归一化（假设在[0,1]范围）
        semantic_norm = semantic_score  
        
        # BM25分数归一化
        if max_bm25 > 0:
            bm25_norm = bm25_score / max_bm25
        else:
            bm25_norm = 0.0
        
        # 加权平均
        final_score = (semantic_norm * self.config.semantic_weight + 
                      bm25_norm * self.config.bm25_weight)
        
        # 更新文档
        data['doc'].score = final_score
        data['doc'].metadata['fusion_score'] = final_score
        data['doc'].metadata['semantic_score'] = semantic_score
        data['doc'].metadata['bm25_score'] = bm25_score
    
    # 排序并返回
    sorted_docs = sorted(
        [data['doc'] for data in doc_map.values()],
        key=lambda x: x.score,
        reverse=True
    )
    
    # 添加详细日志
    if len(sorted_docs) > 0:
        logger.info("混合融合详细分数",
                    query="[查询已省略]",
                    total_docs=len(sorted_docs),
                    semantic_docs=len([d for d in doc_map.values() if d['semantic_score'] > 0]),
                    bm25_docs=len([d for d in doc_map.values() if d['bm25_score'] > 0]),
                    top_3_scores=[f"{d.score:.4f}" for d in sorted_docs[:3]],
                    top_3_semantic=[f"{doc_map[d.id]['semantic_score']:.4f}" for d in sorted_docs[:3] if d.id in doc_map],
                    top_3_bm25=[f"{doc_map[d.id]['bm25_score']:.4f}" for d in sorted_docs[:3] if d.id in doc_map],
                    top_3_bm25_norm=[f"{doc_map[d.id]['bm25_score']/max_bm25:.4f}" for d in sorted_docs[:3] if d.id in doc_map and max_bm25 > 0])
    
    return sorted_docs[:top_k]


def _convert_bm25_results_sync(self, bm25_results: List[Dict[str, Any]]) -> List[RetrievedDocument]:
    """同步转换BM25结果"""
    converted = []
    
    for bm25_doc in bm25_results:
        try:
            doc = RetrievedDocument(
                id=str(bm25_doc.get('id', '')),
                text=bm25_doc.get('text', ''),
                score=float(bm25_doc.get('score', 0.0)),
                metadata=bm25_doc.get('metadata', {}),
                source=bm25_doc.get('metadata', {}).get('source', 'bm25')
            )
            
            # 添加BM25信息
            doc.metadata['bm25_score'] = float(bm25_doc.get('bm25_score', 0.0))
            doc.metadata['retrieval_type'] = 'bm25'
            
            converted.append(doc)
        except Exception as e:
            logger.warning("转换BM25结果失败", error=str(e))
            continue
    
    return converted


async def _convert_bm25_results(self, bm25_results: List[Dict[str, Any]]) -> List[RetrievedDocument]:
    """异步转换BM25结果"""
    return await asyncio.to_thread(
        self._convert_bm25_results_sync,
        bm25_results
    )


async def _fallback_to_semantic(self, query: RAGQuery) -> List[RetrievedDocument]:
    """降级到语义检索"""
    logger.info("降级到语义检索", query=query.query[:50])
    
    try:
        return await self._retrieve_semantic_async(query)
    except Exception as e:
        logger.error("降级语义检索也失败", query=query.query[:50], error=str(e))
        return []


async def health_check(self) -> Dict[str, Any]:
    """健康检查"""
    try:
        # 检查语义检索器
        semantic_health = await self.semantic_retriever.health_check()
        
        # 检查BM25管理器
        bm25_health = await self.bm25_manager.health_check()
        
        return {
            "status": "healthy",
            "semantic_retriever": semantic_health,
            "bm25_manager": bm25_health,
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "bm25_weight": self.config.bm25_weight,
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ── 注入类方法 ──
HybridRetriever._fuse_results_async = _fuse_results_async
HybridRetriever._fuse_results_sync = _fuse_results_sync
HybridRetriever._convert_bm25_results_sync = _convert_bm25_results_sync
HybridRetriever._convert_bm25_results = _convert_bm25_results
HybridRetriever._fallback_to_semantic = _fallback_to_semantic
HybridRetriever.health_check = health_check


# 工厂函数
def create_hybrid_retriever(semantic_retriever, bm25_manager, config=None):
    """创建混合检索器"""
    return HybridRetriever(semantic_retriever, bm25_manager, config)
