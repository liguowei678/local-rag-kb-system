"""
RAG管道（支持混合检索 + GraphRAG + 重排序）
增强版RAG管道，集成语义检索、BM25关键词检索、GraphRAG图检索和Cross-Encoder重排序
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

from src.core.config import settings
from src.core.logging import get_logger
from .retriever import get_retriever
from .generator import get_generator
from .types import RAGQuery, RAGResult, RetrievedDocument
from .hybrid_retriever_simple import create_hybrid_retriever, HybridConfig
from dataclasses import dataclass


@dataclass
class EnhancedHybridConfig:
    """增强版混合检索配置"""
    semantic_weight: float = 0.5
    bm25_weight: float = 0.3
    timeout_seconds: float = 8.0
    enable_fallback: bool = True
    enable_graph: bool = True


from .bm25_async_manager import get_bm25_manager
from .reranker_integration import RerankConfig, get_reranker, rerank_documents
from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
from src.core.monitoring import span, record_request, record_error, get_tracer

# GraphRAG相关导入
from src.core.graph.factory import GraphRAGFactory
from src.core.graph.retriever import GraphRetriever
from src.core.graph.independent_retriever import IndependentGraphRAGRetriever, IndependentGraphRAGConfig

logger = get_logger(__name__)


class HybridRAGPipelineWithGraphRAG:
    """RAG管道（支持混合检索 + GraphRAG + 重排序）"""
    
    def __init__(self, 
                 collection_name: str = "default", 
                 use_hybrid: bool = True,
                 use_graphrag: bool = True,
                 use_rerank: bool = False,
                 hybrid_config: Optional[EnhancedHybridConfig] = None,
                 rerank_config: Optional[RerankConfig] = None):
        """
        初始化RAG管道（支持GraphRAG）
        
        Args:
            collection_name: 集合名称
            use_hybrid: 是否使用混合检索
            use_graphrag: 是否使用GraphRAG
            use_rerank: 是否使用重排序
            hybrid_config: 增强版混合检索配置
            rerank_config: 重排序配置
        """
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid
        self.use_graphrag = use_graphrag
        self.use_rerank = use_rerank
        
        # 检查GraphRAG是否可用
        self.graphrag_available = False
        self.independent_graph_retriever = None  # 只使用独立GraphRAG检索器
        
        if use_graphrag:
            try:
                # 检查GraphRAG配置
                config = GraphRAGFactory.create_config()
                self.graphrag_available = config.graphrag_enabled
                
                if self.graphrag_available:
                    logger.info("GraphRAG已启用，将使用独立GraphRAG检索器")
                    
                    # 创建独立GraphRAG检索器（按照新要求）
                    semantic_retriever = get_retriever(self.collection_name)
                    self.independent_graph_retriever = IndependentGraphRAGRetriever(
                        config=config,
                        semantic_retriever=semantic_retriever,
                        retriever_config=IndependentGraphRAGConfig(
                            vector_top_k=2,  # top_k=2片段
                            max_hop_depth=3,  # 最多3跳
                            timeout_seconds=10.0
                        )
                    )
                    
                    # 测试连接
                    status = GraphRAGFactory.get_graphrag_status()
                    if status["dependencies"]["neo4j_available"]:
                        logger.info("Neo4j连接可用，独立GraphRAG检索器已初始化")
                    else:
                        logger.warning("Neo4j连接不可用，GraphRAG功能受限")
                else:
                    logger.warning("GraphRAG未启用，将跳过图检索")
                    
            except Exception as e:
                logger.warning(f"GraphRAG初始化失败: {e}")
                self.graphrag_available = False
        
        # 初始化检索器
        if use_hybrid:
            # 延迟初始化混合检索器（在异步上下文中）
            self.retriever = None
            self.hybrid_initialized = False
            self.hybrid_config = hybrid_config or EnhancedHybridConfig(
                semantic_weight=0.5,
                bm25_weight=0.3,
                enable_graph=self.graphrag_available
            )
        else:
            # 使用简单语义检索器
            self.retriever = get_retriever(collection_name)
            self.hybrid_initialized = True
        
        # 初始化重排序器配置
        if use_rerank:
            self.rerank_config = rerank_config or RerankConfig(
                enabled=True,
                top_k_before_rerank=20,
                top_k_after_rerank=5,
                original_weight=0.4,
                rerank_weight=0.6,
                score_threshold=0.1,
                timeout_seconds=35.0,  # 从原始成功版本继承：35秒超时
                device="cpu",
                use_fp16=False
            )
        else:
            self.rerank_config = None
        
        # 初始化生成器
        self.generator = get_generator()
        
        logger.info(f"RAG管道初始化完成: hybrid={use_hybrid}, graphrag={use_graphrag}, rerank={use_rerank}")
    
    def _calculate_adaptive_threshold(self, document_scores: List[float], 
                                    base_threshold: float = 0.3) -> float:
        """
        检索后自适应阈值（在重排序前应用）
        
        基于混合检索结果分数分布，在给重排序之前过滤低质量文档
        基准阈值使用0.3（与BM25一致）
        """
        if not document_scores:
            return 0.1  # 无结果时大幅降低阈值
        
        avg_score = sum(document_scores) / len(document_scores)
        
        # 自适应逻辑：基于结果质量调整阈值
        if avg_score > 0.6:
            # 高质量结果：适度提高阈值，过滤噪声
            return min(0.7, base_threshold + 0.15)
        elif avg_score > 0.4:
            # 中等质量：保持基准阈值
            return base_threshold
        else:
            # 低质量结果：降低阈值，避免过滤掉所有结果
            return max(0.05, base_threshold - 0.25)
    
    def _calculate_dynamic_threshold(self, document_scores: List[float], 
                                   base_threshold: float = 0.3) -> float:
        """
        基于文档最终分数的最简动态阈值
        
        在重排序后调用，基于hybrid_score计算
        基准阈值使用0.3（与BM25和语义检索一致）
        """
        if not document_scores:
            return 0.2  # 无结果时降低阈值（从0.5调整到0.2）
        
        avg_score = sum(document_scores) / len(document_scores)
        
        # 核心逻辑：结果质量高就收紧，质量低就放宽
        if avg_score > 0.6:
            # 高质量结果：适度提高阈值，过滤噪声
            return min(0.7, base_threshold + 0.15)
        elif avg_score > 0.4:
            # 中等质量：保持基准阈值
            return base_threshold
        else:
            # 低质量结果：降低阈值，避免过滤掉所有结果
            return max(0.05, base_threshold - 0.25)
    
    async def _initialize_hybrid_retriever(self):
        """初始化混合检索器（异步）"""
        if self.hybrid_initialized:
            return
        
        logger.info("初始化混合检索器...")
        
        # 获取语义检索器
        semantic_retriever = get_retriever(self.collection_name)
        
        bm25_manager = None
        try:
            # 获取BM25管理器
            from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
            bm25_manager = await get_bm25_manager(real_qdrant_manager)
            logger.info("BM25管理器获取成功")
        except Exception as e:
            logger.warning(f"BM25管理器获取失败，将使用纯语义检索: {e}")
        
        try:
            # 创建混合检索器
            if bm25_manager:
                # 使用简单混合检索器（语义+BM25，不含图权重）
                from .hybrid_retriever_simple import HybridRetriever
                self.retriever = HybridRetriever(
                    semantic_retriever=semantic_retriever,
                    bm25_manager=bm25_manager,
                    config=HybridConfig(
                        semantic_weight=self.hybrid_config.semantic_weight,
                        bm25_weight=self.hybrid_config.bm25_weight
                    )
                )
                logger.info("混合检索器创建完成（语义+BM25）")
            else:
                # BM25不可用，直接用语义检索器
                logger.warning("BM25不可用，混合检索器降级为纯语义检索")
                self.retriever = semantic_retriever
            
            self.hybrid_initialized = True
            
        except Exception as e:
            logger.error(f"混合检索器创建失败: {e}")
            # 回退到纯语义检索器
            self.retriever = semantic_retriever
            self.hybrid_initialized = True
    
    async def retrieve(self, 
                      query: str, 
                      top_k: int = 5,
                      score_threshold: float = 0.6,
                      top_k_before_rerank: int = 12) -> List[RetrievedDocument]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            score_threshold: 分数阈值
            top_k_before_rerank: 重排序前的文档数量
            
        Returns:
            检索到的文档列表
        """
        start_time = time.time()
        
        try:
            # 初始化混合检索器（如果需要）
            if self.use_hybrid and not self.hybrid_initialized:
                await self._initialize_hybrid_retriever()
            
            # 创建RAG查询对象
            rag_query = RAGQuery(
                query=query,
                top_k=top_k_before_rerank if self.use_rerank else top_k,
                score_threshold=score_threshold,
                collection_name=self.collection_name
            )
            
            # 执行检索
            if self.use_hybrid:
                # 使用混合检索器
                retrieved_docs = await self.retriever.retrieve(rag_query)
            else:
                # 使用简单语义检索器
                retrieved_docs = await self.retriever.retrieve(rag_query)
            
            # 应用自适应阈值过滤（在重排序前）
            adaptive_threshold_applied = False
            adaptive_threshold_value = 0.0
            
            if retrieved_docs:
                # 提取文档分数
                doc_scores = [doc.score for doc in retrieved_docs]
                
                # 计算自适应阈值
                adaptive_threshold = self._calculate_adaptive_threshold(doc_scores, 0.3)
                
                # 应用自适应阈值过滤
                original_count = len(retrieved_docs)
                filtered_docs = []
                for doc in retrieved_docs:
                    if doc.score >= adaptive_threshold:
                        filtered_docs.append(doc)
                
                retrieved_docs = filtered_docs
                adaptive_threshold_value = adaptive_threshold
                adaptive_threshold_applied = True
                
                logger.info("检索后自适应阈值应用",
                           query=query[:50],
                           original_docs=original_count,
                           filtered_docs=len(retrieved_docs),
                           adaptive_threshold=adaptive_threshold,
                           avg_score=sum(doc_scores)/len(doc_scores) if doc_scores else 0)
            
            # 应用重排序（如果需要）
            if self.use_rerank and self.rerank_config and retrieved_docs:
                logger.debug(f"应用重排序，原始文档数: {len(retrieved_docs)}")
                
                # 重排序
                reranked_docs = await rerank_documents(
                    query=query,
                    documents=retrieved_docs,
                    top_k=top_k,
                    config=self.rerank_config
                )
                
                retrieved_docs = reranked_docs
            
            # 确保不超过top_k
            if len(retrieved_docs) > top_k:
                retrieved_docs = retrieved_docs[:top_k]
            
            # 记录检索统计
            retrieval_time = time.time() - start_time
            logger.info(f"检索完成: query='{query[:50]}...', docs={len(retrieved_docs)}, time={retrieval_time:.3f}s")
            
            # 添加检索元数据
            for doc in retrieved_docs:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                
                doc.metadata.update({
                    "retrieval_time": retrieval_time,
                    "retrieval_method": "hybrid_with_graphrag" if self.use_hybrid and self.graphrag_available else "hybrid",
                    "collection": self.collection_name,
                    "adaptive_threshold_applied": adaptive_threshold_applied,
                    "adaptive_threshold_value": adaptive_threshold_value if adaptive_threshold_applied else None
                })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    async def generate(self, 
                      query: str, 
                      retrieved_docs: List[RetrievedDocument],
                      max_tokens: int = 1000,
                      temperature: float = 0.7) -> str:
        """
        生成答案
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        try:
            # 准备上下文
            context = "\n\n".join([doc.text for doc in retrieved_docs])
            
            if not context:
                return "抱歉，没有找到相关信息。"
            
            # 生成答案
            answer = await self.generator.generate_answer(
                query=query,
                documents=retrieved_docs
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"生成答案时出错: {str(e)}"
    
    async def generate_with_graph(self,
                                 query: str,
                                 retrieved_docs: List[RetrievedDocument],
                                 graph_description: str = "",
                                 max_tokens: int = 1000,
                                 temperature: float = 0.7) -> str:
        """
        生成答案（包含关联子图描述）
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
            graph_description: 关联子图描述
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        try:
            if not retrieved_docs and not graph_description:
                return "抱歉，没有找到相关信息。"
            
                        # 如果有关联子图描述，添加到上下文中
            if graph_description:
                # 创建包含图描述的文档，并添加生成器需要的metadata字段
                graph_doc = RetrievedDocument(
                    id="graph_description",
                    text=graph_description,
                    metadata={
                        "source": "graph_description",
                        "score": 0.8,
                        "retrieval_type": "graph_description",
                        # 生成器需要的字段
                        "graph_structure": {
                            "node_count": len(graph_description.split()),  # 粗略估计
                            "edge_count": graph_description.count("相关于") + graph_description.count("管理") + graph_description.count("约束")
                        },
                        "graph_entities": self._extract_entities_from_description(graph_description),
                        "graph_relations": self._extract_relations_from_description(graph_description)
                    },
                    score=0.8
                )
                all_docs = [graph_doc] + retrieved_docs
            else:
                all_docs = retrieved_docs
            
            # 生成答案
            answer = await self.generator.generate_answer(
                query=query,
                documents=all_docs
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"生成答案时出错: {str(e)}"
            return f"生成答案时出错: {str(e)}"
    
    def _extract_entities_from_description(self, description: str):
        """从关联子图描述中提取实体"""
        try:
            # 简单提取：查找中文实体（通常由中文词组成）
            import re
            
            # 匹配常见的中文实体模式
            patterns = [
                r'涉及实体包括：([^。]+)',  # "涉及实体包括：A、B、C"
                r'实体包括：([^。]+)',      # "实体包括：A、B、C"
                r'([^、，；]+)相关于',      # "A相关于B"
                r'相关于([^、，；]+)',      # "相关于B"
                r'([^、，；]+)管理',        # "A管理B"
                r'管理([^、，；]+)',        # "管理B"
                r'([^、，；]+)约束',        # "A约束B"
                r'约束([^、，；]+)'         # "约束B"
            ]
            
            entities = set()
            for pattern in patterns:
                matches = re.findall(pattern, description)
                for match in matches:
                    # 分割多个实体
                    parts = re.split(r'[、，；]', match)
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 1:  # 至少2个字符
                            entities.add(part)
            
            return list(entities)[:20]  # 最多返回20个实体
            
        except Exception as e:
            logger.error(f"提取实体失败: {e}")
            return []
    
    def _extract_relations_from_description(self, description: str):
        """从关联子图描述中提取关系"""
        try:
            import re
            
            relations = []
            
            # 匹配关系模式
            relation_patterns = [
                (r'([^相关于]+)相关于([^、，；。]+)', "RELATED_TO"),
                (r'([^管理]+)管理([^、，；。]+)', "MANAGES"),
                (r'([^约束]+)约束([^、，；。]+)', "GOVERNS"),
                (r'([^引用]+)引用([^、，；。]+)', "REFERS_TO"),
                (r'([^属于]+)属于([^、，；。]+)', "BELONGS_TO"),
                (r'([^汇报给]+)汇报给([^、，；。]+)', "REPORTS_TO")
            ]
            
            for pattern, rel_type in relation_patterns:
                matches = re.findall(pattern, description)
                for source, target in matches:
                    source = source.strip()
                    target = target.strip()
                    if source and target and len(source) > 1 and len(target) > 1:
                        relations.append({
                            "source": source,
                            "target": target,
                            "type": rel_type
                        })
            
            return relations[:10]  # 最多返回10个关系
            
        except Exception as e:
            logger.error(f"提取关系失败: {e}")
            return []

    
    async def query(self,
                   query: str,
                   top_k: int = 5,
                   score_threshold: float = 0.6,
                   max_tokens: int = 1000,
                   temperature: float = 0.7,
                   top_k_before_rerank: int = 12) -> RAGResult:
        """
        完整的RAG查询（集成独立GraphRAG）
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            score_threshold: 分数阈值
            max_tokens: 最大token数
            temperature: 温度参数
            top_k_before_rerank: 重排序前的文档数量
            
        Returns:
            RAG结果
        """
        total_start_time = time.time()
        
        tracer = get_tracer()
        with tracer.start_as_current_span("rag_query") as root_span:
            root_span.set_attribute("query", query[:100])
            
            try:
                # 1. 执行独立GraphRAG检索（获取关联子图描述）
                graph_description = ""
                
                if self.graphrag_available and self.independent_graph_retriever:
                    try:
                        with span("graphrag_retrieval"):
                            graphrag_start = time.time()
                            graphrag_query = RAGQuery(
                                query=query,
                                top_k=top_k,
                                score_threshold=score_threshold,
                                collection_name=self.collection_name
                            )
                            
                            # 执行独立GraphRAG检索（只获取关联子图描述）
                            _, graph_description = await self.independent_graph_retriever.retrieve(graphrag_query)
                            graphrag_time = time.time() - graphrag_start
                        
                        logger.info(f"独立GraphRAG检索完成: 耗时 {graphrag_time:.3f}s")
                        if graph_description:
                            logger.debug(f"关联子图描述: {graph_description[:200]}...")
                        else:
                            logger.warning("未生成关联子图描述")
                            
                    except Exception as e:
                        logger.warning(f"独立GraphRAG检索失败: {e}")
                
                # 2. 执行混合检索 + 重排序（获取重排序后的文档）
                retrieval_start = time.time()
                with span("hybrid_retrieval"):
                    retrieved_docs = await self.retrieve(
                        query=query,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        top_k_before_rerank=top_k_before_rerank
                    )
                retrieval_time = time.time() - retrieval_start
                
                # 3. 生成答案（包含关联子图描述 + 重排序结果）
                generation_start = time.time()
                # 检查熔断器是否打开
                if getattr(self.generator.llm_service, '_cb_state', 'closed') == 'open':
                    logger.warning("熔断器打开，跳过LLM，使用降级回答")
                    docs_text = "\n\n".join([d.text[:300] for d in retrieved_docs[:3]])
                    answer = f"[降级响应] AI服务暂时不可用，以下为检索到的相关文档片段：\n\n{docs_text}" if docs_text else "[降级响应] AI服务暂时不可用"
                else:
                    with span("llm_generation"):
                        answer = await self.generate_with_graph(
                            query=query,
                            retrieved_docs=retrieved_docs,
                            graph_description=graph_description,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                generation_time = time.time() - generation_start
                
                # 4. 构建结果
                total_time = time.time() - total_start_time
                
                all_retrieved = list(retrieved_docs)
                if graph_description:
                    graph_doc_for_result = RetrievedDocument(
                        id="graph_analysis",
                        text=f"[知识图谱关联分析]\n{graph_description}",
                        metadata={
                            "source": "graph_analysis",
                            "score": 0.8,
                            "retrieval_type": "graphrag"
                        },
                        score=0.8
                    )
                    all_retrieved.insert(0, graph_doc_for_result)

                result = RAGResult(
                    query=query,
                    answer=answer,
                    retrieved_documents=all_retrieved,
                    model=self.generator.llm_service.model,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    total_time=total_time,
                    metadata={
                        "collection": self.collection_name,
                        "use_hybrid": self.use_hybrid,
                        "use_graphrag": self.use_graphrag,
                        "use_rerank": self.use_rerank,
                        "graphrag_available": self.graphrag_available,
                        "has_graph_description": bool(graph_description),
                        "graph_description_length": len(graph_description) if graph_description else 0
                    }
                )
                
                logger.info(f"RAG查询完成: query='{query[:50]}...', time={total_time:.3f}s, docs={len(retrieved_docs)}, 关联子图描述={'有' if graph_description else '无'}")
                record_request(query, total_time, len(retrieved_docs), self.generator.last_token_count)
                return result
                
            except Exception as e:
                logger.error(f"RAG查询失败: {e}")
                record_error(query)
                raise
    
    async def batch_query(self,
                         queries: List[str],
                         top_k: int = 5,
                         score_threshold: float = 0.6,
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> List[RAGResult]:
        """
        批量RAG查询
        
        Args:
            queries: 查询列表
            top_k: 返回文档数量
            score_threshold: 分数阈值
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            RAG结果列表
        """
        tasks = []
        for query in queries:
            task = self.query(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                max_tokens=max_tokens,
                temperature=temperature
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"查询失败: {queries[i]}, error: {result}")
                final_results.append(RAGResult(
                    query=queries[i],
                    answer=f"查询失败: {str(result)}",
                    retrieved_documents=[],
                    model="error",
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0,
                    metadata={"error": str(result)}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def close(self):
        """清理资源"""
        if self.graph_retriever:
            try:
                self.graph_retriever.close()
            except:
                pass


# 全局管道实例缓存
_pipeline_cache = {}


def get_hybrid_rag_pipeline_with_graphrag(
    collection_name: str = "default",
    use_hybrid: bool = True,
    use_graphrag: bool = True,
    use_rerank: bool = False,
    hybrid_config: Optional[EnhancedHybridConfig] = None,
    rerank_config: Optional[RerankConfig] = None,
    force_new: bool = False
) -> HybridRAGPipelineWithGraphRAG:
    """
    获取或创建RAG管道实例（支持GraphRAG）
    
    Args:
        collection_name: 集合名称
        use_hybrid: 是否使用混合检索
        use_graphrag: 是否使用GraphRAG
        use_rerank: 是否使用重排序
        hybrid_config: 增强版混合检索配置
        rerank_config: 重排序配置
        force_new: 是否强制创建新实例
        
    Returns:
        RAG管道实例
    """
    cache_key = f"{collection_name}_{use_hybrid}_{use_graphrag}_{use_rerank}"
    
    if not force_new and cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    
    # 创建新实例
    pipeline = HybridRAGPipelineWithGraphRAG(
        collection_name=collection_name,
        use_hybrid=use_hybrid,
        use_graphrag=use_graphrag,
        use_rerank=use_rerank,
        hybrid_config=hybrid_config,
        rerank_config=rerank_config
    )
    
    _pipeline_cache[cache_key] = pipeline
    return pipeline


async def cleanup_pipeline_cache():
    """清理管道缓存"""
    for pipeline in _pipeline_cache.values():
        pipeline.close()
    
    _pipeline_cache.clear()
    logger.info("管道缓存已清理")