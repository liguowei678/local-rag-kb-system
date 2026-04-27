"""
GraphRAG工厂函数 - 使用正确的neo4j-graphrag组件
"""

import logging
from typing import Optional, Dict, Any

from .config import GraphConfig
from .client import Neo4jClient
from .retriever import GraphRetriever
# 导入稳定的构建器
from .stable_builder import StableKnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class GraphRAGFactory:
    """GraphRAG工厂"""
    
    @staticmethod
    def create_config() -> GraphConfig:
        """创建配置"""
        config = GraphConfig.from_env()
        config.validate_config()
        return config
    
    @staticmethod
    def create_neo4j_client(config: Optional[GraphConfig] = None) -> Neo4jClient:
        """创建Neo4j客户端"""
        if config is None:
            config = GraphRAGFactory.create_config()
        
        return Neo4jClient(config)
    
    @staticmethod
    def create_graph_retriever(config: Optional[GraphConfig] = None) -> Optional[GraphRetriever]:
        """创建图检索器"""
        if config is None:
            config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            logger.debug("GraphRAG disabled, skipping graph retriever creation")
            return None
        
        try:
            return GraphRetriever(config=config)
        except Exception as e:
            logger.error(f"Failed to create graph retriever: {e}")
            return None
    
    @staticmethod
    def create_knowledge_graph_builder(config: Optional[GraphConfig] = None) -> Optional[StableKnowledgeGraphBuilder]:
        """创建知识图谱构建器"""
        if config is None:
            config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            logger.debug("GraphRAG disabled, skipping knowledge graph builder creation")
            return None
        
        if not config.deepseek_api_key:
            logger.warning("DeepSeek API key not set, cannot create knowledge graph builder")
            return None
        
        try:
            # 使用稳定的构建器（修复了自定义标签问题）
            from .stable_builder import StableKnowledgeGraphBuilder
            return StableKnowledgeGraphBuilder(config)
        except Exception as e:
            logger.error(f"Failed to create knowledge graph builder: {e}")
            return None
    
    @staticmethod
    def get_graphrag_status() -> Dict[str, Any]:
        """获取GraphRAG状态"""
        config = GraphRAGFactory.create_config()
        
        status = {
            "enabled": config.graphrag_enabled,
            "config": {
                "retrieval_strategy": config.retrieval_strategy,
                "graph_retrieval_weight": config.graph_retrieval_weight,
                "entity_types": config.entity_types,
                "relation_types": config.relation_types,
                "deepseek_model": config.deepseek_model,
                "deepseek_available": bool(config.deepseek_api_key)
            },
            "dependencies": {
                "neo4j_available": False,
                "deepseek_available": bool(config.deepseek_api_key),
                "graphrag_available": False
            },
            "services": {
                "neo4j": "unknown",
                "graph_retriever": "unknown",
                "knowledge_graph_builder": "unknown"
            }
        }
        
        # 测试Neo4j连接
        try:
            client = GraphRAGFactory.create_neo4j_client(config)
            stats = client.get_statistics()
            status["dependencies"]["neo4j_available"] = True
            status["services"]["neo4j"] = "connected"
            status["statistics"] = stats
            client.close()
        except Exception as e:
            status["services"]["neo4j"] = f"error: {str(e)}"
        
        # 测试图检索器
        try:
            retriever = GraphRAGFactory.create_graph_retriever(config)
            if retriever:
                status["services"]["graph_retriever"] = "available"
                status["dependencies"]["graphrag_available"] = True
            else:
                status["services"]["graph_retriever"] = "unavailable"
        except Exception as e:
            status["services"]["graph_retriever"] = f"error: {str(e)}"
        
        # 测试知识图谱构建器
        try:
            builder = GraphRAGFactory.create_knowledge_graph_builder(config)
            if builder:
                status["services"]["knowledge_graph_builder"] = "available"
            else:
                status["services"]["knowledge_graph_builder"] = "unavailable"
        except Exception as e:
            status["services"]["knowledge_graph_builder"] = f"error: {str(e)}"
        
        return status
    
    @staticmethod
    def test_components() -> Dict[str, Any]:
        """测试所有组件"""
        config = GraphRAGFactory.create_config()
        
        results = {
            "config": {
                "valid": False,
                "graphrag_enabled": config.graphrag_enabled,
                "deepseek_available": bool(config.deepseek_api_key)
            },
            "neo4j": {
                "available": False,
                "error": None
            },
            "llm": {
                "available": False,
                "error": None
            },
            "retriever": {
                "available": False,
                "error": None
            },
            "builder": {
                "available": False,
                "error": None
            }
        }
        
        # 测试配置
        try:
            config.validate_config()
            results["config"]["valid"] = True
        except Exception as e:
            results["config"]["error"] = str(e)
        
        # 测试Neo4j
        try:
            client = GraphRAGFactory.create_neo4j_client(config)
            stats = client.get_statistics()
            results["neo4j"]["available"] = True
            results["neo4j"]["stats"] = stats
            client.close()
        except Exception as e:
            results["neo4j"]["error"] = str(e)
        
        # 测试LLM
        if config.deepseek_api_key:
            try:
                from neo4j_graphrag.llm import OpenAILLM
                llm = OpenAILLM(
                    model_name=config.deepseek_model,
                    api_key=config.deepseek_api_key,
                    base_url=config.deepseek_base_url
                )
                results["llm"]["available"] = True
            except Exception as e:
                results["llm"]["error"] = str(e)
        
        # 测试检索器
        try:
            retriever = GraphRAGFactory.create_graph_retriever(config)
            if retriever:
                results["retriever"]["available"] = True
                results["retriever"]["stats"] = retriever.get_retrieval_stats()
            else:
                results["retriever"]["error"] = "Retriever creation returned None"
        except Exception as e:
            results["retriever"]["error"] = str(e)
        
        # 测试构建器
        try:
            builder = GraphRAGFactory.create_knowledge_graph_builder(config)
            if builder:
                results["builder"]["available"] = True
            else:
                results["builder"]["error"] = "Builder creation returned None"
        except Exception as e:
            results["builder"]["error"] = str(e)
        
        return results