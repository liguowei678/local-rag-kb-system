"""
GraphRAG配置模块
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field


class GraphConfig(BaseModel):
    """GraphRAG配置"""
    
    # Neo4j配置
    neo4j_uri: str = Field(
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        description="Neo4j连接URI"
    )
    neo4j_user: str = Field(
        default=os.getenv("NEO4J_USER", "neo4j"),
        description="Neo4j用户名"
    )
    neo4j_password: str = Field(
        default=os.getenv("NEO4J_PASSWORD", "password"),
        description="Neo4j密码"
    )
    neo4j_database: str = Field(
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        description="Neo4j数据库"
    )
    
    # DeepSeek LLM配置
    deepseek_api_key: str = Field(
        default=os.getenv("DEEPSEEK_API_KEY", ""),
        description="DeepSeek API密钥"
    )
    deepseek_base_url: str = Field(
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        description="DeepSeek API基础URL"
    )
    deepseek_model: str = Field(
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        description="DeepSeek模型名称"
    )
    
    # GraphRAG功能开关
    graphrag_enabled: bool = Field(
        default=os.getenv("GRAPHRAG_ENABLED", "false").lower() == "true",
        description="是否启用GraphRAG"
    )
    
    # 检索策略配置
    retrieval_strategy: str = Field(
        default=os.getenv("RETRIEVAL_STRATEGY", "hybrid"),
        description="检索策略: hybrid, vector_only, bm25_only, graph_only, hybrid_graph"
    )
    graph_retrieval_weight: float = Field(
        default=float(os.getenv("GRAPH_RETRIEVAL_WEIGHT", "0.3")),
        ge=0.0,
        le=1.0,
        description="图检索权重"
    )
    
    # 实体提取配置
    entity_types: List[str] = Field(
        default=os.getenv("GRAPHRAG_ENTITY_TYPES", "PERSON,ORGANIZATION,TECHNOLOGY,CONCEPT,LOCATION").split(","),
        description="实体类型列表"
    )
    relation_types: List[str] = Field(
        default=os.getenv("GRAPHRAG_RELATION_TYPES", "MENTIONS,RELATED_TO,USES,DEVELOPED_BY,LOCATED_IN").split(","),
        description="关系类型列表"
    )
    max_entities_per_doc: int = Field(
        default=int(os.getenv("GRAPHRAG_MAX_ENTITIES_PER_DOC", "10")),
        description="每个文档最大实体数"
    )
    entity_confidence_threshold: float = Field(
        default=float(os.getenv("GRAPHRAG_ENTITY_CONFIDENCE_THRESHOLD", "0.8")),
        description="实体置信度阈值"
    )
    
    # 图检索配置
    graph_search_depth: int = Field(
        default=int(os.getenv("GRAPH_SEARCH_DEPTH", "2")),
        description="图搜索深度"
    )
    graph_search_limit: int = Field(
        default=int(os.getenv("GRAPH_SEARCH_LIMIT", "10")),
        description="图搜索限制"
    )
    
    # 缓存配置
    cache_ttl: int = Field(
        default=int(os.getenv("GRAPH_CACHE_TTL", "3600")),
        description="缓存TTL（秒）"
    )
    
    class Config:
        env_prefix = ""
        case_sensitive = False
        
    @classmethod
    def from_env(cls) -> "GraphConfig":
        """从环境变量创建配置"""
        return cls()
    
    def validate_config(self) -> bool:
        """验证配置"""
        if self.graphrag_enabled and not self.deepseek_api_key:
            raise ValueError("启用GraphRAG需要设置DEEPSEEK_API_KEY")
        
        if self.graph_retrieval_weight < 0 or self.graph_retrieval_weight > 1:
            raise ValueError("图检索权重必须在0-1之间")
        
        return True