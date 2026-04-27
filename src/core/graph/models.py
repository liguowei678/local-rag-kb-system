"""
GraphRAG数据模型
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Entity(BaseModel):
    """实体模型"""
    id: str = Field(..., description="实体ID")
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Relation(BaseModel):
    """关系模型"""
    id: str = Field(..., description="关系ID")
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    type: str = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系描述")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentEntityLink(BaseModel):
    """文档-实体关联模型"""
    document_id: str = Field(..., description="文档ID")
    entity_id: str = Field(..., description="实体ID")
    mentions: List[str] = Field(default_factory=list, description="提及位置")
    frequency: int = Field(1, description="出现频率")
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="相关性")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GraphQuery(BaseModel):
    """图查询模型"""
    query: str = Field(..., description="查询文本")
    entity_types: Optional[List[str]] = Field(None, description="限制实体类型")
    relation_types: Optional[List[str]] = Field(None, description="限制关系类型")
    max_depth: int = Field(2, ge=1, le=5, description="最大搜索深度")
    limit: int = Field(10, ge=1, le=100, description="结果数量限制")
    include_metadata: bool = Field(True, description="是否包含元数据")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "OpenAI的技术发展",
                "entity_types": ["ORGANIZATION", "TECHNOLOGY"],
                "relation_types": ["DEVELOPED_BY", "USES"],
                "max_depth": 2,
                "limit": 10
            }
        }


class GraphResult(BaseModel):
    """图检索结果模型"""
    entities: List[Entity] = Field(default_factory=list, description="相关实体")
    relations: List[Relation] = Field(default_factory=list, description="相关关系")
    paths: List[List[Dict[str, Any]]] = Field(default_factory=list, description="关系路径")
    document_ids: List[str] = Field(default_factory=list, description="相关文档ID")
    scores: Dict[str, float] = Field(default_factory=dict, description="相关性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GraphConfig(BaseModel):
    """图配置模型"""
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j连接URI")
    neo4j_user: str = Field("neo4j", description="Neo4j用户名")
    neo4j_password: str = Field("password", description="Neo4j密码")
    database: str = Field("neo4j", description="数据库名称")
    
    # 实体提取配置
    entity_extraction_model: str = Field("gpt-3.5-turbo", description="实体提取模型")
    relation_extraction_model: str = Field("gpt-3.5-turbo", description="关系提取模型")
    max_entities_per_doc: int = Field(10, description="每个文档最大实体数")
    max_relations_per_doc: int = Field(20, description="每个文档最大关系数")
    entity_confidence_threshold: float = Field(0.8, description="实体置信度阈值")
    relation_confidence_threshold: float = Field(0.7, description="关系置信度阈值")
    
    # 检索配置
    graph_search_depth: int = Field(2, description="图搜索深度")
    graph_search_limit: int = Field(10, description="图搜索限制")
    cache_ttl: int = Field(3600, description="缓存TTL（秒）")
    
    class Config:
        env_prefix = "graph_"
        schema_extra = {
            "example": {
                "neo4j_uri": "bolt://neo4j:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "graphrag123",
                "entity_extraction_model": "gpt-3.5-turbo"
            }
        }


# 实体类型定义
ENTITY_TYPES = [
    "PERSON",        # 人物
    "ORGANIZATION",  # 组织
    "LOCATION",      # 地点
    "PRODUCT",       # 产品
    "EVENT",         # 事件
    "CONCEPT",       # 概念
    "TECHNOLOGY",    # 技术
    "DATE",          # 日期
    "NUMBER",        # 数字
    "OTHER"          # 其他
]

# 关系类型定义
RELATION_TYPES = [
    "WORKS_FOR",     # 为...工作
    "LOCATED_IN",    # 位于
    "PART_OF",       # 属于
    "CREATED",       # 创建
    "USES",          # 使用
    "MENTIONS",      # 提及
    "RELATED_TO",    # 相关
    "SIMILAR_TO",    # 相似
    "OPPOSITE_OF",   # 相反
    "CAUSES",        # 导致
    "DEVELOPED_BY",  # 由...开发
    "BELONGS_TO",    # 属于
    "INFLUENCES",    # 影响
    "PRECEDES",      # 先于
    "FOLLOWS"        # 后于
]