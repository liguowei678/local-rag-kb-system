"""
Neo4j图数据库客户端
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from .models import Entity, Relation, DocumentEntityLink, GraphConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j图数据库客户端"""
    
    def __init__(self, config: GraphConfig):
        """初始化Neo4j客户端"""
        self.config = config
        self.driver: Optional[Driver] = None
        self._connect()
        self._initialize_database()
    
    def _connect(self):
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                database=self.config.database
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"成功连接到Neo4j: {self.config.neo4j_uri}")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def _initialize_database(self):
        """初始化数据库结构和索引"""
        try:
            with self.driver.session() as session:
                # 创建实体节点约束
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                
                # 创建文档节点约束
                session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS 
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
                
                # 创建实体名称索引
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS 
                    FOR (e:Entity) ON (e.name)
                """)
                
                # 创建实体类型索引
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS 
                    FOR (e:Entity) ON (e.type)
                """)
                
                # 创建关系类型索引
                session.run("""
                    CREATE INDEX relation_type IF NOT EXISTS 
                    FOR ()-[r:RELATION]-() ON (r.type)
                """)
                
                logger.info("Neo4j数据库初始化完成")
        except Exception as e:
            logger.error(f"初始化Neo4j数据库失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def create_entity(self, entity: Entity) -> bool:
        """创建实体节点"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.confidence = $confidence,
                        e.metadata = $metadata,
                        e.created_at = $created_at,
                        e.updated_at = $updated_at
                    RETURN e.id
                """, **entity.dict())
                
                return result.single() is not None
        except Neo4jError as e:
            logger.error(f"创建实体失败: {e}")
            return False
    
    def create_relation(self, relation: Relation) -> bool:
        """创建关系边"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (source:Entity {id: $source_entity_id})
                    MATCH (target:Entity {id: $target_entity_id})
                    MERGE (source)-[r:RELATION {id: $id}]->(target)
                    SET r.type = $type,
                        r.description = $description,
                        r.confidence = $confidence,
                        r.metadata = $metadata,
                        r.created_at = $created_at
                    RETURN r.id
                """, **relation.dict())
                
                return result.single() is not None
        except Neo4jError as e:
            logger.error(f"创建关系失败: {e}")
            return False
    
    def link_document_to_entity(self, link: DocumentEntityLink) -> bool:
        """关联文档和实体"""
        try:
            with self.driver.session() as session:
                # 创建或更新文档节点
                session.run("""
                    MERGE (d:Document {id: $document_id})
                    SET d.updated_at = datetime()
                """, document_id=link.document_id)
                
                # 创建关联关系
                result = session.run("""
                    MATCH (d:Document {id: $document_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (d)-[r:MENTIONS]->(e)
                    SET r.mentions = $mentions,
                        r.frequency = $frequency,
                        r.relevance = $relevance,
                        r.created_at = $created_at
                    RETURN r
                """, **link.dict())
                
                return result.single() is not None
        except Neo4jError as e:
            logger.error(f"关联文档实体失败: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity {id: $entity_id})
                    RETURN e
                """, entity_id=entity_id)
                
                record = result.single()
                if record:
                    data = dict(record["e"])
                    return Entity(**data)
                return None
        except Neo4jError as e:
            logger.error(f"获取实体失败: {e}")
            return None
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """搜索实体"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                    RETURN e
                    LIMIT $limit
                """, query=query, limit=limit)
                
                entities = []
                for record in result:
                    data = dict(record["e"])
                    entities.append(Entity(**data))
                return entities
        except Neo4jError as e:
            logger.error(f"搜索实体失败: {e}")
            return []
    
    def get_related_documents(self, entity_id: str, limit: int = 10) -> List[str]:
        """获取实体相关的文档"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity {id: $entity_id})<-[:MENTIONS]-(d:Document)
                    RETURN d.id
                    ORDER BY d.updated_at DESC
                    LIMIT $limit
                """, entity_id=entity_id, limit=limit)
                
                return [record["d.id"] for record in result]
        except Neo4jError as e:
            logger.error(f"获取相关文档失败: {e}")
            return []
    
    def find_entity_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Dict]]:
        """查找实体间路径"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = shortestPath((source:Entity {id: $source_id})-[*1..$max_depth]-(target:Entity {id: $target_id}))
                    RETURN nodes(path) as nodes, relationships(path) as rels
                """, source_id=source_id, target_id=target_id, max_depth=max_depth)
                
                paths = []
                for record in result:
                    nodes = [dict(node) for node in record["nodes"]]
                    rels = [dict(rel) for rel in record["rels"]]
                    paths.append({"nodes": nodes, "relationships": rels})
                return paths
        except Neo4jError as e:
            logger.error(f"查找实体路径失败: {e}")
            return []
    
    def graph_query(self, query_text: str, max_depth: int = 2, limit: int = 10) -> Dict[str, Any]:
        """执行图查询"""
        try:
            with self.driver.session() as session:
                # 查找相关实体
                entities_result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                    RETURN e
                    LIMIT $limit
                """, query=query_text, limit=limit)
                
                entities = []
                entity_ids = []
                for record in entities_result:
                    data = dict(record["e"])
                    entity = Entity(**data)
                    entities.append(entity)
                    entity_ids.append(entity.id)
                
                if not entity_ids:
                    return {"entities": [], "relations": [], "paths": [], "document_ids": []}
                
                # 查找相关关系和文档
                relations_result = session.run("""
                    MATCH (e1:Entity)-[r:RELATION]-(e2:Entity)
                    WHERE e1.id IN $entity_ids OR e2.id IN $entity_ids
                    RETURN e1, r, e2
                    LIMIT $limit * 2
                """, entity_ids=entity_ids, limit=limit)
                
                relations = []
                for record in relations_result:
                    rel_data = dict(record["r"])
                    relations.append({
                        "id": rel_data.get("id"),
                        "type": rel_data.get("type"),
                        "source": dict(record["e1"])["id"],
                        "target": dict(record["e2"])["id"],
                        "confidence": rel_data.get("confidence", 0.0)
                    })
                
                # 查找相关文档
                docs_result = session.run("""
                    MATCH (e:Entity)<-[:MENTIONS]-(d:Document)
                    WHERE e.id IN $entity_ids
                    RETURN DISTINCT d.id
                    LIMIT $limit
                """, entity_ids=entity_ids, limit=limit)
                
                document_ids = [record["d.id"] for record in docs_result]
                
                return {
                    "entities": [e.dict() for e in entities],
                    "relations": relations,
                    "paths": self._find_relation_paths(entity_ids, max_depth),
                    "document_ids": document_ids
                }
        except Neo4jError as e:
            logger.error(f"图查询失败: {e}")
            return {"entities": [], "relations": [], "paths": [], "document_ids": []}
    
    def _find_relation_paths(self, entity_ids: List[str], max_depth: int) -> List[List[Dict]]:
        """查找实体间的关系路径"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = (e1:Entity)-[r:RELATION*1..$max_depth]-(e2:Entity)
                    WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids AND e1.id < e2.id
                    RETURN nodes(path) as nodes, relationships(path) as rels
                    LIMIT 10
                """, entity_ids=entity_ids, max_depth=max_depth)
                
                paths = []
                for record in result:
                    nodes = [dict(node) for node in record["nodes"]]
                    rels = [dict(rel) for rel in record["rels"]]
                    paths.append({"nodes": nodes, "relationships": rels})
                return paths
        except Neo4jError as e:
            logger.error(f"查找关系路径失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        try:
            with self.driver.session() as session:
                # 实体统计
                entity_stats = session.run("""
                    MATCH (e:Entity)
                    RETURN count(e) as total_entities,
                           count(DISTINCT e.type) as entity_types,
                           avg(e.confidence) as avg_confidence
                """).single()
                
                # 关系统计
                relation_stats = session.run("""
                    MATCH ()-[r:RELATION]->()
                    RETURN count(r) as total_relations,
                           count(DISTINCT r.type) as relation_types
                """).single()
                
                # 文档统计
                doc_stats = session.run("""
                    MATCH (d:Document)
                    RETURN count(d) as total_documents
                """).single()
                
                return {
                    "entities": {
                        "total": entity_stats["total_entities"] if entity_stats else 0,
                        "types": entity_stats["entity_types"] if entity_stats else 0,
                        "avg_confidence": float(entity_stats["avg_confidence"]) if entity_stats and entity_stats["avg_confidence"] else 0.0
                    },
                    "relations": {
                        "total": relation_stats["total_relations"] if relation_stats else 0,
                        "types": relation_stats["relation_types"] if relation_stats else 0
                    },
                    "documents": {
                        "total": doc_stats["total_documents"] if doc_stats else 0
                    }
                }
        except Neo4jError as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def clear_all(self) -> bool:
        """清空所有数据（谨慎使用）"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.warning("已清空Neo4j所有数据")
                return True
        except Neo4jError as e:
            logger.error(f"清空数据失败: {e}")
            return False