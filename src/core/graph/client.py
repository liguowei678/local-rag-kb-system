"""
Neo4j图数据库客户端
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver

from .config import GraphConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j图数据库客户端"""
    
    def __init__(self, config: GraphConfig):
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
                database=self.config.neo4j_database
            )
            
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info(f"Connected to Neo4j: {self.config.neo4j_uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _initialize_database(self):
        """初始化数据库结构和索引"""
        try:
            with self.driver.session() as session:
                # 创建约束
                session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS 
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                
                # 创建索引
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS 
                    FOR (e:Entity) ON (e.name)
                """)
                
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS 
                    FOR (e:Entity) ON (e.type)
                """)
                
                logger.info("Neo4j database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            # 不抛出异常，允许继续
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_document_node(self, document_id: str, metadata: Dict[str, Any], 
                           content: str = None, content_hash: str = None) -> bool:
        """创建文档节点"""
        try:
            # 将metadata字典转换为JSON字符串，因为Neo4j不支持Map类型
            import json
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
            
            with self.driver.session() as session:
                # 构建SET子句
                set_clauses = [
                    "d.metadata = $metadata_json",
                    "d.created_at = datetime()",
                    "d.updated_at = datetime()"
                ]
                
                parameters = {
                    "document_id": document_id,
                    "metadata_json": metadata_json
                }
                
                # 添加content（如果提供）
                if content is not None:
                    set_clauses.append("d.content = $content")
                    parameters["content"] = content
                
                # 添加content_hash（如果提供）
                if content_hash is not None:
                    set_clauses.append("d.content_hash = $content_hash")
                    parameters["content_hash"] = content_hash
                
                # 构建完整查询
                set_clause = ",\n                        ".join(set_clauses)
                cypher_query = f"""
                    MERGE (d:Document {{id: $document_id}})
                    SET {set_clause}
                    RETURN d.id
                """
                
                result = session.run(cypher_query, **parameters)
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to create document node: {e}")
            return False
    
    def create_entity_node(self, entity_id: str, name: str, entity_type: str, 
                          confidence: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """创建实体节点"""
        try:
            # 将metadata字典转换为JSON字符串
            import json
            
            # 添加中英文标签映射到metadata
            enhanced_metadata = metadata.copy() if metadata else {}
            
            # 定义中英文标签映射
            type_mapping = {
                "Company": {"en": "Company", "zh": "公司"},
                "Department": {"en": "Department", "zh": "部门"},
                "Employee": {"en": "Employee", "zh": "员工"},
                "Policy": {"en": "Policy", "zh": "制度"},
                "Contract": {"en": "Contract", "zh": "合同"},
                "Leave": {"en": "Leave", "zh": "假期"},
                "Salary": {"en": "Salary", "zh": "薪酬"},
                "Discipline": {"en": "Discipline", "zh": "奖惩"},
                "Duty": {"en": "Duty", "zh": "职责"},
                "Process": {"en": "Process", "zh": "流程"}
            }
            
            # 添加标签映射
            if entity_type in type_mapping:
                enhanced_metadata["type_labels"] = type_mapping[entity_type]
            else:
                enhanced_metadata["type_labels"] = {"en": entity_type, "zh": entity_type}
            
            metadata_json = json.dumps(enhanced_metadata, ensure_ascii=False)
            
            # 清理字符串，移除控制字符
            import unicodedata
            def clean_string(s):
                if not s:
                    return s
                # 移除控制字符
                s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')
                return s
            
            name_encoded = clean_string(name)
            type_encoded = clean_string(entity_type)
            
            with self.driver.session() as session:
                # 根据实体类型动态设置节点标签
                # 保持Entity标签（向后兼容），添加类型特定标签
                valid_labels = ["Company", "Department", "Employee", "Policy", 
                               "Contract", "Leave", "Salary", "Discipline", "Duty", "Process"]
                
                # 调试日志：检查entity_type
                logger.debug(f"创建实体节点: id={entity_id}, name={name}, entity_type={entity_type}")
                logger.debug(f"valid_labels: {valid_labels}")
                logger.debug(f"entity_type in valid_labels: {entity_type in valid_labels}")
                
                if entity_type in valid_labels:
                    # 使用类型特定标签 + Entity标签
                    logger.debug(f"使用双标签: Entity:{entity_type}")
                    cypher_query = f"""
                        MERGE (e:Entity:{entity_type} {{id: $entity_id}})
                        SET e.name = $name,
                            e.type = $type,
                            e.confidence = $confidence,
                            e.metadata = $metadata_json,
                            e.created_at = datetime(),
                            e.updated_at = datetime()
                        RETURN e.id
                    """
                else:
                    # 未知类型，只使用Entity标签
                    logger.warning(f"未知实体类型: {entity_type}，只使用Entity标签")
                    cypher_query = """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.name = $name,
                            e.type = $type,
                            e.confidence = $confidence,
                            e.metadata = $metadata_json,
                            e.created_at = datetime(),
                            e.updated_at = datetime()
                        RETURN e.id
                    """
                
                result = session.run(cypher_query, entity_id=entity_id, name=name_encoded, type=type_encoded,
                    confidence=confidence, metadata_json=metadata_json)
                
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to create entity node: {e}")
            return False
    
    def create_relation(self, source_id: str, target_id: str, relation_type: str,
                       confidence: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """创建关系"""
        try:
            # 将metadata字典转换为JSON字符串（确保中文不转义）
            import json
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
            
            with self.driver.session() as session:
                # 使用动态关系类型（需要验证relation_type是安全的）
                # 只允许字母、数字、下划线
                import re
                safe_relation_type = re.sub(r'[^a-zA-Z0-9_]', '_', relation_type)
                if not safe_relation_type:
                    safe_relation_type = "RELATED_TO"
                
                query = f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MATCH (target:Entity {{id: $target_id}})
                    MERGE (source)-[r:{safe_relation_type}]->(target)
                    SET r.confidence = $confidence,
                        r.metadata = $metadata_json,
                        r.created_at = datetime(),
                        r.original_type = $original_type
                    RETURN r
                """
                
                result = session.run(query, source_id=source_id, target_id=target_id,
                    confidence=confidence, metadata_json=metadata_json, original_type=relation_type)
                
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to create relation: {e}")
            return False
    
    def link_document_to_entity(self, document_id: str, entity_id: str, 
                               mentions: List[str] = None, relevance: float = 1.0) -> bool:
        """关联文档和实体"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {id: $document_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (d)-[r:MENTIONS]->(e)
                    SET r.mentions = $mentions,
                        r.relevance = $relevance,
                        r.created_at = datetime()
                    RETURN r
                """, document_id=document_id, entity_id=entity_id,
                    mentions=mentions or [], relevance=relevance)
                
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to link document to entity: {e}")
            return False
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索实体"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query OR e.type CONTAINS $query
                    RETURN e.id as id, e.name as name, e.type as type, e.confidence as confidence
                    LIMIT $limit
                """, query=query, limit=limit)
                
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []
    
    def get_related_documents(self, entity_names: List[str], limit: int = 10) -> Dict[str, Any]:
        """获取实体相关的文档和图结构（不指定关系类型）
        
        返回：
        {
            "entities": List[str],  # 所有实体
            "relations": List[Dict], # 关系列表
            "graph_structure": Dict # 图结构信息
        }
        """
        try:
            with self.driver.session() as session:
                # 存储所有找到的实体
                all_entities = []
                relations = []  # 存储关系信息
                
                for entity_name in entity_names:
                    # 1. 首先检查实体是否存在
                    exists_result = session.run(
                        "MATCH (e:Entity {name: $entity_name}) RETURN e.name as entity_name",
                        entity_name=entity_name
                    )
                    
                    if exists_result.single():
                        # 实体存在，添加到列表
                        if entity_name not in all_entities:
                            all_entities.append(entity_name)
                        
                        # 2. 查找与该实体有任意关系的其他实体
                        # 不指定关系类型，使用[r]匹配任意关系
                        related_result = session.run("""
                            MATCH (e1:Entity {name: $entity_name})-[r]-(e2:Entity)
                            RETURN e1.name as source_entity, 
                                   e2.name as target_entity, 
                                   type(r) as relation_type,
                                   id(r) as relation_id
                            LIMIT $limit
                        """, entity_name=entity_name, limit=limit)
                        
                        for record in related_result:
                            source_entity = record["source_entity"]
                            target_entity = record["target_entity"]
                            relation_type = record["relation_type"]
                            relation_id = record["relation_id"]
                            
                            # 添加目标实体到列表
                            if target_entity not in all_entities:
                                all_entities.append(target_entity)
                            
                            # 添加关系信息
                            relation_info = {
                                "source": source_entity,
                                "target": target_entity,
                                "type": relation_type,
                                "id": relation_id
                            }
                            relations.append(relation_info)
                            
                            logger.debug(f"实体 '{source_entity}' --[{relation_type}]--> '{target_entity}'")
                    else:
                        logger.debug(f"实体 '{entity_name}' 不在Neo4j中")
                
                if not all_entities:
                    logger.warning(f"未找到相关实体: {entity_names}")
                    return {
                        "entities": [],
                        "relations": [],
                        "graph_structure": {"node_count": 0, "edge_count": 0}
                    }
                
                logger.info(f"找到 {len(all_entities)} 个实体和 {len(relations)} 个关系")
                
                # 构建图结构信息
                graph_structure = {
                    "node_count": len(all_entities),
                    "edge_count": len(relations),
                    "entities": all_entities,
                    "relation_types": list(set([r["type"] for r in relations])) if relations else []
                }
                
                # 由于文档和实体没有直接关系，返回模拟的文档ID
                # 实际系统中，应该通过文档内容匹配实体名称来找到文档
                simulated_docs = []
                for i, entity in enumerate(all_entities[:limit]):
                    simulated_docs.append(f"doc_related_to_{entity}_{i}")
                
                return {
                    "document_ids": simulated_docs,
                    "entities": all_entities,
                    "relations": relations,
                    "graph_structure": graph_structure
                }
                
        except Exception as e:
            logger.error(f"Failed to get related documents: {e}")
            return {
                "document_ids": [],
                "entities": [],
                "relations": [],
                "graph_structure": {"node_count": 0, "edge_count": 0}
            }
    
    def graph_query(self, query_text: str, max_depth: int = 2, limit: int = 10) -> Dict[str, Any]:
        """执行图查询"""
        try:
            with self.driver.session() as session:
                # 查找相关实体
                entities_result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query OR e.type CONTAINS $query
                    RETURN e.id as id, e.name as name, e.type as type
                    LIMIT $limit
                """, query=query_text, limit=limit)
                
                entities = [dict(record) for record in entities_result]
                entity_ids = [e["id"] for e in entities]
                
                if not entity_ids:
                    return {"entities": [], "document_ids": []}
                
                # 查找相关文档
                docs_result = session.run("""
                    MATCH (e:Entity)<-[:MENTIONS]-(d:Document)
                    WHERE e.id IN $entity_ids
                    RETURN DISTINCT d.id as document_id
                    LIMIT $limit
                """, entity_ids=entity_ids, limit=limit)
                
                document_ids = [record["document_id"] for record in docs_result]
                
                return {
                    "entities": entities,
                    "document_ids": document_ids
                }
        except Exception as e:
            logger.error(f"Failed to execute graph query: {e}")
            return {"entities": [], "document_ids": []}
    
    def get_document(self, doc_id: str):
        """获取文档信息"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    RETURN d.id as id, 
                           d.content as content, 
                           COALESCE(d.content_hash, '') as content_hash,
                           d.metadata as metadata,
                           d.created_at as created_at
                    """,
                    doc_id=doc_id
                )
                record = result.single()
                if record:
                    return {
                        "id": record["id"],
                        "content": record["content"],
                        "content_hash": record["content_hash"],
                        "metadata": record["metadata"],
                        "created_at": record["created_at"]
                    }
                return None
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None
    
    def delete_document_relations(self, doc_id: str) -> bool:
        """删除文档的所有关系"""
        try:
            with self.driver.session() as session:
                # 删除文档相关的所有关系
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[r]-()
                    DELETE r
                    RETURN count(r) as deleted_count
                    """,
                    doc_id=doc_id
                )
                deleted = result.single()["deleted_count"]
                logger.info(f"删除文档 {doc_id} 的关系: {deleted} 个")
                return True
        except Exception as e:
            logger.error(f"删除文档关系失败: {e}")
            return False
    
    def delete_document_and_relations(self, doc_id: str) -> bool:
        """删除文档及其所有相关数据（实体、关系）"""
        try:
            with self.driver.session() as session:
                # 更彻底的删除：删除文档、相关实体及其所有关系
                # 实体ID格式: {doc_id}_{entity_name}_{entity_type}_{chunk_index}
                # 所以我们可以通过实体ID前缀来找到属于该文档的实体
                
                # 1. 删除实体之间的关系（这些实体属于该文档）
                delete_relations_result = session.run(
                    """
                    MATCH (e1:Entity)-[r]-(e2:Entity)
                    WHERE e1.id STARTS WITH $doc_prefix AND e2.id STARTS WITH $doc_prefix
                    DELETE r
                    RETURN count(r) as deleted_relations
                    """,
                    doc_prefix=f"{doc_id}_"
                )
                relations_record = delete_relations_result.single()
                deleted_relations = relations_record["deleted_relations"] if relations_record else 0
                
                # 2. 删除属于该文档的实体
                delete_entities_result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.id STARTS WITH $doc_prefix
                    DELETE e
                    RETURN count(e) as deleted_entities
                    """,
                    doc_prefix=f"{doc_id}_"
                )
                entities_record = delete_entities_result.single()
                deleted_entities = entities_record["deleted_entities"] if entities_record else 0
                
                # 3. 删除文档节点
                delete_doc_result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DELETE d
                    RETURN count(d) as deleted_docs
                    """,
                    doc_id=doc_id
                )
                doc_record = delete_doc_result.single()
                deleted_docs = doc_record["deleted_docs"] if doc_record else 0
                
                logger.info(f"删除文档 {doc_id}: {deleted_docs}文档, {deleted_entities}实体, {deleted_relations}关系")
                return True
        except Exception as e:
            logger.error(f"删除文档及其关系失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            with self.driver.session() as session:
                # 实体统计
                entity_stats = session.run("""
                    MATCH (e:Entity)
                    RETURN count(e) as total_entities,
                           count(DISTINCT e.type) as entity_types
                """).single()
                
                # 文档统计
                doc_stats = session.run("""
                    MATCH (d:Document)
                    RETURN count(d) as total_documents
                """).single()
                
                # 关系统计
                relation_stats = session.run("""
                    MATCH ()-[r:RELATION]->()
                    RETURN count(r) as total_relations
                """).single()
                
                return {
                    "entities": {
                        "total": entity_stats["total_entities"] if entity_stats else 0,
                        "types": entity_stats["entity_types"] if entity_stats else 0
                    },
                    "documents": {
                        "total": doc_stats["total_documents"] if doc_stats else 0
                    },
                    "relations": {
                        "total": relation_stats["total_relations"] if relation_stats else 0
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}