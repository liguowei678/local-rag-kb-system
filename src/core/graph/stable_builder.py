"""
稳定的知识图谱构建器 - 使用更可靠的方法
"""

import logging
import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional
import httpx

from .config import GraphConfig
from .client import Neo4jClient
from .relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class StableKnowledgeGraphBuilder:
    """稳定的知识图谱构建器 - 使用分步处理提高成功率"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.neo4j_client = Neo4jClient(config)
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # 初始化关系提取器（使用GraphSchema）
        try:
            # 尝试从实验性组件导入
            from neo4j_graphrag.experimental.components.schema import GraphSchema
            
            # 定义业务蓝图
            # 注意：GraphSchema需要NodeType和RelationshipType对象
            from neo4j_graphrag.experimental.components.schema import NodeType, RelationshipType
            
            # 创建NodeType对象
            node_types = (
                NodeType(label="Company", description="公司"),
                NodeType(label="Department", description="部门"),
                NodeType(label="Employee", description="员工"),
                NodeType(label="Policy", description="制度"),
                NodeType(label="Contract", description="合同"),
                NodeType(label="Leave", description="假期"),
                NodeType(label="Salary", description="薪酬"),
                NodeType(label="Discipline", description="奖惩"),
                NodeType(label="Duty", description="职责"),
                NodeType(label="Process", description="流程"),
                # 新增实体标签
                NodeType(label="Standard", description="各种标准（如餐费标准、住宿标准、补贴标准）"),
                NodeType(label="Amount", description="金额（罚款金额、奖励金额、补贴金额）"),
                NodeType(label="Duration", description="时长（试用期月数、病假天数、年假天数、旷工天数）"),
                NodeType(label="Penalty", description="具体处罚措施（如警告、记过、解除劳动合同）"),
                NodeType(label="Reward", description="具体奖励措施（如嘉奖、记功、奖金）")
            )
            
            # 创建RelationshipType对象
            relationship_types = (
                RelationshipType(label="HAS_DEPARTMENT", description="公司拥有部门"),
                RelationshipType(label="WORKS_FOR", description="员工任职于公司/部门"),
                RelationshipType(label="MANAGES", description="部门管理制度/流程/员工"),
                RelationshipType(label="GOVERNS", description="制度约束员工/行为"),
                RelationshipType(label="SIGNED_WITH", description="员工签订劳动合同"),
                RelationshipType(label="ENTITLED_TO", description="员工享有假期/薪酬/奖金"),
                RelationshipType(label="TRIGGERS", description="行为触发奖惩"),
                RelationshipType(label="FOLLOWS", description="流程遵循步骤"),
                RelationshipType(label="REFERS_TO", description="条款引用其他条款或法规"),
                RelationshipType(label="SUPERVISES", description="部门监督流程"),
                # 新增关系类型
                RelationshipType(label="HAS_STANDARD", description="实体 → Standard（如出差费用有标准）"),
                RelationshipType(label="SETS_AMOUNT", description="实体 → Amount（如警告处分设置罚款50元）"),
                RelationshipType(label="SETS_DURATION", description="实体 → Duration（如试用期设置时长）"),
                RelationshipType(label="IMPOSES", description="违规行为 → Penalty（如迟到触发警告）"),
                RelationshipType(label="GRANTS", description="公司/部门 → Reward（如公司给予嘉奖）")
            )
            
            my_schema = GraphSchema(
                node_types=node_types,
                relationship_types=relationship_types
            )
            schema_available = True
            logger.info("GraphSchema可用，使用业务蓝图")
        except ImportError:
            # GraphSchema不可用，使用默认配置
            my_schema = None
            schema_available = False
            logger.warning("GraphSchema不可用，使用默认提示词")
        
        self.relation_extractor = RelationExtractor(
            api_key=config.deepseek_api_key,
            model=config.deepseek_model,
            schema=my_schema if schema_available else None  # 注入蓝图（如果可用）
        )
        
    async def extract_with_retry(self, text: str, retries: int = 3) -> Dict[str, Any]:
        """带重试的实体关系提取"""
        
        for attempt in range(retries):
            try:
                prompt = f'''请从以下文本中提取实体和关系：

文本：{text}

请严格按照以下JSON格式返回，不要有其他内容：
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|OTHER",
      "description": "简要描述"
    }}
  ],
  "relations": [
    {{
      "subject": "主体实体名称",
      "predicate": "位于|是|开发|竞争|属于|包含|拥有|工作于|毕业于|创立于|成立于",
      "object": "客体实体名称",
      "description": "关系描述"
    }}
  ]
}}

重要：请仔细分析文本中的关系，如"位于"、"是"、"开发"、"竞争"等。如果文本中有明确关系，请提取出来。'''
                
                response = await self.client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.config.deepseek_model,
                        "messages": [
                            {"role": "system", "content": "你是一个严格的实体关系提取助手，只返回JSON格式。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 清理响应，提取JSON
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.startswith('```'):
                        content = content[3:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    try:
                        data = json.loads(content)
                        # 验证数据结构
                        if isinstance(data, dict) and "entities" in data and "relations" in data:
                            return data
                        else:
                            logger.warning(f"第{attempt+1}次尝试：返回格式不正确")
                            return {"entities": [], "relations": []}
                    except json.JSONDecodeError:
                        logger.warning(f"第{attempt+1}次尝试：JSON解析失败")
                        continue
                else:
                    logger.warning(f"第{attempt+1}次尝试：API错误 {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"第{attempt+1}次尝试异常：{e}")
                await asyncio.sleep(1)  # 等待后重试
        
        # 所有重试都失败
        return {
            "success": True,
            "message": "实体提取失败",
            "statistics": {
                "entities": {"total": 0, "types": 0, "type_list": []},
                "relations": {"total": 0}
            }
        }
    
    async def process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个文档"""
        # 优先使用document_id，兼容id字段
        doc_id = doc.get("document_id", doc.get("id", str(uuid.uuid4())))
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        # 调试日志：检查文档字典内容
        logger.info(f"稳定构建器接收的文档字典: keys={list(doc.keys())}")
        logger.info(f"稳定构建器使用的doc_id: {doc_id}")
        logger.info(f"文档中document_id字段: {doc.get('document_id')}")
        logger.info(f"文档中id字段: {doc.get('id')}")
        
        # 如果document_id为空，记录警告
        if not doc.get('document_id'):
            logger.warning(f"⚠️ 文档缺少document_id字段! 使用备用ID: {doc_id}")
            logger.warning(f"   文档内容预览: {content[:100]}...")
            logger.warning(f"   文档metadata: {metadata}")
        
        if not content or len(content.strip()) < 10:
            logger.warning(f"文档 {doc_id} 内容过短")
            return {
                "success": True,
                "message": "文档内容过短",
                "statistics": {
                    "entities": {"total": 0, "types": 0, "type_list": []},
                    "relations": {"total": 0}
                }
            }
        
        logger.info(f"处理文档: {doc_id[:8]}... (长度: {len(content)}字符)")
        
        # 分块处理（避免文本过长）
        chunks = self._chunk_text(content, chunk_size=1000, overlap=100)
        
        total_entities = 0
        total_relations = 0
        entity_types = set()
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"处理块 {i+1}/{len(chunks)}")
            
            # 使用专门的关系提取器
            result = await self.relation_extractor.extract_entities_relations(chunk)
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            
            if not entities and not relations:
                continue
            
            # 存储到Neo4j
            try:
                # 创建文档节点（只创建一次）
                if i == 0:
                    doc_metadata = metadata.copy()
                    doc_metadata['chunks'] = len(chunks)
                    doc_metadata['content_preview'] = content[:300]
                    
                    success = self.neo4j_client.create_document_node(
                        document_id=doc_id,
                        metadata=doc_metadata
                    )
                    if not success:
                        logger.error(f"创建文档节点失败: {doc_id}")
                
                # 创建实体节点
                for entity in entities:
                    entity_id = f"{doc_id}_{entity['name']}_{entity['type']}_{i}"
                    entity_metadata = {
                        "source_doc": doc_id,
                        "chunk_index": i,
                        "description": entity.get("description", "")
                    }
                    
                    success = self.neo4j_client.create_entity_node(
                        entity_id=entity_id,
                        name=entity["name"],
                        entity_type=entity["type"],
                        metadata=entity_metadata
                    )
                    
                    if success:
                        total_entities += 1
                        entity_types.add(entity["type"])
                    else:
                        logger.warning(f"创建实体失败: {entity['name']}")
                
                # 创建关系（查找对应的实体）
                for relation in relations:
                    subject_name = relation.get('subject', '').strip()
                    predicate = relation.get('predicate', '').strip()
                    object_name = relation.get('object', '').strip()
                    
                    if not subject_name or not predicate or not object_name:
                        continue
                    
                    # 查找对应的实体（在同一块内）
                    subject_entity = None
                    object_entity = None
                    
                    for entity in entities:
                        if entity['name'] == subject_name:
                            subject_entity = entity
                        if entity['name'] == object_name:
                            object_entity = entity
                    
                    if not subject_entity or not object_entity:
                        continue
                    
                    # 使用正确的实体ID
                    subject_id = f"{doc_id}_{subject_entity['name']}_{subject_entity['type']}_{i}"
                    object_id = f"{doc_id}_{object_entity['name']}_{object_entity['type']}_{i}"
                    
                    relation_metadata = {
                        "source_doc": doc_id,
                        "chunk_index": i,
                        "description": relation.get("description", "")
                    }
                    
                    success = self.neo4j_client.create_relation(
                        source_id=subject_id,
                        relation_type=predicate,
                        target_id=object_id,
                        metadata=relation_metadata
                    )
                    
                    if success:
                        total_relations += 1
                        logger.debug(f"创建关系: {subject_name} -[{predicate}]-> {object_name}")
                    
            except Exception as e:
                logger.error(f"存储块 {i+1} 失败: {e}")
        
        return {
            "success": True,
            "message": f"处理文档完成",
            "statistics": {
                "entities": {
                    "total": total_entities,
                    "types": len(entity_types),
                    "type_list": list(entity_types)  # 添加类型列表
                },
                "relations": {
                    "total": total_relations
                }
            }
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """智能分块文本"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 找最近的句号、分号或换行
                for split_char in ['。', '.', ';', '\n', '！', '？']:
                    split_pos = text.rfind(split_char, start, end)
                    if split_pos > start + chunk_size * 0.5:  # 至少在一半位置
                        end = split_pos + 1  # 包含分割字符
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap  # 重叠
        
        return chunks
    
    async def build_from_documents(self, documents: List[Dict[str, Any]], 
                                  batch_size: int = 5) -> Dict[str, Any]:
        """从文档构建知识图谱"""
        
        if not documents:
            return {"success": False, "message": "没有文档", "statistics": {}}
        
        logger.info(f"稳定构建器处理 {len(documents)} 个文档")
        
        total_entities = 0
        total_relations = 0
        all_entity_types = set()
        
        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            tasks = [self.process_document(doc) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"文档 {i+j} 处理异常: {result}")
                    continue
                
                # process_document返回格式: {"statistics": {"entities": {"total": X, "types": Y}, "relations": {"total": Z}}}
                stats = result.get("statistics", {})
                entities_stats = stats.get("entities", {})
                relations_stats = stats.get("relations", {})
                
                total_entities += entities_stats.get("total", 0)
                total_relations += relations_stats.get("total", 0)
                
                # 获取实体类型
                entity_types_count = entities_stats.get("types", 0)
                type_list = entities_stats.get("type_list", [])
                
                if type_list:
                    all_entity_types.update(type_list)
                elif entity_types_count > 0:
                    # 如果没有类型列表，只能计数
                    all_entity_types.add(f"type_{len(all_entity_types)+1}")
        
        # 关闭连接（close是同步方法，不需要await）
        self.neo4j_client.close()
        
        return {
            "success": True,
            "message": f"成功处理 {len(documents)} 个文档",
            "statistics": {
                "documents": len(documents),
                "entities": {
                    "total": total_entities,
                    "types": len(all_entity_types)
                },
                "relations": {
                    "total": total_relations
                }
            }
        }
    
    async def close(self):
        """关闭资源"""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()
        
        # Neo4jClient的close()是同步方法，不需要await
        if hasattr(self.neo4j_client, 'close'):
            self.neo4j_client.close()
    
    def clear_graph(self) -> bool:
        """清空图谱"""
        try:
            with self.neo4j_client.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.warning("Knowledge graph cleared")
                return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        try:
            with self.neo4j_client.driver.session() as session:
                # 统计节点
                node_result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
                node_stats = {}
                for record in node_result:
                    node_type = record["type"] or "Unknown"
                    node_stats[node_type] = record["count"]
                
                # 统计关系
                rel_result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                rel_stats = {}
                for record in rel_result:
                    rel_type = record["type"] or "Unknown"
                    rel_stats[rel_type] = record["count"]
                
                return {
                    "node_statistics": node_stats,
                    "relationship_statistics": rel_stats,
                    "total_nodes": sum(node_stats.values()),
                    "total_relationships": sum(rel_stats.values())
                }
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}
    
    async def build_from_directory(self, directory_path: str, **kwargs) -> Dict[str, Any]:
        """从目录构建图谱（简化实现）"""
        logger.warning(f"build_from_directory not fully implemented for {directory_path}")
        return {"success": False, "message": "Method not fully implemented"}