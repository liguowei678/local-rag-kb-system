"""
独立GraphRAG检索器 - 按照新要求实现
1. 向量检索top_k=2片段
2. 大模型从片段中提取实体（避免无关实体）
3. 知识图谱查询实体关系（最多3跳）
4. 生成自然语言关联子图描述
"""

import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.core.rag.types import RAGQuery, RetrievedDocument
from .config import GraphConfig
from .client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class IndependentGraphRAGConfig:
    """独立GraphRAG配置"""
    vector_top_k: int = 2  # 向量检索返回top_k=2片段
    max_hop_depth: int = 3  # 最多3跳
    timeout_seconds: float = 10.0  # 超时时间
    min_entity_relevance: float = 0.5  # 实体相关性阈值


class IndependentGraphRAGRetriever:
    """独立GraphRAG检索器 - 按照新要求实现"""
    
    def __init__(self, 
                 config: GraphConfig,
                 semantic_retriever,  # 语义检索器
                 retriever_config: Optional[IndependentGraphRAGConfig] = None):
        
        self.config = config
        self.retriever_config = retriever_config or IndependentGraphRAGConfig()
        self.semantic_retriever = semantic_retriever
        
        # 初始化Neo4j客户端
        self.neo4j_client = Neo4jClient(config)
        
        # 初始化HTTP客户端用于DeepSeek API调用
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.deepseek_api_key = config.deepseek_api_key
        self.deepseek_model = config.deepseek_model
        self.deepseek_base_url = config.deepseek_base_url or "https://api.deepseek.com"
        
        logger.info("独立GraphRAG检索器初始化完成")
    
    async def retrieve(self, query: RAGQuery) -> Tuple[List[RetrievedDocument], str]:
        """
        执行独立GraphRAG检索
        
        Returns:
            (文档列表, 关联子图自然语言描述)
        """
        try:
            # 设置超时
            task = asyncio.create_task(self._retrieve_internal(query))
            return await asyncio.wait_for(task, timeout=self.retriever_config.timeout_seconds)
            
        except asyncio.TimeoutError:
            logger.warning(f"GraphRAG retrieval timeout after {self.retriever_config.timeout_seconds}s")
            return [], ""
        except Exception as e:
            logger.error(f"GraphRAG retrieval failed: {e}")
            return [], ""
    
    async def _retrieve_internal(self, query: RAGQuery) -> Tuple[List[RetrievedDocument], str]:
        """内部检索实现"""
        logger.info(f"开始独立GraphRAG检索: {query.query}")
        
        # 步骤1: 向量检索top_k=2片段
        vector_docs = await self._vector_retrieval_top2(query)
        if not vector_docs:
            logger.warning("向量检索未找到相关片段")
            return [], ""
        
        logger.info(f"向量检索找到 {len(vector_docs)} 个最相关片段")
        
        # 步骤2: 大模型从片段中提取实体（避免无关实体）
        entities = await self._extract_relevant_entities(query.query, vector_docs)
        if not entities:
            logger.warning("未从向量检索片段中提取到相关实体")
            return [], ""
        
        logger.info(f"提取到相关实体: {entities}")
        
        # 步骤3: 知识图谱查询实体关系（最多3跳）
        graph_result = await self._query_entity_relations(entities)
        if not graph_result:
            logger.warning("知识图谱未找到实体关系")
            return [], ""
        
        # 步骤4: 生成自然语言关联子图描述
        graph_description = self._generate_graph_description(graph_result)
        
        # 步骤5: 返回文档和关联子图描述
        documents = self._create_documents_from_graph(graph_result)
        
        logger.info(f"独立GraphRAG检索完成，找到 {len(documents)} 个文档")
        return documents, graph_description
    
    async def _vector_retrieval_top2(self, query: RAGQuery) -> List[RetrievedDocument]:
        """步骤1: 向量检索top_k=2片段"""
        try:
            if not self.semantic_retriever:
                logger.warning("语义检索器未提供，跳过向量检索")
                return []
            
            # 创建新的查询对象，只获取top_k=2
            from src.core.rag.types import RAGQuery
            vector_query = RAGQuery(
                query=query.query,
                top_k=self.retriever_config.vector_top_k,  # 只获取2个片段
                score_threshold=0.4,  # 提高阈值确保相关性
                collection_name=query.collection_name,
                metadata_filter=query.metadata_filter
            )
            
            # 执行语义检索
            return await self.semantic_retriever.retrieve(vector_query)
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    async def _extract_relevant_entities(self, query_text: str, vector_docs: List[RetrievedDocument]) -> List[str]:
        """步骤2: 大模型从片段中提取实体（避免无关实体）"""
        try:
            # 从向量检索结果中提取文本内容，但先计算句子相似度
            relevant_sentences = await self._extract_most_relevant_sentences(query_text, vector_docs)
            
            if not relevant_sentences:
                logger.warning("未找到相关句子")
                return self._fallback_entity_extraction(query_text)
            
            # 构建提示词 - 严格避免无关实体
            sentences_text = "\n".join(relevant_sentences)
            prompt = f"""请从以下查询和相关的句子中提取关键实体，但必须严格遵守以下规则：

查询: {query_text}

相关句子:
{sentences_text}

提取规则（必须遵守）：
1. 只提取与查询直接相关的实体
2. 不要提取"标准"、"多少"、"什么"、"如何"等非实体词
3. 不要提取过于宽泛的实体（如"公司"、"员工"等，除非在上下文中特别重要）
4. 优先提取具体的、有明确含义的实体
5. 对于"出差住宿费的标准是多少"，应该提取"出差"和"住宿费"
6. 如果实体在句子中多次出现且与查询高度相关，才提取
7. 返回格式：每行一个实体

实体列表（只返回真正相关的）："""
            
            # 使用DeepSeek API提取实体
            try:
                response = await self.http_client.post(
                        f"{self.deepseek_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.deepseek_model,
                        "messages": [
                            {"role": "system", "content": "你是一个严格的实体提取助手，只返回实体列表，每行一个。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    entities_text = result["choices"][0]["message"]["content"]
                    
                    # 解析响应
                    entities = []
                    for line in entities_text.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and ':' not in line:
                            # 清理可能的编号和标点
                            line = line.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '')
                            line = line.replace('、', '').replace('，', '').replace('。', '')
                            line = line.replace('标准', '').replace('多少', '').replace('什么', '')  # 过滤无关词
                            line = line.strip()
                            if line and len(line) > 1:  # 确保不是单个字符
                                entities.append(line)
                    
                    # 去重
                    entities = list(set(entities))
                    logger.info(f"大模型提取的相关实体: {entities}")
                    return entities
                else:
                    logger.error(f"DeepSeek API错误: {response.status_code}")
                    return self._fallback_entity_extraction(query_text)
                    
            except Exception as e:
                logger.error(f"大模型实体提取失败: {e}")
                return self._fallback_entity_extraction(query_text)
            
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return self._fallback_entity_extraction(query_text)
    
    async def _extract_most_relevant_sentences(self, query_text: str, vector_docs: List[RetrievedDocument]) -> List[str]:
        """提取每个片段中最相关的1-2个句子"""
        try:
            relevant_sentences = []
            
            for doc in vector_docs:
                # 将文档文本分割成句子
                sentences = self._split_into_sentences(doc.text)
                
                if not sentences:
                    continue
                
                # 计算每个句子与查询的相似度（使用简单的文本匹配）
                sentence_scores = []
                for sentence in sentences:
                    score = self._calculate_sentence_relevance(query_text, sentence)
                    sentence_scores.append((score, sentence))
                
                # 按相似度排序，取前1-2个句子
                sentence_scores.sort(reverse=True, key=lambda x: x[0])
                top_sentences = [s for _, s in sentence_scores[:2] if _ > 0.1]  # 阈值过滤
                
                if top_sentences:
                    relevant_sentences.extend(top_sentences)
                    logger.debug(f"文档中提取到 {len(top_sentences)} 个相关句子")
            
            logger.info(f"总共提取到 {len(relevant_sentences)} 个相关句子")
            return relevant_sentences
            
        except Exception as e:
            logger.error(f"提取相关句子失败: {e}")
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """简单的中文句子分割"""
        # 使用常见的中文句子分隔符
        import re
        # 中文句子结束符：。！？；
        sentences = re.split(r'[。！？；]+', text)
        # 过滤空字符串和过短的句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences
    
    def _calculate_sentence_relevance(self, query: str, sentence: str) -> float:
        """计算句子与查询的相关性（简化版）"""
        # 简单的关键词匹配
        query_words = set(query)
        sentence_words = set(sentence)
        
        # 共同字符数
        common_chars = len(query_words.intersection(sentence_words))
        
        # 归一化分数
        if len(query_words) == 0:
            return 0.0
        
        return common_chars / len(query_words)
    
    def _fallback_entity_extraction(self, query_text: str) -> List[str]:
        """降级实体提取"""
        # 简单分词提取，但过滤无关词
        import jieba
        words = jieba.lcut(query_text)
        # 过滤掉停用词和短词
        stop_words = {'的', '是', '在', '了', '和', '与', '等', '多少', '标准', '什么', '如何', '怎样', '公司', '员工'}
        entities = [word for word in words if len(word) > 1 and word not in stop_words]
        logger.info(f"降级实体提取结果: {entities}")
        return entities
    
    async def _query_entity_relations(self, entities: List[str]) -> Dict[str, Any]:
        """步骤3: 知识图谱查询实体关系（最多3跳）"""
        try:
            if not entities:
                return {}
            
            # 查询每个实体的关系（最多3跳）
            all_relations = []
            all_entities = set(entities)  # 存储所有找到的实体
            
            for entity in entities:
                entity_relations = self._query_entity_relations_with_depth(entity)
                if entity_relations:
                    all_relations.extend(entity_relations)
                    # 添加相关实体
                    for rel in entity_relations:
                        all_entities.add(rel.get("source"))
                        all_entities.add(rel.get("target"))
            
            if not all_relations:
                logger.warning("未找到实体关系")
                return {}
            
            # 构建结果
            result = {
                "entities": list(all_entities),
                "relations": all_relations,
                "graph_structure": {
                    "node_count": len(all_entities),
                    "edge_count": len(all_relations),
                    "relation_types": list(set([r.get("type", "") for r in all_relations if r.get("type")]))
                }
            }
            
            logger.info(f"找到 {len(all_entities)} 个实体和 {len(all_relations)} 个关系")
            return result
            
        except Exception as e:
            logger.error(f"查询实体关系失败: {e}")
            return {}
    
    def _query_entity_relations_with_depth(self, entity: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        查询实体关系（多跳推理）
        
        支持多跳推理：从起始实体出发，沿着关系链最多走max_depth步。
        返回所有路径上的实体和关系。
        
        例如：Neo4j中如果有：
          住宿费 --[RELATED_TO]--> 出差费用
          出差费用 --[FOLLOWS]--> 公司财务部门
          出差费用 --[MANAGES]--> 部门主管
        
        查询"住宿费"(max_depth=3)会返回：
          - 1跳: 住宿费→出差费用, 住宿费→500元, 住宿费→300元
          - 2跳: 出差费用→公司财务部门, 出差费用→部门主管
          - 3跳: 公司财务部门→财务部门, 部门主管→经理...
        
        这使得LLM能推理出"住宿费受公司财务部门约束"这样的结论。
        """
        try:
            with self.neo4j_client.driver.session() as session:
                # 多跳查询：从起始实体出发，沿任意关系走最多max_depth步
                # 注意：Neo4j的Cypher不支持在[*1..$max_depth]中使用参数，必须用f-string拼接
                cypher = f"""
                    MATCH path = (start:Entity {{name: $entity_name}})-[*1..{max_depth}]-(end:Entity)
                    WHERE start <> end
                    WITH path, length(path) as depth
                    ORDER BY depth ASC
                    LIMIT 200
                    RETURN 
                        [node in nodes(path) | node.name] as node_names,
                        [rel in relationships(path) | type(rel)] as rel_types,
                        depth
                """
                result = session.run(cypher, entity_name=entity)
                
                # 收集所有实体和关系
                all_entities = set()
                all_relations = []
                seen_pairs = set()
                
                for record in result:
                    node_names = record["node_names"]
                    rel_types = record["rel_types"]
                    
                    for name in node_names:
                        if name:
                            all_entities.add(name)
                    
                    for i, rel_type in enumerate(rel_types):
                        if i + 1 < len(node_names):
                            source = node_names[i]
                            target = node_names[i + 1]
                            pair_key = (source, target, rel_type)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                all_relations.append({
                                    "source": source,
                                    "target": target,
                                    "type": rel_type
                                })
                
                if not all_relations:
                    logger.debug(f"实体 '{entity}' 在Neo4j中无关联")
                    return []
                
                # 把起始实体也加入（如果不在的话）
                if entity not in all_entities:
                    all_entities.add(entity)
                
                logger.info(
                    f"多跳推理: 从'{entity}'出发(max_depth={max_depth}), "
                    f"找到 {len(all_entities)} 个实体, {len(all_relations)} 个关系"
                )
                return all_relations
                
        except Exception as e:
            logger.error(f"查询实体关系失败: {e}")
            return []
    
    def _query_multi_level_relations(self, entity: str, relation_type: str = None) -> List[Dict[str, Any]]:
        """查询多级关系，支持复杂推理"""
        try:
            with self.neo4j_client.driver.session() as session:
                # 查找多级关系链
                if relation_type:
                    # 指定关系类型
                    result = session.run("""
                        MATCH path = (e1:Entity {name: $entity_name})-[r:%s*1..5]-(e2:Entity)
                        RETURN 
                            nodes(path) as nodes,
                            relationships(path) as relations,
                            length(path) as depth
                        ORDER BY depth ASC
                        LIMIT 20
                    """ % relation_type, entity_name=entity)
                else:
                    # 查找所有关系类型
                    result = session.run("""
                        MATCH path = (e1:Entity {name: $entity_name})-[*1..5]-(e2:Entity)
                        RETURN 
                            nodes(path) as nodes,
                            relationships(path) as relations,
                            length(path) as depth
                        ORDER BY depth ASC
                        LIMIT 20
                    """, entity_name=entity)
                
                chains = []
                for record in result:
                    nodes = record["nodes"]
                    relations = record["relations"]
                    depth = record["depth"]
                    
                    chain = {
                        "depth": depth,
                        "nodes": [node["name"] for node in nodes],
                        "relations": [rel.type for rel in relations],
                        "full_path": []
                    }
                    
                    # 构建完整路径
                    for i in range(len(nodes) - 1):
                        chain["full_path"].append({
                            "from": nodes[i]["name"],
                            "to": nodes[i+1]["name"],
                            "relation": relations[i].type
                        })
                    
                    chains.append(chain)
                
                return chains
                
        except Exception as e:
            logger.error(f"查询多级关系失败: {e}")
            return []
    
    def _generate_graph_description(self, graph_result: Dict[str, Any]) -> str:
        """步骤4: 生成自然语言关联子图描述（含多跳推理链路）"""
        try:
            entities = graph_result.get("entities", [])
            relations = graph_result.get("relations", [])
            
            if not relations:
                return "知识图谱中未找到相关实体关系。"
            
            description_parts = []
            
            # 1. 实体概述
            if entities:
                entity_list = "、".join(entities[:8])
                if len(entities) > 8:
                    entity_list += f" 等{len(entities)}个实体"
                description_parts.append(f"涉及实体包括：{entity_list}")
            
            # 2. 输出多跳关联路径（使用邻接表去重）
            seen_pairs = set()
            path_parts = []
            for rel in relations:
                src = rel.get("source", "")
                tgt = rel.get("target", "")
                rtype = rel.get("type", "")
                pair_key = tuple(sorted([src, tgt]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                chinese_rel = self._translate_relation_type(rtype)
                path_parts.append(f"{src}{chinese_rel}{tgt}")
            
            if path_parts:
                description_parts.append("知识图谱关联路径：" + "；".join(path_parts[:12]))
            
            # 3. 图结构总结
            graph_structure = graph_result.get("graph_structure", {})
            if graph_structure:
                node_count = graph_structure.get("node_count", 0)
                edge_count = graph_structure.get("edge_count", 0)
                if node_count > 0 and edge_count > 0:
                    description_parts.append(f"关联子图共包含 {node_count} 个实体和 {edge_count} 个关系")
            
            if description_parts:
                description = "知识图谱分析显示：" + "。".join(description_parts) + "。"
            else:
                description = "知识图谱中未找到相关实体关系。"
            
            logger.info(f"生成关联子图描述: {description[:300]}...")
            return description
            
        except Exception as e:
            logger.error(f"生成图描述失败: {e}")
            return "知识图谱关联信息生成失败。"
    
    def _translate_relation_type(self, relation_type: str) -> str:
        """将关系类型翻译为中文描述"""
        translations = {
            "GOVERNS": "约束",
            "WORKS_FOR": "任职于",
            "HAS_DEPARTMENT": "拥有部门",
            "MANAGES": "管理",
            "REFERS_TO": "引用",
            "RELATED_TO": "相关于",
            "SUPERVISES": "监督",
            "HAS_STANDARD": "有标准",
            "SETS_AMOUNT": "设置金额",
            "SETS_DURATION": "设置时长",
            "IMPOSES": "施加处罚",
            "GRANTS": "给予奖励",
            "FOLLOWS": "遵循"
        }
        return translations.get(relation_type, f"({relation_type})")
    
    def _create_documents_from_graph(self, graph_result: Dict[str, Any]) -> List[RetrievedDocument]:
        """从图结果创建文档"""
        try:
            documents = []
            entities = graph_result.get("entities", [])
            
            for i, entity in enumerate(entities[:5]):  # 最多5个文档
                doc = RetrievedDocument(
                    id=f"graph_entity_{entity}_{i}",
                    text=f"与实体'{entity}'相关的知识图谱信息",
                    metadata={
                        "source": "independent_graphrag",
                        "score": 0.7,  # 默认分数
                        "entity": entity,
                        "retrieval_type": "graphrag",
                        "graph_entities": entities,
                        "graph_relations": graph_result.get("relations", []),
                        "graph_structure": graph_result.get("graph_structure", {})
                    },
                    score=0.7
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"创建图文档失败: {e}")
            return []
    
    def close(self):
        """清理资源"""
        self.neo4j_client.close()
