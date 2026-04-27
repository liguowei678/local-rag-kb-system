"""
专门的关系提取器 - 解决DeepSeek返回空实体的问题
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class RelationExtractor:
    """专门的关系提取器"""

    def __init__(self, api_key: str, model: str = "deepseek-chat", schema=None):
        self.api_key = api_key
        self.model = model
        self.schema = schema  # GraphSchema对象
        self.client = httpx.AsyncClient(timeout=60.0)

    async def extract_entities_relations(self, text: str) -> Dict[str, Any]:
        """提取实体和关系 - 使用分步方法"""

        # 第一步:提取实体
        entities = await self._extract_entities(text)
        if not entities:
            return {"entities": [], "relations": []}

        # 第二步:提取关系(基于已识别的实体)
        relations = await self._extract_relations(text, entities)

        return {
            "entities": entities,
            "relations": relations
        }

    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """专门提取实体"""

        # 使用GraphSchema或默认提示词
        if self.schema:
            # 从GraphSchema生成提示词
            node_types = self.schema.node_types
            # 提取标签列表
            node_labels = [node_type.label for node_type in node_types]
            node_descriptions = [f"{node_type.label}（{node_type.description}）" for node_type in node_types]
            
            prompt = f'''请从以下文本中识别所有命名实体：

文本：{text}

请识别以下类型的实体：
{chr(10).join([f'{i+1}. {desc}' for i, desc in enumerate(node_descriptions)])}

格式：
- 名称：[实体名称]，类型：[实体类型]

示例：
文本："公司人力资源部负责招聘流程管理"
实体：
- 名称：公司，类型：Company
- 名称：人力资源部，类型：Department
- 名称：招聘流程，类型：Process

请直接返回实体列表，不要解释。'''
        else:
            # 最简单的基础提示词（先确保能提取实体）
            prompt = f'''请从以下文本中提取所有命名实体：

文本：{text}

请提取实体并分类。实体类型包括：公司、部门、员工、制度、合同、假期、薪酬、奖惩、职责、流程。

请返回JSON格式：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型"}}
  ]
}}

示例：
文本："公司人力资源部负责招聘流程"
返回：
{{
  "entities": [
    {{"name": "公司", "type": "公司"}},
    {{"name": "人力资源部", "type": "部门"}},
    {{"name": "招聘流程", "type": "流程"}}
  ]
}}'''

        try:
            response = await self.client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个命名实体识别专家,准确识别文本中的实体。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # 解析实体
                entities = []
                lines = content.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('文本:') or line.startswith('实体:'):
                        continue

                    # 解析格式:"- 名称:XXX,类型:YYY" 或 JSON格式
                    # 先尝试JSON格式
                    try:
                        import json
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            parsed = json.loads(json_str)
                            if "entities" in parsed:
                                for entity in parsed["entities"]:
                                    if "name" in entity and "type" in entity:
                                        entities.append({
                                            "name": entity["name"],
                                            "type": entity["type"],
                                            "description": ""
                                        })
                                logger.info(f"JSON格式提取到 {len(entities)} 个实体")
                                return entities
                    except (json.JSONDecodeError, Exception):
                        pass
                    
                    # 如果JSON解析失败，尝试行格式
                    match = re.search(r'名称[：:]\s*([^，,]+)[，,]\s*类型[：:]\s*([^\s]+)', line)
                    if match:
                        name = match.group(1).strip()
                        entity_type = match.group(2).strip()
                        if name and entity_type:
                            entities.append({
                                "name": name,
                                "type": entity_type,
                                "description": ""
                            })

                logger.info(f"提取到 {len(entities)} 个实体")
                return entities

        except Exception as e:
            logger.error(f"实体提取失败: {e}")

        return []

    async def _extract_relations(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """基于实体提取关系"""

        if len(entities) < 2:
            return []

        # 构建实体列表
        entity_list = "\n".join([f"- {e['name']} ({e['type']})" for e in entities])

        # 使用GraphSchema或默认提示词
        if self.schema:
            # 从GraphSchema生成提示词
            relationship_types = self.schema.relationship_types
            # 提取关系类型和描述
            rel_descriptions = [f"{rel_type.label}（{rel_type.description}）" for rel_type in relationship_types]
            
            prompt = f'''文本:{text}

已识别的实体:
{entity_list}

请分析这些实体之间的关系。关系类型：
{chr(10).join([f'{i+1}. {desc}' for i, desc in enumerate(rel_descriptions)])}

格式:
主体实体 -[关系类型]-> 客体实体

示例:
公司 -[HAS_DEPARTMENT]-> 人力资源部
员工张三 -[WORKS_FOR]-> 人力资源部

请直接返回关系列表,不要解释。'''
        else:
            # 默认提示词
            prompt = f'''文本:{text}

已识别的实体:
{entity_list}

请分析这些实体之间的关系。关系类型：
1. HAS_DEPARTMENT（公司拥有部门）
2. WORKS_FOR（员工任职于公司/部门）
3. MANAGES（部门管理制度/流程/员工）
4. GOVERNS（制度约束员工/行为）
5. SIGNED_WITH（员工签订劳动合同）
6. ENTITLED_TO（员工享有假期/薪酬/奖金）
7. TRIGGERS（行为触发奖惩）
8. FOLLOWS（流程遵循步骤）
9. REFERS_TO（条款引用其他条款或法规）
10. SUPERVISES（部门监督流程）

格式:
主体实体 -[关系类型]-> 客体实体

示例:
公司 -[HAS_DEPARTMENT]-> 人力资源部
员工张三 -[WORKS_FOR]-> 人力资源部

请直接返回关系列表,不要解释。'''

        try:
            response = await self.client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个关系提取专家,准确识别实体之间的关系。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # 解析关系
                relations = []
                lines = content.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line or '->' not in line:
                        continue

                    # 解析格式:主体 -[关系]-> 客体
                    match = re.search(r'([^-[]+)\s*-\[([^\]]+)\]\s*->\s*([^\[\]]+)', line)
                    if match:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                        obj = match.group(3).strip()

                        if subject and predicate and obj:
                            # 直接使用英文关系类型(用户定义的10种)
                            # 验证关系类型是否在允许的列表中
                            allowed_relations = [
                                "HAS_DEPARTMENT", "WORKS_FOR", "MANAGES", "GOVERNS",
                                "SIGNED_WITH", "ENTITLED_TO", "TRIGGERS", "FOLLOWS",
                                "REFERS_TO", "SUPERVISES"
                            ]

                            # 清理关系类型,确保是大写
                            predicate_clean = predicate.strip().upper()

                            # 如果关系类型不在允许列表中,使用默认
                            if predicate_clean not in allowed_relations:
                                predicate_clean = "RELATED_TO"
                                logger.warning(f"未知关系类型: {predicate}, 使用默认: RELATED_TO")

                            english_predicate = predicate_clean

                            relations.append({
                                    "subject": subject,
                                    "predicate": english_predicate,  # 使用英文
                                    "object": obj,
                                    "description": f"{subject} {predicate} {obj}"
                                })

                logger.info(f"提取到 {len(relations)} 个关系")
                return relations

        except Exception as e:
            logger.error(f"关系提取失败: {e}")

        return []

    async def close(self):
        """关闭连接"""
        await self.client.aclose()