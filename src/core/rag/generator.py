"""
生成器
负责使用LLM生成答案
"""

import time
from typing import List, Dict, Any

from src.core.config import settings
from src.core.logging import get_logger
from src.core.llm import get_llm_service
from src.core.llm.types import ChatMessage, Role
from .types import RetrievedDocument

logger = get_logger(__name__)


class Generator:
    """生成器"""
    
    def __init__(self):
        self.llm_service = get_llm_service()
        self.last_token_count: int = 0
        
        logger.info("生成器初始化",
                   model=self.llm_service.model)
    
    def _build_prompt(self, query: str, documents: List[RetrievedDocument]) -> str:
        """构建提示词"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档{i}] {doc.text}")
        
        context = "\n\n".join(context_parts)
        
        # 构建系统提示
        system_prompt = f"""你是一个专业的AI助手，请基于以下提供的文档内容回答问题。

提供的文档：
{context}

请遵循以下规则：
1. 只使用提供的文档内容回答问题
2. 如果文档中没有相关信息，请如实说明"根据提供的文档，没有找到相关信息"
3. 保持回答简洁、准确
4. 如果文档中有多个相关信息，请综合回答
5. 不要编造文档中没有的信息

用户问题：{query}

请基于以上文档内容回答："""
        
        return system_prompt
    
    def _build_chat_messages(self, query: str, documents: List[RetrievedDocument]) -> List[ChatMessage]:
        """构建聊天消息（包含图结构信息）"""
        # 构建文档上下文
        context_parts = []
        graph_context_parts = []  # 图结构上下文
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档{i}] {doc.text}")
            
            # 提取图结构信息
            metadata = doc.metadata or {}
            if "graph_structure" in metadata:
                graph_structure = metadata["graph_structure"]
                entities = metadata.get("graph_entities", [])
                relations = metadata.get("graph_relations", [])
                
                if entities or relations:
                    # 构建图结构描述
                    graph_desc = f"[文档{i}关联的知识图谱信息]"
                    
                    if entities:
                        graph_desc += f"\n相关实体: {', '.join(entities[:5])}"
                        if len(entities) > 5:
                            graph_desc += f" 等{len(entities)}个实体"
                    
                    if relations:
                        # 只显示前3个关系
                        rel_desc = []
                        for rel in relations[:3]:
                            rel_desc.append(f"{rel['source']} --[{rel['type']}]--> {rel['target']}")
                        
                        graph_desc += f"\n实体关系: {'; '.join(rel_desc)}"
                        if len(relations) > 3:
                            graph_desc += f" 等{len(relations)}个关系"
                    
                    graph_context_parts.append(graph_desc)
        
        context = "\n\n".join(context_parts)
        
        # 如果有图结构信息，添加到上下文
        if graph_context_parts:
            graph_context = "\n\n".join(graph_context_parts)
            context = f"{context}\n\n知识图谱关联信息:\n{graph_context}"
        
        # 系统消息
        system_content = """你是一个专业的AI助手，请基于用户提供的文档内容和知识图谱关联信息回答问题。

请严格遵循以下规则：
1. **必须同时使用文档内容和知识图谱信息回答问题**
2. **如果提供了知识图谱关联信息，必须在答案中明确引用**
3. 首先基于文档内容回答，然后补充知识图谱提供的关联信息
4. 知识图谱信息提供了实体之间的关系，可以帮助你理解文档之间的关联
5. 在答案中明确说明哪些信息来自文档，哪些信息来自知识图谱
6. 示例格式：
   - 根据文档X，...
   - 知识图谱关联信息显示：...，这进一步验证/补充了...
7. 保持回答简洁、准确、有帮助
8. 如果文档中有多个相关信息，请综合回答
9. 只有在文档和知识图谱都完全没有相关信息时，才说"根据提供的文档和知识图谱，没有找到相关信息"
10. 不要编造文档中没有的核心信息"""
        
        # 用户消息（包含上下文和问题）
        user_content = f"""提供的文档内容和知识图谱关联信息：
{context}

问题：{query}

请基于以上文档内容和知识图谱关联信息回答："""
        
        return [
            ChatMessage(role=Role.SYSTEM, content=system_content),
            ChatMessage(role=Role.USER, content=user_content)
        ]
    
    async def generate_answer(self, query: str, documents: List[RetrievedDocument]) -> str:
        """生成答案"""
        start_time = time.time()
        
        try:
            if not documents:
                logger.warning("没有检索到相关文档", query=query[:100])
                return "根据提供的文档，没有找到相关信息。"
            
            # 构建消息
            messages = self._build_chat_messages(query, documents)
            
            # 调用LLM
            response = await self.llm_service.chat(messages)
            
            generation_time = time.time() - start_time
            token_count = response.usage.get("total_tokens", 0)
            self.last_token_count = token_count
            
            logger.info("答案生成完成",
                       query=query[:100],
                       documents_count=len(documents),
                       generation_time=f"{generation_time:.3f}s",
                       tokens_used=token_count)
            
            return response.content
            
        except Exception as e:
            logger.error("答案生成失败",
                        query=query[:100],
                        error=str(e))
            return f"抱歉，生成答案时出现错误：{str(e)}"
    
    async def generate_stream(self, query: str, documents: List[RetrievedDocument]):
        """流式生成答案"""
        try:
            if not documents:
                yield "根据提供的文档，没有找到相关信息。"
                return
            
            # 构建消息
            messages = self._build_chat_messages(query, documents)
            
            # 流式调用LLM
            async for chunk in self.llm_service.chat_stream(messages):
                yield chunk
                
        except Exception as e:
            logger.error("流式答案生成失败",
                        query=query[:100],
                        error=str(e))
            yield f"抱歉，生成答案时出现错误：{str(e)}"
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            llm_health = await self.llm_service.health_check()
            
            return {
                "status": "healthy" if llm_health.get("status") == "healthy" else "degraded",
                "llm": llm_health,
                "model": self.llm_service.model
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.llm_service.model
            }


# 全局生成器实例
_generator_instance = None

def get_generator() -> Generator:
    """获取生成器实例（单例模式）"""
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = Generator()
    
    return _generator_instance