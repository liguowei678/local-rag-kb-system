"""
自定义DeepSeek LLM类 - 禁用tools/functions支持
"""

import logging
from typing import Any, Dict, List, Optional
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)


class DeepSeekLLM(OpenAILLM):
    """自定义DeepSeek LLM类，禁用tools/functions支持"""
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 base_url: str = "https://api.deepseek.com",
                 **kwargs):
        """
        初始化DeepSeek LLM
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
            **kwargs: 传递给OpenAILLM的其他参数
        """
        # 确保禁用tools/functions
        kwargs.setdefault('model_params', {})
        kwargs['model_params'].update({
            'temperature': 0.1,
            'max_tokens': 2000
        })
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    @property
    def supports_structured_output(self) -> bool:
        """DeepSeek标准API不支持structured output（tools/functions）"""
        return False
    
    def invoke_with_tools(self, 
                         input: str, 
                         message_history: Optional[List[Dict[str, Any]]] = None,
                         system_instruction: Optional[str] = None,
                         **kwargs) -> str:
        """
        重写invoke_with_tools，使用标准invoke代替
        
        DeepSeek标准API不支持tools/functions，所以回退到标准调用
        """
        logger.warning("DeepSeek标准API不支持tools/functions，使用标准调用代替")
        return self.invoke(input, message_history, system_instruction, **kwargs)
    
    async def ainvoke_with_tools(self,
                                input: str,
                                message_history: Optional[List[Dict[str, Any]]] = None,
                                system_instruction: Optional[str] = None,
                                **kwargs) -> str:
        """
        异步版本的invoke_with_tools
        """
        logger.warning("DeepSeek标准API不支持tools/functions，使用标准调用代替")
        return await self.ainvoke(input, message_history, system_instruction, **kwargs)