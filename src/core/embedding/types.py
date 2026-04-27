"""
嵌入服务类型定义
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """单个文本嵌入结果"""
    vector: List[float]
    text: str
    dimension: int
    model: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchEmbeddingResult:
    """批量文本嵌入结果"""
    vectors: List[List[float]]
    texts: List[str]
    dimension: int
    model: str
    metadata: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = [{} for _ in range(len(self.texts))]


class EmbeddingConfig(TypedDict, total=False):
    """嵌入服务配置"""
    model_name: str
    device: str
    batch_size: int
    normalize: bool
    show_progress_bar: bool