"""
RAG类型定义
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievedDocument:
    """检索到的文档"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGQuery:
    """RAG查询"""
    query: str
    top_k: int = 3
    score_threshold: float = 0.7
    collection_name: str = "default"
    metadata_filter: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata_filter is None:
            self.metadata_filter = {}


@dataclass
class RAGResult:
    """RAG结果"""
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    model: str
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "text": doc.text[:500],  # 截断
                    "score": doc.score,
                    "metadata": doc.metadata,
                    "source": doc.source
                }
                for doc in self.retrieved_documents
            ],
            "model": self.model,
            "timing": {
                "retrieval": self.retrieval_time,
                "generation": self.generation_time,
                "total": self.total_time
            },
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }