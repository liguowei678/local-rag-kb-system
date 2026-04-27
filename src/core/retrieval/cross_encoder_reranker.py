"""
Cross-Encoder重排序器

使用本地下载的BGE-Reranker模型对检索结果进行重排序：
1. 更准确地计算查询和文档的相关性
2. 解决语义检索的模糊性问题
3. 提升最终答案的质量

企业RAG项目中的重排序应用：
1. 区分相似但不相关的文档（警告处分 vs 记过处分）
2. 提升具体查询的准确性
3. 过滤低质量检索结果
4. 优化多文档答案生成
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RerankResult:
    """重排序结果"""
    id: str
    text: str
    original_score: float  # 原始检索分数
    rerank_score: float    # 重排序分数
    hybrid_score: float    # 混合分数
    metadata: Dict[str, Any]
    retrieval_type: str


class CrossEncoderReranker:
    """Cross-Encoder重排序器"""
    
    def __init__(self,
                 model_path: str = "/app/models/reranker",
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu",
                 max_length: int = 512,
                 batch_size: int = 8,
                 use_fp16: bool = False):
        """
        初始化重排序器
        
        Args:
            model_path: 本地模型路径
            model_name: 模型名称（用于fallback）
            device: 运行设备（cpu/cuda）
            max_length: 最大序列长度
            batch_size: 批处理大小
            use_fp16: 是否使用半精度
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # 模型和tokenizer
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # 缓存
        self.score_cache = {}
        self.max_cache_size = 1000
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            print(f"正在加载重排序模型: {self.model_path}")
            
            # 首先尝试从本地路径加载
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                ).to(self.device)
                
                print(f"[OK] 从本地路径加载模型成功: {self.model_path}")
            else:
                # fallback到在线加载
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
                
                print(f"[OK] 从在线加载模型成功: {self.model_name}")
            
            # 设置为评估模式
            self.model.eval()
            
            # 使用半精度（如果支持）
            if self.use_fp16 and self.device != 'cpu':
                self.model.half()
            
            print(f"模型加载完成，设备: {self.device}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("重排序功能将不可用")
            self.model = None
            self.tokenizer = None
    
    def rerank(self,
               query: str,
               documents: List[Dict[str, Any]],
               top_k: int = 10,
               score_threshold: float = 0.0,
               original_weight: float = 0.4,
               rerank_weight: float = 0.6) -> List[RerankResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回结果数量
            score_threshold: 分数阈值
            original_weight: 原始分数权重
            rerank_weight: 重排序分数权重
            
        Returns:
            重排序后的结果列表
        """
        if not self.model or not self.tokenizer:
            print("重排序模型未加载，返回原始结果")
            return self._format_original_results(documents, top_k)
        
        if not documents:
            return []
        
        start_time = time.time()
        
        try:
            # 准备输入对
            pairs = [(query, doc.get('text', '')) for doc in documents]
            
            # 批量计算分数
            rerank_scores = self._batch_score_pairs(pairs)
            
            # 构建结果
            results = []
            for i, doc in enumerate(documents):
                if i >= len(rerank_scores):
                    break
                
                # 获取原始分数（归一化到0-1）
                original_score = doc.get('score', 0.0)
                original_score_norm = self._normalize_score(original_score, doc.get('retrieval_type', 'unknown'))
                
                # 获取重排序分数（sigmoid转换）
                rerank_score = rerank_scores[i]
                rerank_score_norm = 1 / (1 + np.exp(-rerank_score))  # sigmoid
                
                # 计算混合分数
                hybrid_score = (
                    original_weight * original_score_norm +
                    rerank_weight * rerank_score_norm
                )
                
                # 创建结果对象
                result = RerankResult(
                    id=doc.get('id', str(i)),
                    text=doc.get('text', ''),
                    original_score=original_score,
                    rerank_score=rerank_score,
                    hybrid_score=hybrid_score,
                    metadata=doc.get('metadata', {}),
                    retrieval_type=doc.get('retrieval_type', 'unknown')
                )
                
                results.append(result)
            
            # 过滤和排序
            filtered_results = [
                r for r in results 
                if r.hybrid_score >= score_threshold
            ]
            sorted_results = sorted(
                filtered_results, 
                key=lambda x: x.hybrid_score, 
                reverse=True
            )
            
            elapsed_time = time.time() - start_time
            print(f"重排序完成: {len(documents)}个文档 → {len(sorted_results)}个结果，耗时: {elapsed_time:.2f}s")
            
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"重排序失败: {e}")
            # 返回原始结果作为fallback
            return self._format_original_results(documents, top_k)
    
    def _batch_score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """批量计算查询-文档对分数"""
        if not pairs:
            return []
        
        # 检查缓存
        cached_scores = []
        uncached_pairs = []
        uncached_indices = []
        
        for i, (query, doc) in enumerate(pairs):
            cache_key = f"{query[:50]}_{hash(doc[:100])}"
            if cache_key in self.score_cache:
                cached_scores.append((i, self.score_cache[cache_key]))
            else:
                uncached_pairs.append((query, doc))
                uncached_indices.append(i)
        
        # 计算未缓存的分数
        if uncached_pairs:
            uncached_scores = self._compute_scores(uncached_pairs)
            
            # 更新缓存
            for idx, score in zip(uncached_indices, uncached_scores):
                query, doc = pairs[idx]
                cache_key = f"{query[:50]}_{hash(doc[:100])}"
                self.score_cache[cache_key] = score
                
                # 维护缓存大小
                if len(self.score_cache) > self.max_cache_size:
                    # 移除最旧的缓存项
                    oldest_key = next(iter(self.score_cache))
                    del self.score_cache[oldest_key]
        
        # 合并所有分数
        all_scores = [0.0] * len(pairs)
        
        # 填充缓存分数
        for idx, score in cached_scores:
            all_scores[idx] = score
        
        # 填充新计算分数
        for i, idx in enumerate(uncached_indices):
            all_scores[idx] = uncached_scores[i]
        
        return all_scores
    
    def _compute_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """计算查询-文档对分数"""
        if not pairs:
            return []
        
        scores = []
        
        # 分批处理
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            try:
                # 编码
                features = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 推理
                with torch.no_grad():
                    batch_scores = self.model(**features).logits.squeeze()
                
                # 转换为Python float
                if batch_scores.dim() == 0:  # 单个分数
                    scores.append(float(batch_scores))
                else:
                    scores.extend([float(s) for s in batch_scores])
                    
            except Exception as e:
                print(f"批处理推理失败: {e}")
                # 为失败的批次添加默认分数
                scores.extend([0.0] * len(batch_pairs))
        
        return scores
    
    def _normalize_score(self, score: float, retrieval_type: str) -> float:
        """归一化分数到0-1范围"""
        # 不同检索类型的分数范围不同
        if retrieval_type == 'semantic':
            # 语义检索：余弦相似度，范围[-1, 1] -> [0, 1]
            return (score + 1) / 2
        elif retrieval_type == 'bm25':
            # BM25：通常为正数，但范围不确定
            # 简单归一化：sigmoid(score/10)
            return 1 / (1 + np.exp(-score / 10))
        else:
            # 其他类型：假设已经在0-1范围
            return max(0.0, min(1.0, score))
    
    def _format_original_results(self, 
                                documents: List[Dict[str, Any]], 
                                top_k: int) -> List[RerankResult]:
        """格式化原始结果（当重排序不可用时）"""
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RerankResult(
                id=doc.get('id', str(i)),
                text=doc.get('text', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),  # 使用原始分数
                hybrid_score=doc.get('score', 0.0),  # 使用原始分数
                metadata=doc.get('metadata', {}),
                retrieval_type=doc.get('retrieval_type', 'unknown')
            )
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """清空分数缓存"""
        self.score_cache.clear()
        print("重排序缓存已清空")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_path': self.model_path,
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'cache_size': len(self.score_cache),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }


# 测试函数
if __name__ == "__main__":
    # 创建测试数据
    test_query = "警告处分会受到什么处罚？"
    
    test_documents = [
        {
            'id': 'doc1',
            'text': '根据《员工手册》规定，警告处分适用于情节轻微的违纪行为，扣款50元。',
            'score': 0.85,
            'retrieval_type': 'semantic',
            'metadata': {'source': '员工手册'}
        },
        {
            'id': 'doc2',
            'text': '记过处分适用于情节较为严重的违纪行为，扣款200元。',
            'score': 0.78,
            'retrieval_type': 'semantic', 
            'metadata': {'source': '员工手册'}
        },
        {
            'id': 'doc3',
            'text': '员工旷工1天给予警告处分，并从工资中扣款五十元。',
            'score': 0.72,
            'retrieval_type': 'bm25',
            'metadata': {'source': '考勤制度'}
        },
        {
            'id': 'doc4',
            'text': '工作上消极怠工，情节轻微者给予警告处分，扣款50元。',
            'score': 0.68,
            'retrieval_type': 'semantic',
            'metadata': {'source': '员工手册'}
        },
        {
            'id': 'doc5',
            'text': '年度优秀员工可获得奖励，包括奖金和表彰。',
            'score': 0.45,
            'retrieval_type': 'semantic',
            'metadata': {'source': '奖励制度'}
        }
    ]
    
    print("重排序测试")
    print("=" * 80)
    
    # 创建重排序器
    reranker = CrossEncoderReranker(
        model_path="C:/Users/86158/.openclaw/workspace/openclaw-local-rag/models/reranker",
        device="cpu"
    )
    
    # 获取模型信息
    model_info = reranker.get_model_info()
    print(f"模型状态: {model_info['status']}")
    
    if model_info['status'] == 'loaded':
        # 执行重排序
        results = reranker.rerank(
            query=test_query,
            documents=test_documents,
            top_k=5,
            score_threshold=0.3
        )
        
        print(f"\n查询: {test_query}")
        print(f"重排序结果 ({len(results)} 个):")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"[{i+1}] ID: {result.id}")
            print(f"    文本: {result.text[:60]}...")
            print(f"    原始分数: {result.original_score:.4f}")
            print(f"    重排序分数: {result.rerank_score:.4f}")
            print(f"    混合分数: {result.hybrid_score:.4f}")
            print(f"    检索类型: {result.retrieval_type}")
            print()
    
    print("=" * 80)
    print("测试完成！")