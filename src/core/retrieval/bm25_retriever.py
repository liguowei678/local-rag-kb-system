"""
BM25关键词检索器

实现基于BM25算法的关键词检索，作为语义检索的补充：
1. 精确匹配专有名词、术语、代码
2. 处理拼写错误和同义词
3. 快速检索大规模文档

企业RAG项目中的BM25应用场景：
1. 条款编号检索（"第8条"）
2. 具体术语检索（"警告处分"）
3. 人员/部门名称检索
4. 日期/版本号检索
"""

import jieba
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import re
from dataclasses import dataclass
import pickle
import os


@dataclass
class BM25Document:
    """BM25文档结构"""
    id: str
    text: str
    tokens: List[str]
    metadata: Dict[str, Any]


class BM25Retriever:
    """BM25关键词检索器"""
    
    def __init__(self, 
                 use_jieba: bool = True,
                 remove_stopwords: bool = True,
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            use_jieba: 是否使用jieba分词（中文）
            remove_stopwords: 是否移除停用词
            k1: BM25参数k1（控制词频饱和度）
            b: BM25参数b（控制文档长度归一化）
        """
        self.use_jieba = use_jieba
        self.remove_stopwords = remove_stopwords
        self.k1 = k1
        self.b = b
        
        # 优化的中文停用词（只过滤绝对安全的词语）
        self.stopwords = {
            # 结构助词（绝对安全）
            '的', '地', '得',
            
            # 时态助词（绝对安全）
            '了', '着', '过',
            
            # 语气助词（绝对安全）
            '啊', '呀', '呢', '吗', '吧', '啦', '哇', '哦', '哟',
            
            # 语气副词（相对安全）
            '就', '都', '也', '还', '又', '再', '却', '倒',
        }
        
        # 员工手册重要关键词（用于调试和验证）
        self.employee_handbook_keywords = {
            '出差', '餐费', '标准', '多少', '处罚', '处分', '警告', '记过',
            '罚款', '扣款', '员工', '手册', '制度', '规定', '条款', '政策',
            '请假', '加班', '调休', '年假', '病假', '事假', '旷工', '迟到',
            '早退', '考勤', '绩效', '考核', '薪酬', '工资', '福利', '奖励',
            '表彰', '部门', '经理', '领导', '公司', '企业', '组织'
        }
        
        # BM25模型和文档索引
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[BM25Document] = []
        self.doc_id_to_index: Dict[str, int] = {}
        
        # 初始化jieba
        if use_jieba:
            jieba.initialize()
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        构建BM25倒排索引
        
        Args:
            documents: 文档列表，每个文档包含id、text、metadata
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        # 清空现有索引
        self.documents = []
        self.doc_id_to_index = {}
        
        # 处理文档
        tokenized_docs = []
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', str(i))
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # 分词
            tokens = self._tokenize(text)
            
            # 创建文档对象
            bm25_doc = BM25Document(
                id=doc_id,
                text=text,
                tokens=tokens,
                metadata=metadata
            )
            
            self.documents.append(bm25_doc)
            self.doc_id_to_index[doc_id] = i
            tokenized_docs.append(tokens)
        
        # 初始化BM25模型
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=self.k1,
            b=self.b
        )
        
        print(f"BM25索引构建完成，文档数量: {len(self.documents)}")
    
    def add_document(self, document: Dict[str, Any]):
        """
        添加单个文档到索引（增量更新）
        
        Note: BM25Okapi不支持增量更新，需要重建索引
        对于频繁更新的场景，建议定期重建索引或使用支持增量的实现
        """
        # 这里实现简单的增量更新（实际生产环境需要更复杂的实现）
        # 暂时只添加到文档列表，下次查询时重建索引
        doc_id = document.get('id', str(len(self.documents)))
        text = document.get('text', '')
        metadata = document.get('metadata', {})
        
        tokens = self._tokenize(text)
        bm25_doc = BM25Document(
            id=doc_id,
            text=text,
            tokens=tokens,
            metadata=metadata
        )
        
        self.documents.append(bm25_doc)
        self.doc_id_to_index[doc_id] = len(self.documents) - 1
        
        # 标记需要重建索引
        self.bm25 = None
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        BM25关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            
        Returns:
            检索结果列表
        """
        if not self.bm25:
            # 需要重建索引
            self._rebuild_index()
        
        if not self.bm25 or not self.documents:
            return []
        
        # 查询分词
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            print(f"BM25查询分词为空: '{query}'")
            return []
        
        print(f"BM25查询分词: '{query}' -> {query_tokens}")
        
        # BM25评分
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception as e:
            print(f"BM25评分失败: {e}")
            return []
        
        # 获取Top-K结果
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            
            if score <= score_threshold:
                continue
            
            doc = self.documents[idx]
            
            # 计算查询词在文档中的出现情况
            query_in_doc = self._calculate_query_coverage(query_tokens, doc.tokens)
            
            results.append({
                'id': doc.id,
                'text': doc.text,
                'score': score,
                'metadata': doc.metadata,
                'retrieval_type': 'bm25',
                'query_coverage': query_in_doc,
                'tokens': doc.tokens[:20]  # 只返回前20个token用于调试
            })
        
        return results
    
    def search_with_filters(self,
                           query: str,
                           filters: Dict[str, Any],
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        带过滤条件的BM25检索
        
        Args:
            filters: 过滤条件，如 {'source': '员工手册', 'year': 2024}
        """
        # 先检索所有结果
        all_results = self.search(query, top_k=len(self.documents))
        
        # 应用过滤条件
        filtered_results = []
        for result in all_results:
            if self._match_filters(result['metadata'], filters):
                filtered_results.append(result)
        
        return filtered_results[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词（优化版，确保重要关键词不被过滤）"""
        if not text:
            return []
        
        if self.use_jieba:
            # 中文分词
            tokens = list(jieba.cut(text))
        else:
            # 英文分词（简单空格分割）
            tokens = text.split()
        
        # 移除停用词（优化逻辑）
        if self.remove_stopwords:
            filtered_tokens = []
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                
                # 检查是否是员工手册重要关键词
                is_important_keyword = token in self.employee_handbook_keywords
                
                # 检查是否是长词（通常更重要）
                is_long_word = len(token) > 2
                
                # 检查是否包含数字或字母（通常更重要）
                has_digit_or_letter = any(c.isdigit() or c.isalpha() for c in token)
                
                # 如果是重要关键词、长词或包含数字/字母，不过滤
                if is_important_keyword or is_long_word or has_digit_or_letter:
                    filtered_tokens.append(token)
                # 否则检查是否是停用词
                elif token not in self.stopwords:
                    filtered_tokens.append(token)
            
            tokens = filtered_tokens
        
        return tokens
    
    def _calculate_query_coverage(self, 
                                query_tokens: List[str], 
                                doc_tokens: List[str]) -> Dict[str, float]:
        """计算查询词在文档中的覆盖率"""
        if not query_tokens or not doc_tokens:
            return {'coverage': 0.0, 'matched_tokens': []}
        
        # 查找匹配的token
        matched_tokens = []
        for q_token in query_tokens:
            if q_token in doc_tokens:
                matched_tokens.append(q_token)
        
        # 计算覆盖率
        coverage = len(matched_tokens) / len(query_tokens) if query_tokens else 0
        
        return {
            'coverage': coverage,
            'matched_tokens': matched_tokens
        }
    
    def _match_filters(self, 
                      metadata: Dict[str, Any], 
                      filters: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # 列表匹配：metadata[key]需要在value列表中
                if metadata[key] not in value:
                    return False
            else:
                # 精确匹配
                if metadata[key] != value:
                    return False
        
        return True
    
    def _rebuild_index(self):
        """重建BM25索引"""
        if not self.documents:
            return
        
        tokenized_docs = [doc.tokens for doc in self.documents]
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=self.k1,
            b=self.b
        )
    
    def save_index(self, filepath: str):
        """保存索引到文件"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_id_to_index': self.doc_id_to_index,
                    'k1': self.k1,
                    'b': self.b
                }, f)
            print(f"BM25索引已保存到: {filepath}")
        except Exception as e:
            print(f"保存BM25索引失败: {e}")
    
    def load_index(self, filepath: str):
        """从文件加载索引"""
        if not os.path.exists(filepath):
            print(f"索引文件不存在: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.doc_id_to_index = data['doc_id_to_index']
            self.k1 = data.get('k1', 1.5)
            self.b = data.get('b', 0.75)
            
            # 重建BM25模型
            self._rebuild_index()
            
            print(f"BM25索引已加载，文档数量: {len(self.documents)}")
        except Exception as e:
            print(f"加载BM25索引失败: {e}")
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        if doc_id not in self.doc_id_to_index:
            return None
        
        idx = self.doc_id_to_index[doc_id]
        doc = self.documents[idx]
        
        return {
            'id': doc.id,
            'text': doc.text,
            'metadata': doc.metadata
        }


# 测试函数
if __name__ == "__main__":
    # 创建测试文档
    test_documents = [
        {
            'id': 'doc1',
            'text': '根据《员工手册》第8条第3款，旷工1天给予警告处分，扣款50元。',
            'metadata': {'source': '员工手册', 'section': '第8条'}
        },
        {
            'id': 'doc2', 
            'text': '2024年最新的考勤制度规定，迟到30分钟以内扣款20元。',
            'metadata': {'source': '考勤制度', 'year': 2024}
        },
        {
            'id': 'doc3',
            'text': '警告处分适用于情节轻微的违纪行为，如工作上消极怠工。',
            'metadata': {'source': '员工手册', 'section': '处分条款'}
        },
        {
            'id': 'doc4',
            'text': '记过处分适用于情节较为严重的违纪行为，扣款200元。',
            'metadata': {'source': '员工手册', 'section': '处分条款'}
        },
        {
            'id': 'doc5',
            'text': '请假需要提前向直属领导申请，并填写请假申请表。',
            'metadata': {'source': '请假制度'}
        }
    ]
    
    # 创建BM25检索器
    retriever = BM25Retriever()
    
    # 构建索引
    retriever.build_index(test_documents)
    
    # 测试查询
    test_queries = [
        "警告处分",
        "旷工1天怎么处理",
        "2024年考勤制度",
        "请假申请流程",
        "扣款50元"
    ]
    
    print("BM25检索测试结果:")
    print("=" * 80)
    
    for query in test_queries:
        results = retriever.search(query, top_k=3)
        
        print(f"\n查询: {query}")
        print(f"返回结果: {len(results)} 个")
        
        for i, result in enumerate(results):
            print(f"  [{i+1}] 分数: {result['score']:.4f}")
            print(f"      文本: {result['text'][:80]}...")
            print(f"      覆盖率: {result['query_coverage']['coverage']:.2%}")
            print(f"      匹配词: {result['query_coverage']['matched_tokens']}")
    
    print("\n" + "=" * 80)
    print("测试完成！")