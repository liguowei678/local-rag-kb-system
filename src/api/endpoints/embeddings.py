"""
文本嵌入端点
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# 数据模型
class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    text: str
    model: Optional[str] = "deepseek-embedding"

class EmbeddingResponse(BaseModel):
    """嵌入响应"""
    vector: List[float]
    dimension: int
    model: str
    text_length: int

class BatchEmbeddingRequest(BaseModel):
    """批量嵌入请求"""
    texts: List[str]
    model: Optional[str] = "deepseek-embedding"

class BatchEmbeddingResponse(BaseModel):
    """批量嵌入响应"""
    embeddings: List[List[float]]
    dimension: int
    model: str
    total_texts: int

@router.post("/", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """创建文本嵌入"""
    
    # 验证文本长度
    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="文本过长，最大支持10000字符"
        )
    
    if len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="文本不能为空"
        )
    
    # 这里应该是实际的嵌入模型调用
    # 例如: vector = await embedding_model.encode(request.text)
    
    # 临时模拟嵌入向量（512维）
    import random
    random.seed(hash(request.text) % 10000)  # 相同文本生成相同向量
    
    dimension = 512
    mock_vector = [random.uniform(-1, 1) for _ in range(dimension)]
    
    # 归一化（模拟）
    import math
    norm = math.sqrt(sum(x * x for x in mock_vector))
    if norm > 0:
        mock_vector = [x / norm for x in mock_vector]
    
    return EmbeddingResponse(
        vector=mock_vector,
        dimension=dimension,
        model=request.model,
        text_length=len(request.text)
    )

@router.post("/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    """批量创建文本嵌入"""
    
    # 验证批量大小
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="批量大小超过限制，最大支持100个文本"
        )
    
    # 验证每个文本
    for text in request.texts:
        if len(text) > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"文本过长: {len(text)} > 10000字符"
            )
        if len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="文本不能为空"
            )
    
    # 批量生成嵌入向量
    embeddings = []
    dimension = 512
    
    for text in request.texts:
        # 模拟嵌入生成
        import random
        random.seed(hash(text) % 10000)
        
        mock_vector = [random.uniform(-1, 1) for _ in range(dimension)]
        
        # 归一化
        import math
        norm = math.sqrt(sum(x * x for x in mock_vector))
        if norm > 0:
            mock_vector = [x / norm for x in mock_vector]
        
        embeddings.append(mock_vector)
    
    return BatchEmbeddingResponse(
        embeddings=embeddings,
        dimension=dimension,
        model=request.model,
        total_texts=len(request.texts)
    )

@router.get("/models")
async def list_embedding_models():
    """获取支持的嵌入模型列表"""
    return {
        "models": [
            {
                "id": "deepseek-embedding",
                "name": "DeepSeek Embedding",
                "dimension": 512,
                "max_length": 8192,
                "language": ["zh", "en"],
                "description": "DeepSeek文本嵌入模型"
            },
            {
                "id": "bge-small-zh",
                "name": "BGE Small Chinese",
                "dimension": 512,
                "max_length": 512,
                "language": ["zh"],
                "description": "百度BGE中文小模型"
            },
            {
                "id": "text-embedding-ada-002",
                "name": "OpenAI Ada v2",
                "dimension": 1536,
                "max_length": 8191,
                "language": ["multi"],
                "description": "OpenAI文本嵌入模型"
            }
        ],
        "default_model": "deepseek-embedding"
    }

@router.get("/info/{model_id}")
async def get_model_info(model_id: str):
    """获取模型信息"""
    models = {
        "deepseek-embedding": {
            "id": "deepseek-embedding",
            "name": "DeepSeek Embedding",
            "dimension": 512,
            "max_length": 8192,
            "language": ["zh", "en"],
            "description": "DeepSeek文本嵌入模型",
            "provider": "DeepSeek",
            "api_endpoint": "https://api.deepseek.com/embeddings"
        },
        "bge-small-zh": {
            "id": "bge-small-zh",
            "name": "BGE Small Chinese",
            "dimension": 512,
            "max_length": 512,
            "language": ["zh"],
            "description": "百度BGE中文小模型",
            "provider": "BAAI",
            "model_card": "https://huggingface.co/BAAI/bge-small-zh"
        }
    }
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    return models[model_id]

@router.post("/similarity")
async def calculate_similarity(
    text1: str,
    text2: str,
    model: Optional[str] = "deepseek-embedding"
):
    """计算文本相似度"""
    
    # 生成两个文本的嵌入
    import random
    import math
    
    # 模拟嵌入生成
    random.seed(hash(text1) % 10000)
    vec1 = [random.uniform(-1, 1) for _ in range(512)]
    norm1 = math.sqrt(sum(x * x for x in vec1))
    if norm1 > 0:
        vec1 = [x / norm1 for x in vec1]
    
    random.seed(hash(text2) % 10000)
    vec2 = [random.uniform(-1, 1) for _ in range(512)]
    norm2 = math.sqrt(sum(x * x for x in vec2))
    if norm2 > 0:
        vec2 = [x / norm2 for x in vec2]
    
    # 计算余弦相似度
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    similarity = max(0, min(1, dot_product))  # 限制在0-1之间
    
    return {
        "text1": text1[:100] + ("..." if len(text1) > 100 else ""),
        "text2": text2[:100] + ("..." if len(text2) > 100 else ""),
        "similarity": similarity,
        "model": model,
        "interpretation": self._interpret_similarity(similarity)
    }
    
    def _interpret_similarity(self, score: float) -> str:
        """解释相似度分数"""
        if score >= 0.9:
            return "几乎相同"
        elif score >= 0.7:
            return "高度相似"
        elif score >= 0.5:
            return "中等相似"
        elif score >= 0.3:
            return "低度相似"
        else:
            return "不相关"