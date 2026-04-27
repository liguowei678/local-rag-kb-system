"""
文档管理端点
"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.core.task_processor import process_document_async
from src.core.vector_db import search_similar
from src.core.storage.document_store import document_store

router = APIRouter()

# 数据模型
class DocumentMetadata(BaseModel):
    """文档元数据"""
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    description: Optional[str] = None

class DocumentResponse(BaseModel):
    """文档响应"""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    status: str
    message: str
    metadata: Optional[dict] = None
    created_at: str

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    limit: Optional[int] = 5
    threshold: Optional[float] = 0.3

class SearchResult(BaseModel):
    """搜索结果"""
    text: str
    score: float
    source: str
    page: Optional[int] = None
    metadata: Optional[dict] = None

class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float

# 临时存储（实际应该用数据库）
documents_store = []

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """上传文档"""
    
    # 验证文件大小
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {file_size} > {settings.MAX_UPLOAD_SIZE}"
        )
    
    # 验证文件类型
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}. 支持的类型: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # 解析元数据
    doc_metadata = {}
    if metadata:
        try:
            import json
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            doc_metadata = {"raw_metadata": metadata}
    
    # 生成文档ID
    document_id = f"doc_{uuid.uuid4().hex[:10]}"
    
    # 保存文件（实际应该保存到存储系统）
    upload_dir = os.path.join(settings.UPLOAD_DIR, document_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # 创建文档记录
    document = {
        "document_id": document_id,
        "filename": file.filename,
        "file_size": file_size,
        "file_type": file_ext,
        "file_path": file_path,
        "metadata": doc_metadata,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "processed_at": None
    }
    
    # 存储文档记录到数据库
    document_store.save_document(document)
    
    # 触发异步处理：文本提取、分块、向量化
    asyncio.create_task(process_document_async(
        document_id=document_id,
        file_path=file_path,
        metadata={
            "collection_name": "default",
            **doc_metadata,
            "original_filename": file.filename,
            "uploaded_at": document["created_at"]
        }
    ))
    
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename,
        file_size=file_size,
        file_type=file_ext,
        status="queued",
        message="文档已加入处理队列",
        metadata=doc_metadata,
        created_at=document["created_at"]
    )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    page: int = 1,
    per_page: int = 20,
    status: Optional[str] = None
):
    """获取文档列表"""
    # 从数据库获取文档列表
    offset = (page - 1) * per_page
    db_docs = document_store.list_documents(limit=per_page, offset=offset)
    
    # 如果指定了状态，需要过滤
    if status:
        db_docs = [doc for doc in db_docs if doc.get("status") == status]
    
    return [
        DocumentResponse(
            document_id=doc["document_id"],
            filename=doc["filename"],
            file_size=doc["file_size"],
            file_type=doc["file_type"],
            status=doc["status"],
            message="处理完成" if doc.get("processed_at") else "等待处理",
            metadata=doc.get("metadata", {}),
            created_at=doc["created_at"]
        )
        for doc in db_docs
    ]

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """获取文档详情"""
    for doc in documents_store:
        if doc["document_id"] == document_id:
            return DocumentResponse(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_size=doc["file_size"],
                file_type=doc["file_type"],
                status=doc["status"],
                message="处理完成" if doc.get("processed_at") else "等待处理",
                metadata=doc.get("metadata"),
                created_at=doc["created_at"]
            )
    
    raise HTTPException(status_code=404, detail="文档不存在")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    global documents_store
    
    for i, doc in enumerate(documents_store):
        if doc["document_id"] == document_id:
            # 删除文件（实际实现）
            if os.path.exists(doc.get("file_path", "")):
                try:
                    os.remove(doc["file_path"])
                    upload_dir = os.path.dirname(doc["file_path"])
                    if os.path.exists(upload_dir):
                        os.rmdir(upload_dir)
                except:
                    pass
            
            # 从存储中删除
            documents_store.pop(i)
            
            return {
                "success": True,
                "message": "文档已删除",
                "document_id": document_id
            }
    
    raise HTTPException(status_code=404, detail="文档不存在")

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """搜索文档"""
    import time
    start_time = time.time()
    
    try:
        # 使用真实的向量搜索
        search_results = await search_similar(
            collection_name="default",
            query_text=request.query,
            limit=request.limit,
            score_threshold=request.threshold
        )
        
        # 格式化结果
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "source": result.get("metadata", {}).get("original_filename", "未知文档"),
                "metadata": result.get("metadata", {})
            })
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    text=result["text"],
                    score=result["score"],
                    source=result["source"],
                    metadata=result.get("metadata")
                )
                for result in formatted_results
            ],
            total_results=len(formatted_results),
            search_time=search_time
        )
        
    except Exception as e:
        logger.error("搜索失败", query=request.query, error=str(e))
        
        # 失败时返回空结果
        return SearchResponse(
            query=request.query,
            results=[],
            total_results=0,
            search_time=time.time() - start_time
        )