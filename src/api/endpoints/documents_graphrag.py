"""
文档上传 + GraphRAG增量构建端点
"""

import os
import hashlib
import uuid
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.core.config import settings
from src.core.task_processor import process_document_async
from src.core.storage.document_store import document_store
from src.core.graph.factory import GraphRAGFactory
from src.core.graph.incremental_builder import IncrementalKnowledgeGraphBuilder, IncrementalConfig

router = APIRouter(tags=["documents-graphrag"])  # 去掉前缀，由主路由统一添加


# 数据模型
class GraphRAGDocumentMetadata(BaseModel):
    """GraphRAG文档元数据"""
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = "api_upload"
    enable_incremental: bool = True  # 是否启用增量构建
    merge_strategy: str = "update"  # 合并策略


class GraphRAGUploadResponse(BaseModel):
    """GraphRAG上传响应"""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    status: str
    message: str
    metadata: Optional[dict] = None
    created_at: str
    graphrag_task_id: Optional[str] = None
    incremental_enabled: bool


class GraphRAGBuildStatus(BaseModel):
    """GraphRAG构建状态"""
    task_id: str
    status: str  # pending, building, completed, failed
    document_id: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# 内存中存储GraphRAG任务状态
graphrag_tasks = {}


async def extract_text_from_file(file_path: str, file_type: str) -> str:
    """从文件中提取文本内容"""
    try:
        if file_type == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_type == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_type == '.pdf':
            # 尝试使用docling解析PDF
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_markdown()
                
                if not text or len(text.strip()) < 10:
                    # 如果markdown导出失败，尝试从text_blocks提取
                    text = "\n".join([
                        str(item) for item in result.document.text_blocks
                        if hasattr(item, 'text') and item.text
                    ])
                
                if text and len(text.strip()) >= 10:
                    return text
                else:
                    return f"[PDF文件: {os.path.basename(file_path)}，docling提取内容过少]"
                    
            except Exception as e:
                # 回退到简单文本提取
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text
                except:
                    return f"[PDF文件: {os.path.basename(file_path)}，提取失败]"

        
        elif file_type == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 尝试提取文本内容
                if isinstance(data, dict):
                    return data.get('content', '') or json.dumps(data, ensure_ascii=False)
                elif isinstance(data, list):
                    return json.dumps(data, ensure_ascii=False)
                else:
                    return str(data)
        
        else:
            # 对于其他文件类型，返回占位符
            return f"[文件内容: {os.path.basename(file_path)}，类型: {file_type}]"
            
    except Exception as e:
        return f"[文件提取失败: {str(e)}]"


async def trigger_incremental_graphrag_build(
    document_id: str,
    file_path: str,
    file_type: str,
    metadata: Optional[dict] = None,
    incremental_config: Optional[IncrementalConfig] = None
) -> str:
    """触发GraphRAG增量构建"""
    try:
        # 提取文本内容
        content = metadata.get("extracted_content") if metadata else None
        if not content:
            content = await extract_text_from_file(file_path, file_type)
        
        if not content or len(content.strip()) < 10:
            raise ValueError("提取的文本内容过少")
        
        # 准备文档数据
        document = {
            "document_id": document_id,  # incremental_builder期望document_id
            "content": content,
            "metadata": {
                "filename": os.path.basename(file_path),
                "file_type": file_type,
                "file_size": os.path.getsize(file_path),
                "extracted_at": datetime.now().isoformat()
            }
        }
        
        # 如果有传递的哈希值，使用它（确保一致性）
        if metadata and "content_hash" in metadata:
            document["content_hash"] = metadata["content_hash"]
        
        # 检查GraphRAG配置
        config = GraphRAGFactory.create_config()
        if not config.graphrag_enabled:
            raise ValueError("GraphRAG功能未启用")
        
        if not config.deepseek_api_key:
            raise ValueError("DeepSeek API密钥未配置")
        
        # 创建增量构建器
        builder = IncrementalKnowledgeGraphBuilder(config, incremental_config)
        
        # 创建任务ID
        task_id = f"graphrag_{uuid.uuid4().hex[:8]}"
        graphrag_tasks[task_id] = {
            "task_id": task_id,
            "document_id": document_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 异步执行构建任务
        async def build_task():
            try:
                graphrag_tasks[task_id]["status"] = "building"
                graphrag_tasks[task_id]["progress"] = 0.3
                graphrag_tasks[task_id]["updated_at"] = datetime.now().isoformat()
                
                # 执行增量构建
                result = await builder.incremental_build([document])
                
                graphrag_tasks[task_id]["status"] = "completed"
                graphrag_tasks[task_id]["progress"] = 1.0
                graphrag_tasks[task_id]["result"] = result
                graphrag_tasks[task_id]["updated_at"] = datetime.now().isoformat()
                
                logger.info(f"GraphRAG增量构建完成: {task_id}, 文档: {document_id}")
                
            except Exception as e:
                logger.error(f"GraphRAG增量构建失败: {e}")
                graphrag_tasks[task_id]["status"] = "failed"
                graphrag_tasks[task_id]["error"] = str(e)
                graphrag_tasks[task_id]["updated_at"] = datetime.now().isoformat()
                
            finally:
                # 清理资源
                try:
                    if hasattr(builder, 'close') and asyncio.iscoroutinefunction(builder.close):
                        await builder.close()
                    elif hasattr(builder, 'close'):
                        builder.close()
                except Exception as close_error:
                    logger.warning(f"清理构建器资源失败: {close_error}")
        
        # 启动异步任务
        asyncio.create_task(build_task())
        
        return task_id
        
    except Exception as e:
        logger.error(f"触发GraphRAG构建失败: {e}")
        raise


@router.post("/upload", response_model=GraphRAGUploadResponse)
async def upload_document_with_graphrag(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    metadata: Optional[str] = Form(None),
    enable_incremental: bool = Form(True),
    merge_strategy: str = Form("update"),
    background_tasks: BackgroundTasks = None
):
    """上传文档并触发GraphRAG增量构建"""
    try:
        # 读取文件内容
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {file_size} > {settings.MAX_UPLOAD_SIZE}"
            )
        
        # 检查文件类型
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
                doc_metadata = json.loads(metadata)
            except:
                doc_metadata = {"raw_metadata": metadata}
        
        # 生成稳定的document_id（基于文件名，用户可控）
        # 用户规则：相同文件名=替换，不同文件名=新增
        filename_without_ext = os.path.splitext(file.filename)[0]
        filename_hash = hashlib.md5(filename_without_ext.encode()).hexdigest()
        document_id = f"doc_{filename_hash[:10]}"
        
        # 仍然计算内容哈希，用于记录（可选）
        content_hash = hashlib.md5(content).hexdigest()
        
        # 保存文件
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
            "created_at": datetime.now().isoformat(),
        }
        
        # 触发异步处理：文本提取、分块、向量化（Qdrant索引）
        asyncio.create_task(process_document_async(
            document_id=document_id,
            file_path=file_path,
            metadata={
                "collection_name": collection_name,
                "original_filename": file.filename,
                **doc_metadata
            }
        ))
        
        graphrag_task_id = None
        if enable_incremental:
            document["status"] = "graphrag_pending"
            document["graphrag_enabled"] = True
            
            # 触发GraphRAG增量构建
            incremental_config = IncrementalConfig(
                merge_strategy=merge_strategy,
                enable_content_hash=True,
                enable_timestamp_check=True
            )
            
            graphrag_task_id = await trigger_incremental_graphrag_build(
                document_id=document_id,
                file_path=file_path,
                file_type=file_ext,
                metadata={
                    "original_filename": file.filename,
                    "content_hash": content_hash,  # 传递完整内容哈希
                    **doc_metadata
                },
                incremental_config=incremental_config
            )
            
            document["graphrag_task_id"] = graphrag_task_id
        else:
            document["status"] = "uploaded"
            document["graphrag_enabled"] = False
        
        # 保存到文档存储
        document_store.save_document(document)
        
        return GraphRAGUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            file_type=file_ext,
            status=document["status"],
            message="文档上传成功" + ("，GraphRAG增量构建已触发" if enable_incremental else ""),
            metadata=doc_metadata,
            created_at=document["created_at"],
            graphrag_task_id=graphrag_task_id,
            incremental_enabled=enable_incremental
        )
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.get("/status/{task_id}", response_model=GraphRAGBuildStatus)
async def get_graphrag_build_status(task_id: str):
    """获取GraphRAG构建状态"""
    if task_id not in graphrag_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = graphrag_tasks[task_id]
    return GraphRAGBuildStatus(
        task_id=task_id,
        status=task_info["status"],
        document_id=task_info["document_id"],
        progress=task_info.get("progress"),
        result=task_info.get("result"),
        error=task_info.get("error"),
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"]
    )


@router.get("/tasks", response_model=List[GraphRAGBuildStatus])
async def list_graphrag_tasks(limit: int = 10, status: Optional[str] = None):
    """列出GraphRAG任务"""
    tasks = list(graphrag_tasks.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return tasks[:limit]


@router.post("/batch-upload", response_model=Dict[str, Any])
async def batch_upload_documents_with_graphrag(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    metadata: Optional[str] = Form(None),
    enable_incremental: bool = Form(True),
    merge_strategy: str = Form("update")
):
    """批量上传文档"""
    responses = []
    for file in files:
        try:
            response = await upload_document_with_graphrag(
                file=file,
                collection_name=collection_name,
                metadata=metadata,
                enable_incremental=enable_incremental,
                merge_strategy=merge_strategy
            )
            responses.append(response.dict())
        except Exception as e:
            responses.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful": len([r for r in responses if r.get("success", True)]),
        "failed": len([r for r in responses if not r.get("success", True)]),
        "responses": responses
    }


@router.post("/trigger-graphrag/{document_id}")
async def trigger_graphrag_for_existing_document(
    document_id: str,
    enable_incremental: bool = True,
    merge_strategy: str = "update"
):
    """为现有文档触发GraphRAG构建"""
    try:
        # 获取文档信息
        document = document_store.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        file_path = document.get("file_path")
        file_type = document.get("file_type")
        metadata = document.get("metadata", {})
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="文档文件不存在")
        
        # 触发GraphRAG构建
        incremental_config = IncrementalConfig(
            merge_strategy=merge_strategy,
            enable_content_hash=True,
            enable_timestamp_check=True
        )
        
        task_id = await trigger_incremental_graphrag_build(
            document_id=document_id,
            file_path=file_path,
            file_type=file_type,
            metadata=metadata,
            incremental_config=incremental_config
        )
        
        # 更新文档状态
        document["graphrag_task_id"] = task_id
        document["graphrag_enabled"] = enable_incremental
        document_store.save_document(document)
        
        return {
            "success": True,
            "task_id": task_id,
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"为现有文档触发GraphRAG构建失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发构建失败: {str(e)}")


# 日志配置
import logging
logger = logging.getLogger(__name__)