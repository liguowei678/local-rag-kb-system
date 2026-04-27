"""
异步任务处理器

处理文档上传后的异步处理流程：
1. 文档解析和分块
2. 文本嵌入
3. 向量存储
4. 状态更新
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from src.core.logging import get_logger
from src.core.document_processor.processor import DocumentProcessor
from src.core.document_processor.models import DocumentFormat, ChunkingStrategy, DocumentChunk
from src.core.vector_db.real_qdrant_manager import real_qdrant_manager
from src.core.embedding import get_embedding_service
from src.core.storage.document_store import document_store
from src.core.config import settings

logger = get_logger(__name__)


class TaskProcessor:
    """异步任务处理器"""
    
    def __init__(self):
        # 使用全局document_processor实例，确保分块策略一致
        from src.core.document_processor.processor import document_processor
        self.document_processor = document_processor
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """启动任务处理器"""
        if self.is_running:
            logger.warning("任务处理器已经在运行")
            return
        
        self.is_running = True
        logger.info("启动异步任务处理器")
        
        # 启动后台处理任务
        asyncio.create_task(self._process_queue())
        
    async def stop(self):
        """停止任务处理器"""
        self.is_running = False
        logger.info("停止异步任务处理器")
        
    async def add_document_task(self, document_id: str, file_path: str, metadata: Dict[str, Any]):
        """添加文档处理任务到队列"""
        task = {
            "document_id": document_id,
            "file_path": file_path,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        await self.processing_queue.put(task)
        logger.info("文档处理任务已添加到队列", document_id=document_id)
        
        return task
    
    async def _process_queue(self):
        """处理队列中的任务"""
        while self.is_running:
            try:
                # 从队列获取任务
                task = await self.processing_queue.get()
                
                if task is None:
                    continue
                
                document_id = task["document_id"]
                file_path = task["file_path"]
                metadata = task["metadata"]
                
                logger.info("开始处理文档", document_id=document_id)
                
                # 更新任务状态
                task["status"] = "processing"
                task["started_at"] = datetime.now().isoformat()
                
                # 更新数据库中的文档状态
                document_store.update_document_status(document_id, "processing")
                
                try:
                    # 执行文档处理流程
                    result = await self._process_document(document_id, file_path, metadata)
                    
                    # 更新任务状态为完成
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now().isoformat()
                    task["result"] = result
                    
                    # 更新数据库中的文档状态
                    document_store.update_document_status(
                        document_id, 
                        "completed",
                        processing_result=result
                    )
                    
                    logger.info("文档处理完成", 
                               document_id=document_id,
                               chunks_count=result.get("chunks_count", 0),
                               processing_time=result.get("processing_time", 0))
                    
                    # 如果启用了GraphRAG，在向量化完成后触发构建
                    if metadata.get("graphrag_enabled", False):
                        try:
                            from src.api.endpoints.documents_graphrag import trigger_incremental_graphrag_build
                            from src.core.graph.incremental_builder import IncrementalConfig
                            
                            logger.info("触发GraphRAG构建", document_id=document_id)
                            
                            incremental_config = IncrementalConfig(
                                merge_strategy="merge",
                                enable_content_hash=True,
                                enable_timestamp_check=True
                            )
                            
                            # 读取文档内容（避免重新提取）
                            content = await self._read_document_content(file_path)
                            
                            # 增强metadata，包含内容
                            enhanced_metadata = {
                                **metadata,
                                "extracted_content": content[:5000],  # 只传前5000字符
                                "content_length": len(content)
                            }
                            
                            graphrag_task_id = await trigger_incremental_graphrag_build(
                                document_id=document_id,
                                file_path=file_path,
                                file_type=metadata.get("file_type", "pdf"),
                                metadata=enhanced_metadata,
                                incremental_config=incremental_config
                            )
                            
                            logger.info("GraphRAG构建已触发", 
                                       document_id=document_id,
                                       task_id=graphrag_task_id,
                                       content_length=len(content))
                            
                        except Exception as graphrag_error:
                            logger.warning("GraphRAG构建触发失败", 
                                         document_id=document_id,
                                         error=str(graphrag_error))
                    
                except Exception as e:
                    # 更新任务状态为失败
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["failed_at"] = datetime.now().isoformat()
                    
                    # 更新数据库中的文档状态
                    document_store.update_document_status(
                        document_id, 
                        "failed",
                        error_message=str(e)
                    )
                    
                    logger.error("文档处理失败", 
                                document_id=document_id,
                                error=str(e),
                                exc_info=True)
                    
                finally:
                    # 标记任务完成
                    self.processing_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info("任务处理器被取消")
                break
            except Exception as e:
                logger.error("任务处理器异常", error=str(e), exc_info=True)
                await asyncio.sleep(1)  # 避免快速循环
    
    async def _process_document(self, document_id: str, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个文档的完整流程"""
        start_time = time.time()
        
        try:
            # 1. 读取文档内容
            logger.info("读取文档内容", document_id=document_id, file_path=file_path)
            content = await self._read_document_content(file_path)
            
            if not content:
                raise ValueError("文档内容为空")
            
            # 2. 预处理和分块
            logger.info("文档预处理和分块", document_id=document_id)
            # 使用文档处理器的默认分块策略（现在是HIERARCHICAL）
            chunks = self.document_processor.chunk_text(
                text=content,
                strategy=None,  # 使用DocumentProcessor的默认策略
                metadata=metadata
            )
            
            if not chunks:
                raise ValueError("文档分块失败")
            
            # 3. 创建或获取集合
            collection_name = metadata.get("collection_name", "default")
            logger.info("准备向量存储", document_id=document_id, collection_name=collection_name)
            
            # 检查集合是否存在，不存在则创建
            if not await real_qdrant_manager.collection_exists(collection_name):
                await real_qdrant_manager.create_collection(
                    collection_name=collection_name,
                    vector_size=settings.EMBEDDING_DIMENSION
                )
            
            # 4. 处理每个分块：嵌入 + 存储
            logger.info("处理文档分块", document_id=document_id, chunks_count=len(chunks))
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # 文本嵌入
                    embedding_service = get_embedding_service()
                    embedding_result = await embedding_service.embed_text(chunk.text)
                    if not embedding_result:
                        logger.warning("分块嵌入失败", 
                                      document_id=document_id,
                                      chunk_index=i)
                        continue
                    
                    # 准备向量存储的payload
                    payload = {
                        "text": chunk.text,
                        "document_id": document_id,  # 关键：用于Qdrant文档级删除
                        "chunk_index": i,
                        "metadata": {
                            **metadata,
                            "chunk_size": len(chunk.text),
                            "chunk_strategy": self.document_processor.chunking_strategy.value
                        }
                    }
                    
                    # 存储到向量数据库
                    # Qdrant 要求点 ID 是整数或 UUID，这里使用哈希值
                    import hashlib
                    point_hash = hashlib.md5(f"{document_id}_{i}".encode()).hexdigest()
                    # 将哈希转换为整数（取前8位）
                    point_id = int(point_hash[:8], 16)
                    await real_qdrant_manager.upsert_points(
                        collection_name=collection_name,
                        points=[{
                            "id": point_id,
                            "vector": embedding_result.get("vector", []) if isinstance(embedding_result, dict) else (embedding_result.vector if hasattr(embedding_result, 'vector') else []),
                            "payload": payload
                        }]
                    )
                    
                    processed_chunks.append({
                        "chunk_id": point_id,
                        "text_length": len(chunk.text),
                        "embedding_success": True
                    })
                    
                    logger.debug("分块处理完成", 
                                document_id=document_id,
                                chunk_index=i,
                                text_length=len(chunk.text))
                    
                except Exception as e:
                    logger.error("分块处理失败", 
                                document_id=document_id,
                                chunk_index=i,
                                error=str(e))
                    processed_chunks.append({
                        "chunk_id": f"{document_id}_{i}",
                        "text_length": len(chunk.text),
                        "embedding_success": False,
                        "error": str(e)
                    })
            
            # 5. 返回处理结果
            processing_time = time.time() - start_time
            
            return {
                "document_id": document_id,
                "chunks_count": len(chunks),
                "processed_chunks": len([c for c in processed_chunks if c.get("embedding_success")]),
                "failed_chunks": len([c for c in processed_chunks if not c.get("embedding_success")]),
                "collection_name": collection_name,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("文档处理流程失败", 
                        document_id=document_id,
                        error=str(e),
                        exc_info=True)
            raise
    
    async def _read_document_content(self, file_path: str) -> str:
        """读取文档内容（智能处理PDF和文本文件）"""
        from pathlib import Path
        
        path = Path(file_path)
        
        # 如果是PDF文件，使用Docling解析
        if path.suffix.lower() == '.pdf':
            try:
                logger.info("使用Docling解析PDF文档", file_path=file_path)
                
                # 调用document_processor的PDF解析方法
                content = await self.document_processor._read_pdf_file(path)
                
                if content:
                    logger.info("Docling解析成功", 
                               file_path=file_path,
                               content_length=len(content))
                    return content
                else:
                    logger.error("Docling解析返回空内容", file_path=file_path)
                    return ""
                    
            except Exception as e:
                logger.error("Docling解析PDF失败", 
                           file_path=file_path,
                           error=str(e))
                # 尝试回退到文本读取（可能得到乱码，但至少不会崩溃）
                pass
        
        # 对于非PDF文件或PDF解析失败，使用原来的文本读取逻辑
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        if content.strip():  # 检查是否有实际内容
                            logger.debug("文档读取成功", 
                                        file_path=file_path, 
                                        encoding=encoding,
                                        content_length=len(content))
                            return content
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning("编码尝试失败", 
                                 file_path=file_path, 
                                 encoding=encoding,
                                 error=str(e))
                    continue
            
            # 所有编码都失败，尝试二进制读取
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    # 尝试解码为utf-8，忽略错误
                    return content.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error("二进制读取失败", file_path=file_path, error=str(e))
                return ""
                
        except Exception as e:
            logger.error("读取文档失败", file_path=file_path, error=str(e))
            return ""


# 全局任务处理器实例
task_processor = TaskProcessor()


async def init_task_processor():
    """初始化任务处理器"""
    await task_processor.start()
    return task_processor


async def shutdown_task_processor():
    """关闭任务处理器"""
    await task_processor.stop()


async def process_document_async(document_id: str, file_path: str, metadata: Dict[str, Any]):
    """异步处理文档（供API端点调用）"""
    return await task_processor.add_document_task(document_id, file_path, metadata)