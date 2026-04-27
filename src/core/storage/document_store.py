"""
文档元数据存储服务
使用 PostgreSQL 持久化存储文档信息
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# SQLAlchemy 基础类
Base = declarative_base()


class DocumentModel(Base):
    """文档数据模型"""
    __tablename__ = "documents"
    
    id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), nullable=False, default="queued")
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # 文档内容（从表结构看有这些字段）
    content = Column(Text, nullable=True)
    doc_metadata_json = Column('metadata', JSON, nullable=True)  # 表字段名metadata，但Python属性名不能是metadata
    embedding_vector = Column(JSON, nullable=True)
    
    # 处理结果
    chunks_count = Column(Integer, nullable=True)
    processed_chunks = Column(Integer, nullable=True)
    failed_chunks = Column(Integer, nullable=True)
    collection_name = Column(String(100), nullable=True)
    processing_time = Column(Integer, nullable=True)  # 毫秒


class DocumentStore:
    """文档存储管理器"""
    
    def __init__(self):
        # 创建数据库连接
        database_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # 创建表（如果不存在）
        self._create_tables()
        
        logger.info("文档存储服务已初始化", database_url=database_url)
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("数据库表创建/检查完成")
        except Exception as e:
            logger.error("创建数据库表失败", error=str(e))
            raise
    
    def save_document(self, document_data: Dict[str, Any]) -> bool:
        """保存文档信息"""
        session = self.Session()
        try:
            # 检查是否已存在
            existing = session.query(DocumentModel).filter_by(id=document_data["document_id"]).first()
            
            if existing:
                # 更新现有记录
                existing.filename = document_data.get("filename", "")
                existing.file_size = document_data.get("file_size", 0)
                existing.file_type = document_data.get("file_type", "")
                existing.file_path = document_data.get("file_path", "")
                existing.status = document_data.get("status", "queued")
                existing.doc_metadata = document_data.get("metadata", {})
                existing.updated_at = datetime.utcnow()
            else:
                # 创建新记录
                document = DocumentModel(
                    id=document_data["document_id"],
                    filename=document_data.get("filename", ""),
                    file_size=document_data.get("file_size", 0),
                    file_type=document_data.get("file_type", ""),
                    file_path=document_data.get("file_path", ""),
                    status=document_data.get("status", "queued"),
                    doc_metadata=document_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(document_data.get("created_at", datetime.utcnow().isoformat()))
                )
                session.add(document)
            
            session.commit()
            logger.debug("文档信息保存成功", document_id=document_data["document_id"])
            return True
        except Exception as e:
            session.rollback()
            logger.error("保存文档信息失败", 
                        document_id=document_data.get("document_id"),
                        error=str(e))
            return False
        finally:
            session.close()
    
    def update_document_status(self, document_id: str, status: str, 
                              error_message: str = None, 
                              processing_result: Dict[str, Any] = None) -> bool:
        """更新文档状态"""
        session = self.Session()
        try:
            document = session.query(DocumentModel).filter_by(id=document_id).first()
            if not document:
                logger.warning("文档不存在", document_id=document_id)
                return False
            
            document.status = status
            document.updated_at = datetime.utcnow()
            
            if status == "processing":
                document.processed_at = datetime.utcnow()
            elif status == "completed" and processing_result:
                document.processed_at = datetime.utcnow()
                document.chunks_count = processing_result.get("chunks_count")
                document.processed_chunks = processing_result.get("processed_chunks")
                document.failed_chunks = processing_result.get("failed_chunks")
                document.collection_name = processing_result.get("collection_name")
                document.processing_time = int(processing_result.get("processing_time", 0) * 1000)
            elif status == "failed" and error_message:
                document.error_message = error_message
            
            session.commit()
            logger.debug("文档状态更新成功", 
                        document_id=document_id,
                        status=status)
            return True
        except Exception as e:
            session.rollback()
            logger.error("更新文档状态失败", 
                        document_id=document_id,
                        error=str(e))
            return False
        finally:
            session.close()
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档信息"""
        session = self.Session()
        try:
            document = session.query(DocumentModel).filter_by(id=document_id).first()
            if not document:
                return None
            
            return {
                "document_id": document.id,
                "filename": document.filename,
                "file_size": document.file_size,
                "file_type": document.file_type,
                "file_path": document.file_path,
                "status": document.status,
                "metadata": document.doc_metadata or {},
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "processed_at": document.processed_at.isoformat() if document.processed_at else None,
                "error_message": document.error_message,
                "chunks_count": document.chunks_count,
                "processed_chunks": document.processed_chunks,
                "failed_chunks": document.failed_chunks,
                "collection_name": document.collection_name,
                "processing_time": document.processing_time
            }
        except Exception as e:
            logger.error("获取文档信息失败", 
                        document_id=document_id,
                        error=str(e))
            return None
        finally:
            session.close()
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出文档"""
        session = self.Session()
        try:
            documents = session.query(DocumentModel)\
                .order_by(DocumentModel.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            result = []
            for doc in documents:
                result.append({
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "file_size": doc.file_size,
                    "file_type": doc.file_type,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat()
                })
            
            return result
        except Exception as e:
            logger.error("列出文档失败", error=str(e))
            return []
        finally:
            session.close()
    
    def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        session = self.Session()
        try:
            document = session.query(DocumentModel).filter_by(id=document_id).first()
            if not document:
                logger.warning("文档不存在", document_id=document_id)
                return False
            
            session.delete(document)
            session.commit()
            logger.info("文档删除成功", document_id=document_id)
            return True
        except Exception as e:
            session.rollback()
            logger.error("删除文档失败", 
                        document_id=document_id,
                        error=str(e))
            return False
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        session = self.Session()
        try:
            total = session.query(DocumentModel).count()
            queued = session.query(DocumentModel).filter_by(status="queued").count()
            processing = session.query(DocumentModel).filter_by(status="processing").count()
            completed = session.query(DocumentModel).filter_by(status="completed").count()
            failed = session.query(DocumentModel).filter_by(status="failed").count()
            
            return {
                "total": total,
                "queued": queued,
                "processing": processing,
                "completed": completed,
                "failed": failed
            }
        except Exception as e:
            logger.error("获取统计信息失败", error=str(e))
            return {}
        finally:
            session.close()


# 全局实例
document_store = DocumentStore()