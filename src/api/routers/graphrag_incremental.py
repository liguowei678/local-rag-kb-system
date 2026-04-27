"""
增量GraphRAG API端点
支持增量构建、文档去重、智能合并
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.graph.factory import GraphRAGFactory
from src.core.graph.incremental_builder import IncrementalKnowledgeGraphBuilder, IncrementalConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graphrag-incremental"])  # 去掉前缀，由主路由统一添加


# 请求/响应模型
class IncrementalConfigRequest(BaseModel):
    """增量配置请求"""
    enable_content_hash: bool = Field(True, description="启用内容哈希去重")
    enable_timestamp_check: bool = Field(True, description="启用时间戳检查")
    merge_strategy: str = Field("update", description="合并策略: update/replace/merge")
    max_retention_days: int = Field(30, description="最大保留天数")
    batch_size: int = Field(10, description="批次大小")


class IncrementalBuildRequest(BaseModel):
    """增量构建请求"""
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    collection_name: str = Field("default", description="集合名称")
    incremental_config: Optional[IncrementalConfigRequest] = Field(None, description="增量配置")
    auto_cleanup: bool = Field(False, description="自动清理旧数据")


class IncrementalBuildResponse(BaseModel):
    """增量构建响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    task_id: str = Field(..., description="任务ID")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="统计信息")


class DocumentChangeDetectionRequest(BaseModel):
    """文档变化检测请求"""
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    collection_name: str = Field("default", description="集合名称")


class DocumentChangeDetectionResponse(BaseModel):
    """文档变化检测响应"""
    new_documents: List[Dict[str, Any]] = Field(..., description="新文档")
    updated_documents: List[Dict[str, Any]] = Field(..., description="更新文档")
    unchanged_documents: List[Dict[str, Any]] = Field(..., description="未变化文档")


class CleanupRequest(BaseModel):
    """清理请求"""
    max_days: int = Field(30, description="最大保留天数")
    collection_name: str = Field("default", description="集合名称")


class CleanupResponse(BaseModel):
    """清理响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    deleted_count: int = Field(0, description="删除数量")


@router.post("/build", response_model=IncrementalBuildResponse)
async def incremental_build(request: IncrementalBuildRequest):
    """增量构建知识图谱"""
    try:
        # 获取配置
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        if not config.deepseek_api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API密钥未设置")
        
        # 创建增量配置
        incremental_config = None
        if request.incremental_config:
            incremental_config = IncrementalConfig(
                enable_content_hash=request.incremental_config.enable_content_hash,
                enable_timestamp_check=request.incremental_config.enable_timestamp_check,
                merge_strategy=request.incremental_config.merge_strategy,
                max_retention_days=request.incremental_config.max_retention_days,
                batch_size=request.incremental_config.batch_size
            )
        
        # 创建增量构建器
        builder = IncrementalKnowledgeGraphBuilder(config, incremental_config)
        
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # 直接执行增量构建（同步）
        try:
            # 执行增量构建
            result = await builder.incremental_build(
                documents=request.documents,
                batch_size=incremental_config.batch_size if incremental_config else 10
            )
            
            # 如果需要自动清理
            if request.auto_cleanup and incremental_config:
                cleanup_result = await builder.cleanup_old_data(
                    max_days=incremental_config.max_retention_days
                )
                result["cleanup_result"] = cleanup_result
            
            logger.info(f"增量构建任务完成: {task_id}")
            
            # 关闭构建器
            # 异步关闭构建器
            import asyncio
            if asyncio.iscoroutinefunction(builder.close):
                await builder.close()
            else:
                builder.close()
            
            return IncrementalBuildResponse(
                success=True,
                message="增量构建任务已完成",
                task_id=task_id,
                statistics=result.get("statistics", {})
            )
            
        except Exception as e:
            logger.error(f"增量构建任务失败: {e}")
            
            # 关闭构建器
            try:
                import asyncio
                if asyncio.iscoroutinefunction(builder.close):
                    await builder.close()
                else:
                    builder.close()
            except:
                pass
            
            raise HTTPException(status_code=500, detail=f"增量构建失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"启动增量构建失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动增量构建失败: {str(e)}")


@router.post("/detect-changes", response_model=DocumentChangeDetectionResponse)
async def detect_document_changes(request: DocumentChangeDetectionRequest):
    """检测文档变化"""
    try:
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建增量构建器
        builder = IncrementalKnowledgeGraphBuilder(config)
        
        # 检测变化
        changes = await builder.detect_document_changes(
            documents=request.documents,
            collection_name=request.collection_name
        )
        
        # 关闭构建器
        if asyncio.iscoroutinefunction(builder.close):
            await builder.close()
        else:
            builder.close()
        
        return DocumentChangeDetectionResponse(
            new_documents=changes.get("new_documents", []),
            updated_documents=changes.get("updated_documents", []),
            unchanged_documents=changes.get("unchanged_documents", [])
        )
        
    except Exception as e:
        logger.error(f"检测文档变化失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测文档变化失败: {str(e)}")


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_old_data(request: CleanupRequest):
    """清理旧数据"""
    try:
        config = GraphRAGFactory.create_config()
        
        if not config.graphrag_enabled:
            raise HTTPException(status_code=400, detail="GraphRAG未启用")
        
        # 创建增量构建器
        builder = IncrementalKnowledgeGraphBuilder(config)
        
        # 执行清理
        result = await builder.cleanup_old_data(
            max_days=request.max_days,
            collection_name=request.collection_name
        )
        
        # 关闭构建器
        if asyncio.iscoroutinefunction(builder.close):
            await builder.close()
        else:
            builder.close()
        
        return CleanupResponse(
            success=True,
            message=f"清理完成，删除 {result.get('deleted_count', 0)} 个文档",
            deleted_count=result.get("deleted_count", 0)
        )
        
    except Exception as e:
        logger.error(f"清理旧数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理旧数据失败: {str(e)}")


@router.get("/config")
async def get_incremental_config():
    """获取增量配置"""
    try:
        config = GraphRAGFactory.create_config()
        
        return {
            "graphrag_enabled": config.graphrag_enabled,
            "incremental_support": True,
            "default_batch_size": 10,
            "supported_merge_strategies": ["update", "replace", "merge"],
            "max_retention_days": 30
        }
        
    except Exception as e:
        logger.error(f"获取增量配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取增量配置失败: {str(e)}")