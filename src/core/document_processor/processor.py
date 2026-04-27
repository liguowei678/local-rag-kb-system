# src/core/document_processor/processor.py
"""
文档处理器

实现多格式文档解析、智能分块和预处理流水线
"""

import asyncio
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

from src.core.logging import get_logger
from src.core.document_processor.models import (
    DocumentFormat,
    ChunkingStrategy,
    DocumentMetadata,
    DocumentChunk,
    ProcessingResult
)
from src.core.document_processor.markdown_normalizer import normalize_markdown

# LangChain分块器
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

logger = get_logger(__name__)


class DocumentProcessor:
    """文档处理器"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,  # 默认改为分层分块
        supported_formats: List[DocumentFormat] = None
    ):
        """
        初始化文档处理器

        Args:
            chunk_size: 分块大小(字符数)
            chunk_overlap: 分块重叠大小(字符数)
            chunking_strategy: 分块策略
            supported_formats: 支持的文档格式列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        if supported_formats is None:
            supported_formats = [
                DocumentFormat.TXT,
                DocumentFormat.MARKDOWN,
                DocumentFormat.PDF,
                DocumentFormat.DOCX
            ]
        self.supported_formats = supported_formats
        
        # 初始化LangChain分块器
        self._init_langchain_splitters()

        logger.info(
            "文档处理器已初始化",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy.value,
            supported_formats=[fmt.value for fmt in supported_formats]
        )

    def preprocess_text(self, text: str) -> str:
        """
        预处理文本

        Args:
            text: 原始文本

        Returns:
            预处理后的文本
        """
        if not text:
            return ""

        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符(保留中文、英文、数字和基本标点)
        text = re.sub(r'[^\w\s\u4e00-\u9fff,。!?;:、()《》【】「」,.!?;:()\[\]{}]', ' ', text)

        # 移除首尾空白
        text = text.strip()

        return text
    
    def _init_langchain_splitters(self):
        """初始化LangChain分块器"""
        # 定义标题层级（Markdown格式）
        self.headers_to_split_on = [
            ("#", "标题1"),
            ("##", "标题2"),
            ("###", "标题3"),
            ("####", "标题4")
        ]
        
        # 创建Markdown标题分块器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        # 创建递归字符分块器（用于二次分块）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,           # 中文优化：500字符
            chunk_overlap=100,        # 重叠100字符
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        
        logger.info("LangChain分块器已初始化")

    def _chunk_fixed(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        固定大小分块

        Args:
            text: 文本
            metadata: 元数据

        Returns:
            文档分块列表
        """
        if not text:
            return []

        chunks = []
        text_length = len(text)

        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            start = i
            end = min(i + self.chunk_size, text_length)

            chunk_text = text[start:end]

            # 确保不在单词中间分割
            if end < text_length and chunk_text and not chunk_text[-1].isspace():
                # 找到最后一个空格
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"chunk_{len(chunks):04d}",
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata=metadata or {}
            )
            chunks.append(chunk)

            if end >= text_length:
                break

        return chunks

    def _chunk_overlap(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        重叠分块

        Args:
            text: 文本
            metadata: 元数据

        Returns:
            文档分块列表
        """
        if not text:
            return []

        chunks = []
        text_length = len(text)

        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            start = max(0, i - self.chunk_overlap // 2)
            end = min(i + self.chunk_size, text_length)

            chunk_text = text[start:end]

            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"chunk_{len(chunks):04d}",
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata=metadata or {}
            )
            chunks.append(chunk)

            if end >= text_length:
                break

        return chunks

    def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy = None,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        文本分块

        Args:
            text: 文本
            strategy: 分块策略
            metadata: 元数据

        Returns:
            文档分块列表
        """
        if not text:
            return []

        # 使用指定策略或默认策略
        if strategy is None:
            strategy = self.chunking_strategy

        # 预处理文本
        processed_text = self.preprocess_text(text)

        # 根据策略选择分块方法
        if strategy == ChunkingStrategy.FIXED:
            chunks = self._chunk_fixed(processed_text, metadata)
        elif strategy == ChunkingStrategy.OVERLAP:
            chunks = self._chunk_overlap(processed_text, metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            # 语义分块(简化实现:按段落分块)
            chunks = self._chunk_semantic(processed_text, metadata)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            # 分层分块（按标题层级）
            chunks = self._chunk_hierarchical(processed_text, metadata)
        else:
            # 默认使用固定分块
            chunks = self._chunk_fixed(processed_text, metadata)

        logger.debug(
            "文本分块完成",
            original_length=len(text),
            processed_length=len(processed_text),
            chunk_count=len(chunks),
            strategy=strategy.value
        )

        return chunks

    def _chunk_semantic(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        语义分块(简化实现:按段落分块)

        Args:
            text: 文本
            metadata: 元数据

        Returns:
            文档分块列表
        """
        if not text:
            return []

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落不超过分块大小,则合并
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块
                if current_chunk:
                    chunk = DocumentChunk(
                        text=current_chunk,
                        chunk_id=f"chunk_{len(chunks):04d}",
                        index=len(chunks),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata=metadata or {}
                    )
                    chunks.append(chunk)
                    current_start += len(current_chunk) + 2

                # 开始新块
                current_chunk = para

        # 保存最后一个块
        if current_chunk:
            chunk = DocumentChunk(
                text=current_chunk,
                chunk_id=f"chunk_{len(chunks):04d}",
                index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks
    
    def _chunk_hierarchical(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        分层分块：按标题层级分块，大块进行二次分块
        
        Args:
            text: Markdown格式文本
            metadata: 元数据
            
        Returns:
            文档分块列表
        """
        if not text:
            return []
        
        # 在分块前彻底清理文本
        import re
        # 1. 移除所有开头空行
        text = re.sub(r'^\n+', '', text)
        # 2. 移除所有结尾空行
        text = re.sub(r'\n+$', '', text)
        # 3. 多个连续空行变一个
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 4. 移除开头空格
        text = text.lstrip()
        
        logger.debug(f"分块前文本清理，长度: {len(text)}")
        
        chunks = []
        
        try:
            # 第一步：按标题层级分块
            header_chunks = self.markdown_splitter.split_text(text)
            
            logger.debug(
                "标题层级分块完成",
                header_chunks_count=len(header_chunks)
            )
            
            # 第二步：处理每个标题块
            for i, header_chunk in enumerate(header_chunks):
                chunk_text = header_chunk.page_content
                chunk_metadata = header_chunk.metadata
                
                # 构建标题路径（用于元数据注入）
                header_path_parts = []
                for header_level in ["标题1", "标题2", "标题3", "标题4"]:
                    if header_level in chunk_metadata:
                        header_path_parts.append(chunk_metadata[header_level])
                
                header_path = " > ".join(header_path_parts) if header_path_parts else ""  # 空字符串而不是"无标题"
                
                # 检查是否需要二次分块（大于800字符）
                if len(chunk_text) > 800:
                    logger.debug(
                        "进行二次分块",
                        chunk_index=i,
                        original_length=len(chunk_text),
                        header_path=header_path
                    )
                    
                    # 使用递归分块器进行二次分块
                    sub_chunks = self.recursive_splitter.split_text(chunk_text)
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        # 注入标题路径
                        final_text = f"{header_path}\n\n{sub_chunk}"
                        
                        # 在存储前彻底清理文本
                        final_text = self._clean_chunk_text(final_text)
                        
                        chunk = DocumentChunk(
                            text=final_text,
                            chunk_id=f"chunk_{len(chunks):04d}",
                            index=len(chunks),
                            start_char=0,  # 分层分块不保留原始位置
                            end_char=len(final_text),
                            metadata={
                                **(metadata or {}),
                                "header_path": header_path,
                                "chunk_strategy": "hierarchical",
                                "chunk_level": "secondary",
                                "original_chunk_index": i,
                                "sub_chunk_index": j
                            }
                        )
                        chunks.append(chunk)
                else:
                    # 直接使用标题块
                    final_text = f"{header_path}\n\n{chunk_text}"
                    
                    # 在存储前彻底清理文本
                    final_text = self._clean_chunk_text(final_text)
                    
                    chunk = DocumentChunk(
                        text=final_text,
                        chunk_id=f"chunk_{len(chunks):04d}",
                        index=len(chunks),
                        start_char=0,
                        end_char=len(final_text),
                        metadata={
                            **(metadata or {}),
                            "header_path": header_path,
                            "chunk_strategy": "hierarchical",
                            "chunk_level": "primary"
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(
                "分层分块完成",
                original_length=len(text),
                final_chunks_count=len(chunks),
                avg_chunk_size=sum(len(c.text) for c in chunks) // max(len(chunks), 1)
            )
            
        except Exception as e:
            logger.error("分层分块失败", error=str(e))
            # 失败时回退到语义分块
            chunks = self._chunk_semantic(text, metadata)
        
        return chunks

    async def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析文件

        Args:
            file_path: 文件路径

        Returns:
            解析结果
        """
        start_time = time.time()

        try:
            path = Path(file_path)

            # 检查文件是否存在
            if not path.exists():
                logger.error("文件不存在", file_path=file_path)
                return {"error": "文件不存在"}

            # 获取文件格式
            file_format = DocumentFormat.from_extension(path.suffix)

            # 检查是否支持该格式
            if file_format not in self.supported_formats:
                logger.error("不支持的文档格式", format=file_format.value)
                return {"error": f"不支持的文档格式: {file_format.value}"}

            # 获取文件信息
            file_size = path.stat().st_size

            # 读取文件内容
            try:
                if file_format == DocumentFormat.TXT:
                    text = await self._read_text_file(path)
                elif file_format == DocumentFormat.MARKDOWN:
                    text = await self._read_markdown_file(path)
                elif file_format == DocumentFormat.PDF:
                    text = await self._read_pdf_file(path)
                elif file_format == DocumentFormat.DOCX:
                    text = await self._read_docx_file(path)
                else:
                    text = ""
            except Exception as e:
                logger.error("文件读取失败", file_path=file_path, error=str(e))
                return {"error": f"文件读取失败: {str(e)}"}

            # 创建元数据
            metadata = DocumentMetadata(
                filename=path.name,
                format=file_format,
                size=file_size,
                pages=None,  # 需要特定格式解析器提供
                language="zh-CN",  # 默认中文
                author=None,
                created_at=None,
                modified_at=None
            )

            processing_time = time.time() - start_time

            logger.info(
                "文件解析完成",
                file_path=file_path,
                format=str(file_format),
                size=file_size,
                text_length=len(text),
                processing_time=processing_time
            )

            return {
                "text": text,
                "metadata": metadata,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error("文件解析异常", file_path=file_path, error=str(e))
            return {"error": f"解析异常: {str(e)}"}

    async def _read_text_file(self, path: Path) -> str:
        """读取文本文件"""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                return path.read_text(encoding='gbk')
            except:
                return path.read_text(encoding='latin-1')

    async def _read_markdown_file(self, path: Path) -> str:
        """读取Markdown文件"""
        return await self._read_text_file(path)

    async def _read_pdf_file(self, path: Path) -> str:
        """读取PDF文件(使用Docling，自动判断解析方式)"""
        try:
            from docling.document_converter import DocumentConverter
            
            logger.info(
                "使用Docling解析PDF文档",
                file_path=str(path),
                format="docling"
            )

            # 简化：让Docling自动判断使用文本层还是OCR
            converter = DocumentConverter()
            result = converter.convert(str(path))

            # 导出为Markdown格式
            markdown_text = result.document.export_to_markdown()
            
            # 只做基本清理，保持原始结构
            import re
            # 清理开头空行
            markdown_text = re.sub(r'^\n+', '', markdown_text)
            # 清理过多空行
            markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
            # 清理首尾空白
            markdown_text = markdown_text.strip()
            
            logger.info(
                "Docling解析完成",
                file_path=str(path),
                text_length=len(markdown_text),
                format="markdown"
            )

            # 保存解析结果到文件(用于调试和质量检查)
            self._save_parsed_text(path, markdown_text, "docling_parsed")

            return markdown_text

        except ImportError as e:
            logger.error("Docling未安装,无法解析PDF")
            raise RuntimeError("Docling未安装,请安装docling包")
        except Exception as e:
            logger.error(
                "Docling解析失败",
                file_path=str(path),
                error=str(e)
            )
            raise RuntimeError(f"Docling解析PDF失败: {e}")
    def _fix_encoding(self, text: str) -> str:
        """修复文本编码和清理问题"""
        if not text:
            return text
        
        try:
            # 1. 清理开头和结尾的空白
            text = text.strip()
            
            # 2. 清理多余的空行（多个连续换行变一个）
            import re
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # 3. 清理开头的空行
            while text.startswith('\n'):
                text = text[1:]
            
            # 4. 清理结尾的空行
            while text.endswith('\n'):
                text = text[:-1]
            
            # 5. 替换常见的乱码字符
            replacements = {
                '�': ':',          # 乱码冒号
                '�': '工',         # 乱码汉字
                '�': '文',         # 乱码汉字
                '�': '件',         # 乱码汉字
                '�': '员',         # 乱码汉字
                '�': '称',         # 乱码汉字
                '�': '?',          # 未知字符
                '??': '?',         # 双问号
            }
            for wrong, right in replacements.items():
                text = text.replace(wrong, right)
            
            # 6. 清理多余空格（多个空格变一个）
            text = re.sub(r'\s{2,}', ' ', text)
            
            # 7. 如果文本是bytes，尝试解码
            if isinstance(text, bytes):
                try:
                    text = text.decode('utf-8', errors='ignore')
                except:
                    try:
                        text = text.decode('gbk', errors='ignore')
                    except:
                        text = text.decode('latin-1', errors='ignore')
            
            logger.debug(f"文本清理完成，长度: {len(text)} -> {len(text.strip())}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"文本清理过程中出错: {e}")
            return text
    
    def _clean_chunk_text(self, text: str) -> str:
        """专门清理chunk文本，确保存储质量"""
        if not text:
            return text
        
        try:
            import re
            
            # 1. 移除所有开头和结尾的空白字符
            text = text.strip()
            
            # 2. 移除开头的空行
            while text.startswith('\n'):
                text = text[1:].strip()
            
            # 3. 移除结尾的空行
            while text.endswith('\n'):
                text = text[:-1].strip()
            
            # 4. 规范化空格：多个空格变一个
            text = re.sub(r'\s{2,}', ' ', text)
            
            # 5. 规范化换行：多个换行变一个
            text = re.sub(r'\n{2,}', '\n', text)
            
            # 6. 移除开头的标点符号（如：、等）
            text = re.sub(r'^[、。，；：！？\s]+', '', text)
            
            # 7. 确保文本不为空
            if not text.strip():
                return "无内容"
            
            logger.debug(f"chunk文本清理完成，长度: {len(text)}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"chunk文本清理出错: {e}")
            return text.strip() if text else ""
    
    def _save_parsed_text(self, path: Path, text: str, parser: str) -> None:
        """保存解析的文本到文件(用于调试和质量检查)"""
        try:
            # 创建parsed_documents目录(在工作空间根目录)
            workspace_root = Path(__file__).parent.parent.parent.parent
            parsed_dir = workspace_root / "parsed_documents"
            parsed_dir.mkdir(exist_ok=True)

            # 生成文件名
            original_name = path.stem
            timestamp = int(time.time())
            # 如果是docling解析,保存为.md格式(Markdown)
            if parser == "docling":
                save_path = parsed_dir / f"{original_name}_{parser}_{timestamp}.md"
            else:
                save_path = parsed_dir / f"{original_name}_{parser}_{timestamp}.txt"

            # 保存文本
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"# 解析信息\n")
                f.write(f"- 原文件: {path.name}\n")
                f.write(f"- 解析器: {parser}\n")
                f.write(f"- 时间: {time.ctime()}\n")
                f.write(f"- 文本长度: {len(text)} 字符\n")
                f.write(f"\n# 解析内容\n\n")
                f.write(text)

            logger.debug(
                "解析文本已保存",
                save_path=str(save_path),
                text_length=len(text),
                parser=parser
            )
        except Exception as e:
            logger.warning(
                "保存解析文本失败",
                error=str(e)
            )

    async def _read_docx_file(self, path: Path) -> str:
        """读取DOCX文件(简化实现)"""
        # 在实际项目中,应该使用python-docx
        # 这里返回一个占位符文本
        logger.warning("DOCX解析使用简化实现,实际项目应集成python-docx")
        return f"[DOCX文件内容: {path.name}]"

    async def process_document(
        self,
        file_path: str,
        chunking_strategy: ChunkingStrategy = None
    ) -> Dict[str, Any]:
        """
        处理文档(完整流水线)

        Args:
            file_path: 文件路径
            chunking_strategy: 分块策略

        Returns:
            处理结果
        """
        start_time = time.time()

        try:
            # 1. 解析文件
            parse_result = await self.parse_file(file_path)

            if "error" in parse_result:
                return parse_result

            text = parse_result["text"]
            metadata = parse_result["metadata"]

            # 2. 文本分块
            chunks = self.chunk_text(
                text,
                strategy=chunking_strategy,
                metadata={"source": file_path}
            )

            # 3. 计算统计信息
            statistics = self.get_chunk_statistics(chunks)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                chunks=chunks,
                metadata=metadata,
                processing_time=processing_time,
                statistics=statistics
            )

            logger.info(
                "文档处理完成",
                file_path=file_path,
                format=metadata.format.value,
                chunk_count=len(chunks),
                processing_time=processing_time
            )

            return result.to_dict()

        except Exception as e:
            logger.error("文档处理失败", file_path=file_path, error=str(e))
            return {"error": f"处理失败: {str(e)}"}

    async def batch_process(
        self,
        file_paths: List[str],
        chunking_strategy: ChunkingStrategy = None
    ) -> List[Dict[str, Any]]:
        """
        批量处理文档

        Args:
            file_paths: 文件路径列表
            chunking_strategy: 分块策略

        Returns:
            处理结果列表
        """
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(file_path, chunking_strategy)
                results.append(result)
            except Exception as e:
                logger.error("批量处理失败", file_path=file_path, error=str(e))
                results.append({"error": f"处理失败: {str(e)}", "file_path": file_path})

        logger.info("批量处理完成", total_files=len(file_paths), success_count=len([r for r in results if "error" not in r]))

        return results

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        获取分块统计信息

        Args:
            chunks: 文档分块列表

        Returns:
            统计信息
        """
        if not chunks:
            return {}

        chunk_sizes = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks) if chunks else 0,
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "total_characters": sum(chunk_sizes),
            "chunk_size_distribution": {
                "small": len([s for s in chunk_sizes if s < 500]),
                "medium": len([s for s in chunk_sizes if 500 <= s <= 1500]),
                "large": len([s for s in chunk_sizes if s > 1500])
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态
        """
        return {
            "status": "healthy",
            "supported_formats": [fmt.value for fmt in self.supported_formats],
            "chunking_strategies": [s.value for s in ChunkingStrategy],
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_strategy": self.chunking_strategy.value
            }
        }


# 全局文档处理器实例
document_processor = DocumentProcessor()