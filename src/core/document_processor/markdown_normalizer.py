"""
Markdown标准化工具

将Docling等工具输出的非标准Markdown格式转换为标准Markdown格式，
确保LangChain的MarkdownHeaderTextSplitter能正确解析。
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class MarkdownNormalizer:
    """Markdown标准化器"""
    
    def __init__(self):
        pass
    
    def normalize(self, markdown_text: str) -> str:
        """
        标准化Markdown文本
        
        Args:
            markdown_text: 原始Markdown文本
            
        Returns:
            标准化后的Markdown文本
        """
        if not markdown_text:
            return ""
        
        # 记录原始长度
        original_length = len(markdown_text)
        
        # 执行标准化步骤
        text = markdown_text
        
        # 步骤1: 修复列表格式
        text = self._fix_list_formats(text)
        
        # 步骤2: 修复标题格式
        text = self._fix_header_formats(text)
        
        # 步骤3: 清理冗余符号
        text = self._clean_redundant_symbols(text)
        
        # 步骤4: 表格格式标准化
        text = self._fix_table_formats(text)
        
        # 步骤5: 特殊符号规范化
        text = self._normalize_special_symbols(text)
        
        # 步骤6: Markdown语法验证和修复
        text = self._validate_and_fix_markdown(text)
        
        # 记录标准化结果
        normalized_length = len(text)
        logger.debug(
            "Markdown标准化完成",
            original_length=original_length,
            normalized_length=normalized_length,
            changes_made=original_length != normalized_length
        )
        
        return text
    
    def _fix_list_formats(self, text: str) -> str:
        """修复列表格式"""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            normalized_line = line
            
            # 修复格式1: "- 1 、标题" → "1. 标题"
            # 匹配: - 数字 、内容
            pattern1 = r'^(\s*)- (\d+) 、(.+)$'
            match1 = re.match(pattern1, normalized_line)
            if match1:
                indent = match1.group(1)
                number = match1.group(2)
                content = match1.group(3)
                normalized_line = f"{indent}{number}. {content}"
            
            # 修复格式2: "1.标题" → "1. 标题"
            # 匹配: 数字.内容（无空格）
            pattern2 = r'^(\s*)(\d+)\.([^\s].+)$'
            match2 = re.match(pattern2, normalized_line)
            if match2:
                indent = match2.group(1)
                number = match2.group(2)
                content = match2.group(3)
                normalized_line = f"{indent}{number}. {content}"
            
            # 修复格式3: "(1)内容" → "1. 内容"
            # 匹配: (数字)内容
            pattern3 = r'^(\s*)\((\d+)\)([^\s].+)$'
            match3 = re.match(pattern3, normalized_line)
            if match3:
                indent = match3.group(1)
                number = match3.group(2)
                content = match3.group(3)
                normalized_line = f"{indent}{number}. {content}"
            
            # 修复格式4: "一、标题" → "1. 标题"（中文数字转阿拉伯数字）
            chinese_numbers = {
                '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
            }
            pattern4 = r'^(\s*)([一二三四五六七八九十])、(.+)$'
            match4 = re.match(pattern4, normalized_line)
            if match4:
                indent = match4.group(1)
                chinese_num = match4.group(2)
                content = match4.group(3)
                arabic_num = chinese_numbers.get(chinese_num, chinese_num)
                normalized_line = f"{indent}{arabic_num}. {content}"
            
            normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines)
    
    def _fix_header_formats(self, text: str) -> str:
        """修复标题格式"""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            normalized_line = line
            
            # 修复格式1: "#标题" → "# 标题"
            pattern1 = r'^(#{1,6})([^\s#].+)$'
            match1 = re.match(pattern1, normalized_line)
            if match1:
                hashes = match1.group(1)
                content = match1.group(2)
                normalized_line = f"{hashes} {content}"
            
            # 修复格式2: "##1.2" → "## 1.2"
            pattern2 = r'^(#{1,6})(\d+\.\d+.*)$'
            match2 = re.match(pattern2, normalized_line)
            if match2:
                hashes = match2.group(1)
                content = match2.group(2)
                normalized_line = f"{hashes} {content}"
            
            # 修复格式3: "###(1)" → "### (1)"
            pattern3 = r'^(#{1,6})(\(.+\))$'
            match3 = re.match(pattern3, normalized_line)
            if match3:
                hashes = match3.group(1)
                content = match3.group(2)
                normalized_line = f"{hashes} {content}"
            
            normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines)
    
    def _clean_redundant_symbols(self, text: str) -> str:
        """清理冗余符号"""
        # 1. 全角空格转半角空格
        text = text.replace('　', ' ')
        
        # 2. 制表符转空格（4个空格）
        text = text.replace('\t', '    ')
        
        # 3. 合并连续空行（最多保留2个空行）
        lines = text.split('\n')
        cleaned_lines = []
        empty_line_count = 0
        
        for line in lines:
            if line.strip() == '':
                empty_line_count += 1
                if empty_line_count <= 2:
                    cleaned_lines.append(line)
            else:
                empty_line_count = 0
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 4. 移除行尾空格
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 5. 移除行首多余空格（保留缩进）
        lines = text.split('\n')
        lines = [re.sub(r'^ {1,3}(?=\S)', '', line) for line in lines]  # 移除1-3个前导空格
        text = '\n'.join(lines)
        
        return text
    
    def _fix_table_formats(self, text: str) -> str:
        """修复表格格式"""
        lines = text.split('\n')
        normalized_lines = []
        in_table = False
        table_lines = []
        
        for i, line in enumerate(lines):
            # 检测表格开始（包含|字符的行）
            if '|' in line and not line.strip().startswith('#'):
                in_table = True
                table_lines.append(line)
            elif in_table:
                # 表格结束，处理收集的表格行
                if table_lines:
                    normalized_table = self._normalize_table(table_lines)
                    normalized_lines.extend(normalized_table)
                    table_lines = []
                in_table = False
                normalized_lines.append(line)
            else:
                normalized_lines.append(line)
        
        # 处理最后可能剩余的表格行
        if table_lines:
            normalized_table = self._normalize_table(table_lines)
            normalized_lines.extend(normalized_table)
        
        return '\n'.join(normalized_lines)
    
    def _normalize_table(self, table_lines: List[str]) -> List[str]:
        """标准化表格行"""
        if len(table_lines) < 2:
            return table_lines
        
        normalized = []
        
        # 处理表头
        header = table_lines[0]
        normalized.append(header)
        
        # 处理分隔线（第二行）
        if len(table_lines) > 1:
            separator = table_lines[1]
            # 确保分隔线连续且长度足够
            separator = re.sub(r'-{2,}', '---', separator)
            # 确保每个单元格都有分隔线
            cells = separator.split('|')
            normalized_separator = '|'.join(['---' if '---' in cell else cell for cell in cells])
            normalized.append(normalized_separator)
        
        # 处理数据行
        for i in range(2, len(table_lines)):
            data_line = table_lines[i]
            # 修复单元格内的换行问题
            data_line = re.sub(r'\s+', ' ', data_line)  # 合并多个空格
            # 确保单元格内容不包含换行符
            data_line = data_line.replace('\n', ' ').replace('\r', ' ')
            normalized.append(data_line)
        
        return normalized
    
    def _normalize_special_symbols(self, text: str) -> str:
        """特殊符号规范化"""
        # 1. 中文顿号转逗号（可选，根据需求）
        # text = text.replace('、', ',')
        
        # 2. 中文分号转英文分号
        text = text.replace('；', ';')
        
        # 3. 中文冒号转英文冒号
        text = text.replace('：', ':')
        
        # 4. 中文括号转英文括号（可选）
        # text = text.replace('（', '(').replace('）', ')')
        # text = text.replace('【', '[').replace('】', ']')
        # text = text.replace('《', '<').replace('》', '>')
        
        # 5. 统一引号（中文引号转英文引号）
        text = text.replace('「', '"').replace('」', '"')
        text = text.replace('『', "'").replace('』', "'")
        
        # 6. 移除零宽空格等不可见字符
        text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)
        
        # 7. 修复连续标点符号
        text = re.sub(r'([,.;:!?])\1+', r'\1', text)  # 重复标点保留一个
        
        return text
    
    def _validate_and_fix_markdown(self, text: str) -> str:
        """验证和修复Markdown语法"""
        lines = text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # 检查标题层级
            if line.strip().startswith('#'):
                # 确保标题后有空行（除非是文档开头）
                if i > 0 and lines[i-1].strip() != '':
                    # 在标题前插入空行
                    if fixed_lines and fixed_lines[-1].strip() != '':
                        fixed_lines.append('')
            
            # 检查列表项格式
            if re.match(r'^\s*[\d\-*+.]\s', line):
                # 确保列表项前有空行（除非是连续列表项）
                if i > 0 and not re.match(r'^\s*[\d\-*+.]\s', lines[i-1]):
                    if fixed_lines and fixed_lines[-1].strip() != '':
                        fixed_lines.append('')
            
            fixed_lines.append(fixed_line)
        
        # 确保文档以空行结束
        if fixed_lines and fixed_lines[-1].strip() != '':
            fixed_lines.append('')
        
        return '\n'.join(fixed_lines)
    
    def get_normalization_stats(self, original: str, normalized: str) -> dict:
        """获取标准化统计信息"""
        return {
            "original_length": len(original),
            "normalized_length": len(normalized),
            "lines_changed": sum(1 for o, n in zip(original.split('\n'), normalized.split('\n')) if o != n),
            "change_percentage": round((1 - sum(o == n for o, n in zip(original.split('\n'), normalized.split('\n'))) / max(len(original.split('\n')), 1)) * 100, 2)
        }


# 全局标准化器实例
markdown_normalizer = MarkdownNormalizer()


def normalize_markdown(text: str) -> str:
    """标准化Markdown文本（便捷函数）"""
    return markdown_normalizer.normalize(text)


if __name__ == "__main__":
    # 测试代码
    test_input = """#解析信息
- 原文件: employee_handbook.pdf
- 解析器: docling
- 时间: Sat Apr 11 17:07:48 2026
- 文本长度: 26819 字符

#解析内容

##尊敬的各位员工：

您好！为健全管理制度和组织功能，规范员工行为，提升员工队伍整体素质， 明确公司和员工双方的权利和义务， 公司（以下简称'公司'） 特地制订本手册。

##员工手册

##前言

- 1 、公司简介（略）
- 2 、公司使命（略）

##第一条 适用范围

本制度适用于公司所有部门（包括生产、销售、财务等各部分，以下简称各部 门），适用于所有与公司建立全日制劳动关系员工。

| 审批权限（副总以下人员任免）   | 权限对应职位                 |
|------------------|------------------------|
| 建议权              | 用人部门主管副总经理、人力资源部长、总 经理 |
| 确定、否决权           | 总经理                    |"""

    normalizer = MarkdownNormalizer()
    result = normalizer.normalize(test_input)
    
    print("原始文本:")
    print(test_input)
    print("\n" + "="*80 + "\n")
    print("标准化后:")
    print(result)
    
    stats = normalizer.get_normalization_stats(test_input, result)
    print("\n" + "="*80 + "\n")
    print("标准化统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")