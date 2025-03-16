#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EPUB转文本工具：将EPUB电子书转换为纯文本格式
用于提取王小波作品内容，为大模型训练准备语料
"""

import os
import sys
import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chapter_to_text(chapter_content):
    """将HTML章节内容转换为纯文本"""
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(chapter_content, 'html.parser')
    
    # 去除脚本和样式内容
    for script in soup(["script", "style"]):
        script.extract()
    
    # 获取文本
    text = soup.get_text()
    
    # 处理空白字符
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def epub_to_text(epub_path, output_path=None, chapter_split=True):
    """
    将EPUB文件转换为文本文件
    
    参数:
    epub_path: EPUB文件路径
    output_path: 输出文本文件路径，如果不指定则使用EPUB文件名.txt
    chapter_split: 是否按章节分割文本
    
    返回:
    输出文件路径
    """
    try:
        logger.info(f"正在处理EPUB文件: {epub_path}")
        
        # 读取EPUB文件
        book = epub.read_epub(epub_path)
        
        # 如果未指定输出路径，则使用EPUB文件名
        if output_path is None:
            base_name = os.path.basename(epub_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = name_without_ext + ".txt"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 提取书籍元数据
        title = book.get_metadata('DC', 'title')
        title = title[0][0] if title else "未知书名"
        
        creator = book.get_metadata('DC', 'creator')
        creator = creator[0][0] if creator else "未知作者"
        
        logger.info(f"书名: {title}, 作者: {creator}")
        
        # 提取章节内容
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content()
                # 将内容解码为文本
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                text = chapter_to_text(content)
                if text.strip():  # 只添加非空文本
                    chapters.append(text)
        
        logger.info(f"共提取 {len(chapters)} 个章节")
        
        # 写入文本文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入书籍信息
            f.write(f"书名: {title}\n")
            f.write(f"作者: {creator}\n\n")
            
            # 写入章节内容
            for i, chapter in enumerate(chapters):
                if chapter_split:
                    f.write(f"\n\n{'='*20} 章节 {i+1} {'='*20}\n\n")
                f.write(chapter)
                f.write("\n\n")
        
        logger.info(f"文本已保存至: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"处理EPUB文件时出错: {str(e)}")
        raise

def process_directory(input_dir, output_dir):
    """处理目录中的所有EPUB文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epub_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.epub')]
    total = len(epub_files)
    
    logger.info(f"在目录 {input_dir} 中找到 {total} 个EPUB文件")
    
    for i, epub_file in enumerate(epub_files):
        input_path = os.path.join(input_dir, epub_file)
        output_name = os.path.splitext(epub_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)
        
        logger.info(f"处理文件 {i+1}/{total}: {epub_file}")
        try:
            epub_to_text(input_path, output_path)
        except Exception as e:
            logger.error(f"处理 {epub_file} 失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="将EPUB电子书转换为纯文本格式")
    parser.add_argument("input", help="输入的EPUB文件或包含EPUB文件的目录")
    parser.add_argument("-o", "--output", help="输出的文本文件或目录")
    parser.add_argument("--no-chapter-split", action="store_true", help="不按章节分割文本")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    chapter_split = not args.no_chapter_split
    
    # 检查输入路径
    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    if os.path.isdir(input_path):
        # 处理目录
        output_dir = output_path if output_path else input_path + "_text"
        process_directory(input_path, output_dir)
    else:
        # 处理单个文件
        try:
            epub_to_text(input_path, output_path, chapter_split)
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main() 