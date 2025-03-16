#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本清洗与去重工具
用于处理王小波作品文本，去除重复内容和噪声数据
"""

import os
import sys
import argparse
import re
import logging
import hashlib
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text, remove_patterns=None):
    """
    清洗文本内容
    
    参数:
    text: 要清洗的文本
    remove_patterns: 要移除的正则表达式模式列表
    
    返回:
    清洗后的文本
    """
    # 处理默认移除模式
    if remove_patterns is None:
        remove_patterns = [
            r'书名:.*?\n',                    # 移除书名信息
            r'作者:.*?\n',                    # 移除作者信息
            r'={20} 章节 \d+ ={20}\n',        # 移除章节分隔符
            r'\[图片\]',                      # 移除图片标记
            r'第\s*[一二三四五六七八九十零〇百千万亿\d]+\s*[章节卷回集部分]',  # 移除章节标记
            r'\*{3,}',                       # 移除分隔符
            r'[_＿]{3,}',                     # 移除下划线分隔符
            r'\.{3,}',                       # 移除省略号分隔符
            r'^[\s\d]+$',                    # 移除只包含数字和空白的行
            r'http[s]?://\S+',              # 移除URL
            r'提示：.*?文字版.*'               # 移除电子书提示信息
        ]
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 应用移除模式
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text)
    
    # 删除空行
    lines = [line for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    return text

def split_into_paragraphs(text):
    """将文本分割为段落"""
    # 按空行分割文本
    paragraphs = re.split(r'\n\s*\n', text)
    # 过滤掉过短或空的段落
    return [p.strip() for p in paragraphs if len(p.strip()) > 10]

def is_duplicate(p1, p2, threshold=0.8):
    """判断两个段落是否重复（基于简单相似度）"""
    # 使用Jaccard相似度
    if not p1 or not p2:
        return False
    
    # 对于非常短的段落，使用直接比较
    if len(p1) < 20 or len(p2) < 20:
        return p1 == p2
    
    # 计算字符级别的Jaccard相似度
    set1 = set(p1)
    set2 = set(p2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union > 0 else 0
    
    return similarity > threshold

def remove_duplicates_by_similarity(paragraphs, threshold=0.85):
    """使用相似度计算去除重复段落"""
    if len(paragraphs) < 2:
        return paragraphs
    
    logger.info(f"开始去除重复段落，共 {len(paragraphs)} 个段落...")
    
    # 使用TF-IDF向量化段落
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
    except ValueError as e:
        logger.error(f"向量化段落时出错: {str(e)}")
        # 如果无法向量化，返回原始段落
        return paragraphs
    
    # 计算相似度矩阵
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 标记要保留的段落
    keep = np.ones(len(paragraphs), dtype=bool)
    
    # 使用tqdm显示进度
    for i in tqdm(range(len(paragraphs))):
        if keep[i]:  # 如果当前段落标记为保留
            # 找出与当前段落相似度高于阈值的所有段落
            similar_indices = np.where((cosine_sim[i] > threshold) & (np.arange(len(paragraphs)) > i))[0]
            # 将这些相似段落标记为不保留
            keep[similar_indices] = False
    
    # 过滤掉不保留的段落
    unique_paragraphs = [p for i, p in enumerate(paragraphs) if keep[i]]
    logger.info(f"去重完成，剩余 {len(unique_paragraphs)} 个段落")
    
    return unique_paragraphs

def clean_file(input_file, output_file=None, duplicate_threshold=0.85):
    """
    清洗文本文件，去除噪声和重复内容
    
    参数:
    input_file: 输入文本文件路径
    output_file: 输出文本文件路径，如果不指定则使用input_file_cleaned.txt
    duplicate_threshold: 判断重复的相似度阈值
    
    返回:
    输出文件路径
    """
    try:
        logger.info(f"正在处理文件: {input_file}")
        
        # 如果未指定输出路径，则使用输入文件名+_cleaned后缀
        if output_file is None:
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = name_without_ext + "_cleaned.txt"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 读取文本文件
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # 清洗文本
        text = clean_text(text)
        
        # 分割为段落
        paragraphs = split_into_paragraphs(text)
        logger.info(f"从文件中提取 {len(paragraphs)} 个段落")
        
        # 去除重复段落
        unique_paragraphs = remove_duplicates_by_similarity(paragraphs, threshold=duplicate_threshold)
        
        # 写入清洗后的文本
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(unique_paragraphs))
        
        logger.info(f"清洗后的文本已保存至: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        raise

def process_directory(input_dir, output_dir, duplicate_threshold=0.85):
    """处理目录中的所有文本文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    total = len(txt_files)
    
    logger.info(f"在目录 {input_dir} 中找到 {total} 个文本文件")
    
    for i, txt_file in enumerate(txt_files):
        input_path = os.path.join(input_dir, txt_file)
        output_name = os.path.splitext(txt_file)[0] + "_cleaned.txt"
        output_path = os.path.join(output_dir, output_name)
        
        logger.info(f"处理文件 {i+1}/{total}: {txt_file}")
        try:
            clean_file(input_path, output_path, duplicate_threshold)
        except Exception as e:
            logger.error(f"处理 {txt_file} 失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="清洗文本文件，去除噪声和重复内容")
    parser.add_argument("input", help="输入的文本文件或包含文本文件的目录")
    parser.add_argument("-o", "--output", help="输出的文本文件或目录")
    parser.add_argument("-t", "--threshold", type=float, default=0.85, 
                        help="判断重复的相似度阈值 (0-1之间，默认0.85)")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    threshold = args.threshold
    
    # 检查输入路径
    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    if os.path.isdir(input_path):
        # 处理目录
        output_dir = output_path if output_path else os.path.join(input_path, "cleaned")
        process_directory(input_path, output_dir, threshold)
    else:
        # 处理单个文件
        try:
            clean_file(input_path, output_path, threshold)
        except Exception as e:
            logger.error(f"清洗失败: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main() 