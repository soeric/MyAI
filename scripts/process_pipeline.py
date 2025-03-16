#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
王小波小说微调数据处理流水线
整合了从EPUB提取文本、清洗去重、格式化训练数据的全过程
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

def run_command(command, description):
    """运行命令并记录输出"""
    logger.info(f"执行任务: {description}")
    logger.info(f"命令: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 实时输出命令执行情况
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # 获取退出码和错误输出
        return_code = process.poll()
        error_output = process.stderr.read()
        
        if return_code != 0:
            logger.error(f"命令执行失败，退出码: {return_code}")
            logger.error(f"错误信息: {error_output}")
            return False
        
        logger.info(f"任务完成: {description}")
        return True
    
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        return False

def process_epub_to_text(input_path, output_dir, no_chapter_split=False):
    """步骤1: 将EPUB转换为文本"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epub_to_text.py")
    
    command = [sys.executable, script_path, input_path, "-o", output_dir]
    if no_chapter_split:
        command.append("--no-chapter-split")
    
    return run_command(command, "EPUB转文本")

def clean_text_files(input_dir, output_dir, threshold=0.85):
    """步骤2: 清洗和去重文本"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_cleaner.py")
    
    command = [sys.executable, script_path, input_dir, "-o", output_dir, "-t", str(threshold)]
    
    return run_command(command, "文本清洗与去重")

def format_for_training(input_dir, output_dir, format_type="mixed", num_samples=None):
    """步骤3: 格式化为训练数据"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "format_to_training.py")
    
    command = [sys.executable, script_path, input_dir, "-o", output_dir, "-f", format_type]
    if num_samples:
        command.extend(["-n", str(num_samples)])
    
    return run_command(command, "格式化为训练数据")

def process_pipeline(input_path, output_base_dir, config):
    """
    执行完整的处理流水线
    
    参数:
    input_path: EPUB文件或包含EPUB文件的目录
    output_base_dir: 输出基础目录
    config: 配置参数
    """
    start_time = time.time()
    logger.info("开始处理流水线")
    
    # 创建输出目录结构
    ensure_directory(output_base_dir)
    
    text_dir = os.path.join(output_base_dir, "01_text")
    cleaned_dir = os.path.join(output_base_dir, "02_cleaned")
    training_dir = os.path.join(output_base_dir, "03_training")
    
    ensure_directory(text_dir)
    ensure_directory(cleaned_dir)
    ensure_directory(training_dir)
    
    # 步骤1: EPUB转文本
    if config.skip_epub_conversion:
        logger.info("跳过EPUB转文本步骤")
    else:
        if not process_epub_to_text(input_path, text_dir, config.no_chapter_split):
            logger.error("EPUB转文本失败，流水线中断")
            return False
    
    # 步骤2: 清洗文本
    if config.skip_cleaning:
        logger.info("跳过文本清洗步骤")
    else:
        text_input_dir = text_dir if not config.skip_epub_conversion else input_path
        if not clean_text_files(text_input_dir, cleaned_dir, config.similarity_threshold):
            logger.error("文本清洗失败，流水线中断")
            return False
    
    # 步骤3: 格式化为训练数据
    if config.skip_formatting:
        logger.info("跳过格式化为训练数据步骤")
    else:
        cleaned_input_dir = cleaned_dir if not config.skip_cleaning else (text_dir if not config.skip_epub_conversion else input_path)
        if not format_for_training(cleaned_input_dir, training_dir, config.format_type, config.num_samples):
            logger.error("格式化为训练数据失败，流水线中断")
            return False
    
    # 完成
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"流水线处理完成，总耗时: {duration:.2f}秒")
    
    # 输出最终结果路径
    final_output = os.path.join(training_dir, "combined_training.json")
    if os.path.exists(final_output):
        logger.info(f"最终训练数据位于: {final_output}")
    else:
        logger.info(f"训练数据目录: {training_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="王小波小说微调数据处理流水线")
    parser.add_argument("input", help="输入的EPUB文件或包含EPUB文件的目录")
    parser.add_argument("-o", "--output", help="输出基础目录",
                        default="output")
    
    # EPUB转文本选项
    parser.add_argument("--no-chapter-split", action="store_true", 
                        help="不按章节分割文本")
    parser.add_argument("--skip-epub-conversion", action="store_true", 
                        help="跳过EPUB转文本步骤")
    
    # 清洗选项
    parser.add_argument("--similarity-threshold", type=float, default=0.85, 
                        help="判断重复的相似度阈值 (0-1之间，默认0.85)")
    parser.add_argument("--skip-cleaning", action="store_true", 
                        help="跳过文本清洗步骤")
    
    # 格式化选项
    parser.add_argument("--format-type", choices=["qa_pairs", "instruction_response", "mixed"], 
                        default="mixed", help="训练样本格式类型 (默认: mixed)")
    parser.add_argument("--num-samples", type=int, 
                        help="生成的训练样本数量")
    parser.add_argument("--skip-formatting", action="store_true", 
                        help="跳过格式化为训练数据步骤")
    
    args = parser.parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 执行流水线
    success = process_pipeline(args.input, args.output, args)
    
    if not success:
        logger.error("流水线处理失败")
        sys.exit(1)
    
    logger.info("流水线处理成功")

if __name__ == "__main__":
    main() 