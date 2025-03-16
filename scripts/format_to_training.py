#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本格式化工具
将清洗后的王小波文本转换为适合大模型训练的格式
支持生成问答对、指令-响应格式等
"""

import os
import sys
import argparse
import re
import json
import random
import logging
from tqdm import tqdm
import jieba
import jieba.analyse
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 王小波风格的特点描述
STYLE_FEATURES = [
    "理性思考",
    "幽默反讽",
    "冷静叙述",
    "荒诞现实",
    "简洁直白",
    "批判精神",
    "哲理思考",
    "独特视角",
    "人文关怀",
    "黑色幽默",
    "自由意识"
]

# 不同类型的指令模板
INSTRUCTION_TEMPLATES = {
    "continuation": [
        "请用王小波的风格续写以下段落：",
        "以王小波的语言风格，继续写下去：",
        "模仿王小波的写作特点，接着往下写：",
        "用王小波的文风，为下面的文本写一个结尾：",
        "像王小波一样，继续这个故事："
    ],
    "style_conversion": [
        "请将以下文字改写为王小波的风格：",
        "用王小波式的语言表达下面的内容：",
        "如果王小波来写这段话，他会怎么写？",
        "将下面的内容转换为王小波的文风：",
        "以王小波的视角和语言风格，重写以下段落："
    ],
    "question_answering": [
        "作为王小波，你会如何回答这个问题？",
        "用王小波的语言风格回答以下问题：",
        "以王小波的思维方式，对以下问题发表看法：",
        "如果王小波在场，他会怎么回应这个问题？",
        "模仿王小波的口吻，回答下面的问题："
    ],
    "character_simulation": [
        "假设你是王小波，请对以下话题发表评论：",
        "以王小波的身份，谈谈你对这个现象的看法：",
        "作为王小波，请写一段关于以下主题的文字：",
        "模拟王小波的视角，对以下事件进行点评：",
        "用王小波的方式思考和表达对以下情境的看法："
    ]
}

# 分析段落特征的关键词
FEATURE_KEYWORDS = {
    "理性思考": ["理性", "思考", "逻辑", "分析", "理智", "推理", "判断", "检验"],
    "幽默反讽": ["幽默", "讽刺", "反讽", "嘲弄", "嘲笑", "调侃", "滑稽", "荒谬"],
    "冷静叙述": ["冷静", "平静", "客观", "叙述", "陈述", "描述", "记录", "观察"],
    "荒诞现实": ["荒诞", "怪诞", "荒唐", "离奇", "现实", "不合理", "超现实"],
    "简洁直白": ["简洁", "直白", "简单", "明了", "清晰", "简练", "直接", "明白"],
    "批判精神": ["批判", "质疑", "反对", "否定", "揭露", "批评", "怀疑", "责问"],
    "哲理思考": ["哲理", "思考", "哲学", "意义", "本质", "存在", "真相", "智慧"],
    "独特视角": ["独特", "视角", "角度", "眼光", "立场", "观点", "视野", "独到"],
    "人文关怀": ["人文", "关怀", "人性", "尊严", "自由", "价值", "人道", "尊重"],
    "黑色幽默": ["黑色", "幽默", "荒诞", "讽刺", "无奈", "黑暗", "无助", "嘲讽"],
    "自由意识": ["自由", "意识", "解放", "独立", "自主", "个性", "独立思考"]
}

def read_text_file(file_path):
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {str(e)}")
        return ""

def analyze_paragraph_style(paragraph):
    """分析段落中的王小波风格特征"""
    if not paragraph or len(paragraph) < 50:
        return []
    
    # 提取关键词
    keywords = jieba.analyse.extract_tags(paragraph, topK=10, withWeight=False)
    
    # 匹配风格特征
    matched_features = []
    for feature, feature_keywords in FEATURE_KEYWORDS.items():
        for kw in keywords:
            if kw in feature_keywords:
                matched_features.append(feature)
                break
    
    # 如果没有匹配到任何特征，尝试通过简单规则匹配
    if not matched_features:
        if "？" in paragraph and "。" in paragraph:
            matched_features.append("理性思考")
        if any(word in paragraph for word in ["哈哈", "呵呵", "笑话"]):
            matched_features.append("幽默反讽")
        if len(re.findall(r'[。！？]', paragraph)) > 5:
            matched_features.append("冷静叙述")
    
    # 如果仍然没有匹配到特征，随机选择1-2个
    if not matched_features:
        matched_features = random.sample(STYLE_FEATURES, min(2, len(STYLE_FEATURES)))
    
    return matched_features

def split_into_meaningful_chunks(paragraphs, min_length=100, max_length=500):
    """将段落分割成有意义的文本块，适合训练"""
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if not p.strip():
            continue
        
        # 如果当前段落加上当前块长度超过最大长度，保存当前块并重新开始
        if len(current_chunk) + len(p) > max_length and len(current_chunk) >= min_length:
            chunks.append(current_chunk.strip())
            current_chunk = p
        else:
            if current_chunk:
                current_chunk += "\n\n" + p
            else:
                current_chunk = p
    
    # 添加最后一个块
    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_training_samples(text_chunks, format_type="mixed", num_samples=None):
    """
    从文本块创建训练样本
    
    参数:
    text_chunks: 文本块列表
    format_type: 训练样本格式类型，可选 "qa_pairs", "instruction_response", "mixed"
    num_samples: 要生成的样本数量，None表示使用所有可能的块
    
    返回:
    训练样本列表
    """
    if not text_chunks:
        return []
    
    # 如果未指定样本数，使用所有块
    if num_samples is None or num_samples > len(text_chunks):
        num_samples = len(text_chunks)
    
    # 随机选择块
    selected_chunks = random.sample(text_chunks, num_samples)
    samples = []
    
    for chunk in tqdm(selected_chunks, desc="生成训练样本"):
        # 分析风格特征
        features = analyze_paragraph_style(chunk)
        feature_text = "、".join(features) if features else "独特文风"
        
        # 根据格式类型创建样本
        if format_type == "qa_pairs" or (format_type == "mixed" and random.random() < 0.3):
            # 创建问答对
            qa_type = random.choice(["content", "style", "analysis"])
            
            if qa_type == "content":
                # 基于内容的问答
                question = f"这段文字主要表达了什么？\n\n{chunk}"
                answer = f"这段文字体现了王小波的{feature_text}。它通过{random.choice(['独特的视角', '冷静的叙述', '幽默的语言', '理性的思考'])}，表达了对{random.choice(['生活', '社会', '人性', '现实', '自由', '思考'])}的看法。"
            
            elif qa_type == "style":
                # 基于风格的问答
                question = f"这段文字有哪些王小波的风格特点？\n\n{chunk}"
                answer = f"这段文字体现了王小波的{feature_text}。王小波的写作风格特点包括理性思考、幽默反讽、冷静叙述和批判精神等。在这段文字中，可以看到他{random.choice(['用简洁直白的语言表达深刻的思想', '以冷静的语调描述荒诞的事物', '用幽默反讽的方式揭示现实问题', '理性思考生活中的荒谬现象'])}。"
            
            else:
                # 分析型问答
                question = f"请分析这段王小波式的文字：\n\n{chunk}"
                answer = f"这段文字展现了王小波的{feature_text}。从内容上看，{'它探讨了'+random.choice(['人的尊严与自由', '个体在集体中的处境', '荒诞与现实的关系', '权力与知识的冲突']) if random.random() < 0.5 else '它描述了日常生活中的'+random.choice(['荒诞现象', '人性矛盾', '个体挣扎', '理性思考'])}；从形式上看，{'它采用了'+random.choice(['冷静客观的叙述方式', '简洁直白的语言风格', '幽默反讽的表达手法', '跳跃性的叙事结构']) if random.random() < 0.5 else '它体现了王小波善于'+random.choice(['用简单的语言表达复杂的思想', '以个人经历反映社会现实', '通过荒诞揭示真实', '在平淡中见出不平凡'])}。"
            
            samples.append({
                "instruction": question,
                "input": "",
                "output": answer
            })
        
        elif format_type == "instruction_response" or format_type == "mixed":
            # 创建指令-响应格式
            # 随机选择指令类型
            instruction_type = random.choice(list(INSTRUCTION_TEMPLATES.keys()))
            
            if instruction_type == "continuation":
                # 文本续写
                # 随机截取前半部分作为输入
                split_point = len(chunk) // 2
                input_text = chunk[:split_point].strip()
                output_text = chunk[split_point:].strip()
                
                instruction = random.choice(INSTRUCTION_TEMPLATES["continuation"])
                
                samples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
            
            elif instruction_type == "style_conversion":
                # 风格转换（模拟）
                # 创建一个"普通风格"的文本，然后让模型转换为王小波风格
                input_text = "这是一段需要改写的普通文字。它可能来自新闻报道、学术论文或日常对话。请将它转换为王小波的风格，保持原意但增加王小波式的语言特点。"
                
                instruction = random.choice(INSTRUCTION_TEMPLATES["style_conversion"])
                
                samples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": chunk
                })
            
            elif instruction_type == "question_answering":
                # 问题回答
                questions = [
                    "你认为什么是真正的自由？",
                    "现代社会中，个体如何保持独立思考？",
                    "权力与知识的关系是什么？",
                    "如何看待生活中的荒诞现象？",
                    "理性思考在当今社会的重要性是什么？",
                    "个人的价值应该如何体现？",
                    "如何评价当代文化现象？",
                    "人应该如何面对生活中的困境？",
                    "什么是有意义的人生？",
                    "如何看待社会规则与个人自由的冲突？"
                ]
                
                input_text = random.choice(questions)
                instruction = random.choice(INSTRUCTION_TEMPLATES["question_answering"])
                
                samples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": chunk
                })
            
            elif instruction_type == "character_simulation":
                # 角色模拟
                topics = [
                    "现代教育",
                    "城市生活",
                    "科技发展",
                    "人际关系",
                    "文学创作",
                    "社会现象",
                    "历史事件",
                    "文化传统",
                    "个人成长",
                    "权力与自由"
                ]
                
                input_text = random.choice(topics)
                instruction = random.choice(INSTRUCTION_TEMPLATES["character_simulation"])
                
                samples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": chunk
                })
    
    return samples

def format_text_for_training(input_file, output_file=None, format_type="mixed", max_samples=None):
    """
    将文本格式化为训练数据
    
    参数:
    input_file: 输入文本文件路径
    output_file: 输出JSON文件路径
    format_type: 训练样本格式类型
    max_samples: 最大样本数量
    
    返回:
    输出文件路径
    """
    try:
        logger.info(f"正在处理文件: {input_file}")
        
        # 如果未指定输出路径，则使用输入文件名+_training后缀
        if output_file is None:
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = name_without_ext + "_training.json"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 读取文本文件
        text = read_text_file(input_file)
        if not text:
            logger.error("文本内容为空")
            return None
        
        # 分割为段落
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        logger.info(f"从文件中提取 {len(paragraphs)} 个段落")
        
        # 将段落组合成有意义的文本块
        text_chunks = split_into_meaningful_chunks(paragraphs)
        logger.info(f"创建了 {len(text_chunks)} 个文本块")
        
        # 创建训练样本
        samples = create_training_samples(text_chunks, format_type, max_samples)
        logger.info(f"生成了 {len(samples)} 个训练样本")
        
        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"训练数据已保存至: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        raise

def process_directory(input_dir, output_dir, format_type="mixed", max_samples_per_file=None):
    """处理目录中的所有文本文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    total = len(txt_files)
    
    logger.info(f"在目录 {input_dir} 中找到 {total} 个文本文件")
    
    all_samples = []
    
    for i, txt_file in enumerate(txt_files):
        input_path = os.path.join(input_dir, txt_file)
        output_name = os.path.splitext(txt_file)[0] + "_training.json"
        output_path = os.path.join(output_dir, output_name)
        
        logger.info(f"处理文件 {i+1}/{total}: {txt_file}")
        try:
            format_text_for_training(input_path, output_path, format_type, max_samples_per_file)
        except Exception as e:
            logger.error(f"处理 {txt_file} 失败: {str(e)}")
    
    # 合并所有样本到一个文件（可选）
    if total > 1:
        combined_output = os.path.join(output_dir, "combined_training.json")
        logger.info(f"合并所有样本到: {combined_output}")
        
        with open(combined_output, 'w', encoding='utf-8') as f:
            for sample_file in os.listdir(output_dir):
                if sample_file.endswith("_training.json") and sample_file != "combined_training.json":
                    file_path = os.path.join(output_dir, sample_file)
                    with open(file_path, 'r', encoding='utf-8') as sf:
                        for line in sf:
                            f.write(line)

def main():
    parser = argparse.ArgumentParser(description="将文本转换为适合大模型训练的格式")
    parser.add_argument("input", help="输入的文本文件或包含文本文件的目录")
    parser.add_argument("-o", "--output", help="输出的JSON文件或目录")
    parser.add_argument("-f", "--format", choices=["qa_pairs", "instruction_response", "mixed"], 
                        default="mixed", help="训练样本格式类型 (默认: mixed)")
    parser.add_argument("-n", "--num_samples", type=int, help="每个文件生成的最大样本数")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    format_type = args.format
    max_samples = args.num_samples
    
    # 检查输入路径
    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    if os.path.isdir(input_path):
        # 处理目录
        output_dir = output_path if output_path else os.path.join(input_path, "training")
        process_directory(input_path, output_dir, format_type, max_samples)
    else:
        # 处理单个文件
        try:
            format_text_for_training(input_path, output_path, format_type, max_samples)
        except Exception as e:
            logger.error(f"格式化失败: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main() 