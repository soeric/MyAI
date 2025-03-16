#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据准备脚本：将原始数据转换为LLaMA Factory可用的格式
"""

import json
import argparse
import pandas as pd
import os
from tqdm import tqdm

def convert_to_alpaca_format(input_file, output_file):
    """
    将自定义格式的数据转换为Alpaca格式
    Alpaca格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    print(f"正在将 {input_file} 转换为Alpaca格式...")
    
    # 读取输入文件 (示例，根据实际格式调整)
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # 假设CSV有question和answer列
            item = {
                "instruction": row.get("question", ""),
                "input": "",  # 如果有上下文，可以放在这里
                "output": row.get("answer", "")
            }
            data.append(item)
    elif input_file.endswith('.json') or input_file.endswith('.jsonl'):
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                raw_data = json.load(f)
                # 假设JSON是一个列表
                for item in tqdm(raw_data):
                    # 根据实际格式调整键名
                    alpaca_item = {
                        "instruction": item.get("question", item.get("prompt", "")),
                        "input": item.get("context", ""),
                        "output": item.get("answer", item.get("response", ""))
                    }
                    data.append(alpaca_item)
            else:  # JSONL格式
                for line in tqdm(f):
                    item = json.loads(line)
                    alpaca_item = {
                        "instruction": item.get("question", item.get("prompt", "")),
                        "input": item.get("context", ""),
                        "output": item.get("answer", item.get("response", ""))
                    }
                    data.append(alpaca_item)
    else:
        raise ValueError(f"不支持的文件格式: {input_file}")
    
    # 保存为JSONL格式
    print(f"正在保存 {len(data)} 条数据到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("转换完成！")
    return len(data)

def split_data(input_file, train_ratio=0.9):
    """
    将数据集拆分为训练集和验证集
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line)
    
    # 数据集拆分
    import random
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 保存拆分后的数据集
    base_name = os.path.splitext(input_file)[0]
    train_file = f"{base_name}_train.json"
    val_file = f"{base_name}_val.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    print(f"数据集拆分完成！训练集: {len(train_data)}条, 验证集: {len(val_data)}条")
    print(f"训练集保存至: {train_file}")
    print(f"验证集保存至: {val_file}")

def main():
    parser = argparse.ArgumentParser(description="数据准备工具：将原始数据转换为LLaMA Factory支持的格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--split", action="store_true", help="是否拆分为训练集和验证集")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    
    args = parser.parse_args()
    
    # 如果未指定输出文件，则使用输入文件名 + _alpaca.json
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_alpaca.json"
    
    count = convert_to_alpaca_format(args.input, args.output)
    
    if args.split and count > 0:
        split_data(args.output, args.train_ratio)

if __name__ == "__main__":
    main() 