#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试两个引用分析脚本的数据加载功能（不调用API）
"""

import os
import sys
sys.path.append('.')

from citation_analysis_method1 import CitationAnalysisMethod1
from citation_analysis_method2 import CitationAnalysisMethod2

def test_method1():
    print("=== 测试方案1 ===")
    analyzer1 = CitationAnalysisMethod1()
    
    # 测试数据加载
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    df = analyzer1.load_data(csv_path)
    
    if df is not None:
        print(f"CSV数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 测试前3行的引用提取
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            answer = str(row['答案'])
            citations = analyzer1.extract_citations(answer)
            print(f"第{i+1}行提取到的引用: {citations}")
    else:
        print("CSV数据加载失败")

def test_method2():
    print("\n=== 测试方案2 ===")
    analyzer2 = CitationAnalysisMethod2()
    
    # 测试JSON数据加载
    json_path = "local_data/citation_results.json"
    json_data = analyzer2.load_json_data(json_path)
    
    if json_data:
        print(f"JSON数据条数: {len(json_data)}")
        
        # 显示前3条数据结构
        for i, item in enumerate(json_data[:3]):
            print(f"第{i+1}条: rank={item['rank']}, citations={item['citation']}")
            print(f"  topic预览: {item['topic'][:50]}...")
        
        # 测试CSV数据加载
        csv_path = "local_data/副本正文引文内容（纯净版）.csv"
        csv_df = analyzer2.load_csv_data(csv_path)
        
        if csv_df is not None:
            print(f"\nCSV数据形状: {csv_df.shape}")
            
            # 测试引文提取
            citations_dict = analyzer2.get_citations_from_csv(csv_df, 0)
            print(f"第1行可用引文数量: {len(citations_dict)}")
            print(f"引文编号: {list(citations_dict.keys())}")
    else:
        print("JSON数据加载失败")

def main():
    """主测试函数"""
    print("开始测试引用分析脚本的数据加载功能...")
    
    # 检查数据文件是否存在
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    json_path = "local_data/citation_results.json"
    
    if not os.path.exists(csv_path):
        print(f"错误：CSV文件不存在 - {csv_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"错误：JSON文件不存在 - {json_path}")
        return
    
    print("数据文件检查通过\n")
    
    test_method1()
    test_method2()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()