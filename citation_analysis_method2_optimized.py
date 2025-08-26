#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案2优化版：逐个topic分析引用关系（避免TPM超限）
"""

import pandas as pd
import json
import requests
import os
import re
from typing import Dict, List, Any
import time

class CitationAnalysisMethod2Optimized:
    def __init__(self):
        self.api_key = os.getenv('SF_KEY')
        self.api_ep = 'https://api.siliconflow.cn/v1/chat/completions'
        self.model = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
        
        if not self.api_key:
            print("警告：未找到SF_KEY环境变量，无法调用API")
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载JSON数据：{len(data)}条")
            return data
        except Exception as e:
            print(f"加载JSON文件失败：{e}")
            return []
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """加载CSV数据获取引文内容"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"成功加载CSV数据：{len(df)}行")
            return df
        except Exception as e:
            print(f"加载CSV文件失败：{e}")
            return None
    
    def get_citations_from_csv(self, df: pd.DataFrame, row_index: int) -> Dict[int, str]:
        """从CSV中获取指定行的引文内容"""
        if row_index >= len(df):
            return {}
        
        row = df.iloc[row_index]
        citations_dict = {}
        
        for i in range(1, 21):
            col_name = f'引文{i}'
            if col_name in row and pd.notna(row[col_name]):
                citations_dict[i] = str(row[col_name])
        
        return citations_dict
    
    def prepare_topic_analysis_prompt(self, topic: str, citation_numbers: List[int], citations_dict: Dict[int, str]) -> str:
        """为单个topic准备分析prompt（简化版）"""
        
        prompt = f"""分析这段文字中的引用是否合适：

文字：{topic}

使用的引用：{citation_numbers}

对应引文：
"""
        
        for num in citation_numbers:
            if num in citations_dict:
                # 限制引文长度避免token过多
                cite_text = citations_dict[num]
                if len(cite_text) > 150:
                    cite_text = cite_text[:150] + "..."
                prompt += f"{num}. {cite_text}\n"
            else:
                prompt += f"{num}. （未找到引文）\n"
        
        prompt += """
请简要分析：
1. 各引用是否支持文字观点？
2. 评分(1-5分)
3. 简要建议

请简洁回答。
"""
        return prompt
    
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """调用硅基流动API进行分析"""
        if not self.api_key:
            return {
                'success': False,
                'error': '缺少SF_KEY环境变量',
                'content': None
            }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2,
            'max_tokens': 600  # 进一步减少
        }
        
        try:
            print(f"    调用API...【超时30秒】")
            response = requests.post(self.api_ep, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                return {
                    'success': False,
                    'error': 'TPM超限',
                    'content': None,
                    'retry_after': 60
                }
            
            response.raise_for_status()
            result = response.json()
            
            return {
                'success': True,
                'error': None,
                'content': result['choices'][0]['message']['content']
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': '网络超时(30秒)',
                'content': None
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': '网络连接失败',
                'content': None
            }
        except requests.exceptions.HTTPError as e:
            return {
                'success': False,
                'error': f'HTTP错误: {e.response.status_code}',
                'content': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'未知错误: {str(e)}',
                'content': None
            }
    
    def analyze_topic_citation(self, topic_data: Dict, citations_dict: Dict[int, str]) -> Dict[str, Any]:
        """分析单个topic的引用质量"""
        topic = topic_data['topic']
        rank = topic_data['rank']
        citation_numbers = topic_data['citation']
        
        # 生成分析prompt
        analysis_prompt = self.prepare_topic_analysis_prompt(topic, citation_numbers, citations_dict)
        
        # 调用API分析
        api_result = self.call_api(analysis_prompt)
        
        result = {
            'rank': rank,
            'topic': topic,
            'citation_numbers': citation_numbers,
            'available_citations': [num for num in citation_numbers if num in citations_dict],
            'missing_citations': [num for num in citation_numbers if num not in citations_dict],
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': api_result['content'] if api_result['success'] else None
        }
        
        return result
    
    def batch_analyze(self, json_path: str, csv_path: str, num_topics: int = 10) -> List[Dict[str, Any]]:
        """批量分析指定数量的topic"""
        json_data = self.load_json_data(json_path)
        csv_df = self.load_csv_data(csv_path)
        
        if not json_data or csv_df is None:
            return []
        
        results = []
        success_count = 0
        failed_count = 0
        
        print(f"开始分析前{num_topics}个topic...")
        print("注意：每次调用间隔8秒避免TPM超限")
        
        for i, topic_data in enumerate(json_data[:num_topics]):
            print(f"\n=== 第{i+1}/{num_topics}个topic ===")
            print(f"Topic: {topic_data['topic'][:50]}...")
            
            # 获取对应的引文内容
            rank = topic_data['rank']
            csv_row_index = rank - 1 if rank > 0 else 0
            citations_dict = self.get_citations_from_csv(csv_df, csv_row_index)
            
            # 分析
            result = self.analyze_topic_citation(topic_data, citations_dict)
            
            if result['api_success']:
                print("    成功")
                success_count += 1
            else:
                print(f"    失败: {result['api_error']}")
                failed_count += 1
                
                # 如果是TPM限制，等待
                if 'TPM' in str(result['api_error']):
                    print("    等待60秒...")
                    time.sleep(60)
            
            results.append(result)
            
            # 保守的间隔时间
            if result['api_success']:
                time.sleep(8)  # 成功后等8秒
            else:
                time.sleep(3)   # 失败后等3秒
        
        print(f"\n=== 分析完成 ===")
        print(f"总共: {len(results)}个topic")
        print(f"成功: {success_count}, 失败: {failed_count}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存分析结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到：{output_path}")
        except Exception as e:
            print(f"保存结果失败：{e}")

def main():
    analyzer = CitationAnalysisMethod2Optimized()
    
    # 数据路径
    json_path = "local_data/citation_results.json"
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_method2_optimized_results.json"
    
    # 先测试前10个topic
    results = analyzer.batch_analyze(json_path, csv_path, num_topics=10)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n方案2优化版分析完成！共处理{len(results)}个topic")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析的topic示例：")
            for i, result in enumerate(success_results[:2]):
                print(f"{i+1}. 引用{result['citation_numbers']} - {result['topic'][:30]}...")
    else:
        print("分析失败，请检查数据文件和配置")

if __name__ == "__main__":
    main()