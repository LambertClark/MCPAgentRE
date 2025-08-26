#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案2：单个topic+引文分析脚本
使用JSON文件中的每个topic片段和对应引文来逐一分析引用关系
"""

import pandas as pd
import json
import requests
import os
import re
from typing import Dict, List, Any
import time

class CitationAnalysisMethod2:
    def __init__(self):
        self.api_key = os.getenv('SF_KEY')
        self.api_ep = os.getenv('SF_EP', 'https://api.siliconflow.cn/v1/chat/completions')
        self.model = os.getenv('SF_MODEL', 'Qwen/Qwen3-235B-A22B-Instruct-2507')
        
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
        """为单个topic准备分析prompt"""
        
        prompt = f"""请分析以下文本片段中的引用是否恰当：

文本片段：{topic}

该片段中引用的条目编号：{citation_numbers}

对应的引文内容：
"""
        
        for num in citation_numbers:
            if num in citations_dict:
                prompt += f"引文{num}：{citations_dict[num]}\n"
            else:
                prompt += f"引文{num}：（未找到内容）\n"
        
        prompt += """
请分析：
1. 每个引用是否与其对应的引文内容相关？
2. 引文内容是否支持该片段的观点或陈述？
3. 引用的位置是否合适？
4. 是否存在引用过多或不足的问题？
5. 给出该片段的引用质量评分(1-5分)和具体建议

请给出简洁但详细的分析结果。
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
            'max_tokens': 2000
        }
        
        try:
            print(f"  调用API...【超时时间30秒】")
            response = requests.post(self.api_ep, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                return {
                    'success': False,
                    'error': 'Token Per Minute超限',
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
    
    def group_topics_by_source(self, json_data: List[Dict]) -> Dict[int, List[Dict]]:
        """按rank分组topics（假设rank对应CSV的行号）"""
        grouped = {}
        for item in json_data:
            rank = item['rank']
            if rank not in grouped:
                grouped[rank] = []
            grouped[rank].append(item)
        return grouped
    
    def batch_analyze(self, json_path: str, csv_path: str, num_samples: int = 50) -> List[Dict[str, Any]]:
        """批量分析前N条数据"""
        json_data = self.load_json_data(json_path)
        csv_df = self.load_csv_data(csv_path)
        
        if not json_data or csv_df is None:
            return []
        
        # 按rank分组
        grouped_data = self.group_topics_by_source(json_data)
        
        results = []
        processed_count = 0
        total_topics_processed = 0
        total_topics_success = 0
        total_topics_failed = 0
        
        print(f"开始分析前{num_samples}条数据的topic片段...")
        print(f"注意：发生错误时会立即显示并继续下一条")
        
        for rank in sorted(grouped_data.keys()):
            if processed_count >= num_samples:
                break
                
            # CSV行号从0开始，rank可能从1开始，需要调整
            csv_row_index = rank - 1 if rank > 0 else 0
            
            # 获取对应的引文内容
            citations_dict = self.get_citations_from_csv(csv_df, csv_row_index)
            
            print(f"\n=== 分析第{rank}条数据（包含{len(grouped_data[rank])}个topic） ===")
            
            # 分析该rank下的所有topics
            source_results = []
            source_success = 0
            source_failed = 0
            
            for i, topic_data in enumerate(grouped_data[rank]):
                print(f"  Topic {i+1}/{len(grouped_data[rank])}: ", end="")
                result = self.analyze_topic_citation(topic_data, citations_dict)
                
                if result['api_success']:
                    print("✅ 成功")
                    source_success += 1
                    total_topics_success += 1
                else:
                    print(f"❌ 失败: {result['api_error']}")
                    source_failed += 1
                    total_topics_failed += 1
                    
                    # 如果是TPM限制，等待
                    if 'Token Per Minute' in str(result['api_error']):
                        print("    等径60秒后重试...")
                        time.sleep(60)
                
                source_results.append(result)
                total_topics_processed += 1
                
                # 正常间隔
                if result['api_success']:
                    time.sleep(1.5)
            
            print(f"  第{rank}条数据完成: 成功{source_success}, 失败{source_failed}")
            
            results.append({
                'source_rank': rank,
                'csv_row_index': csv_row_index,
                'topics_analysis': source_results,
                'total_topics': len(source_results),
                'success_topics': source_success,
                'failed_topics': source_failed
            })
            
            processed_count += 1
        
        print(f"\n=== 批量分析完成 ===")
        print(f"处理{processed_count}条数据，共{total_topics_processed}个topic")
        print(f"成功: {total_topics_success}个topic, 失败: {total_topics_failed}个topic")
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
    analyzer = CitationAnalysisMethod2()
    
    # 数据路径
    json_path = "local_data/citation_results.json"
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_method2_results.json"
    
    # 分析前50条数据
    results = analyzer.batch_analyze(json_path, csv_path, num_samples=50)
    
    if results:
        analyzer.save_results(results, output_path)
        
        # 统计信息
        total_topics = sum(r['total_topics'] for r in results)
        print(f"\n方案2分析完成！")
        print(f"共分析{len(results)}条源数据")
        print(f"总计{total_topics}个topic片段")
        
        # 显示前几条的概要
        print("\n分析概要：")
        for i, result in enumerate(results[:3]):
            print(f"第{result['source_rank']}条数据：包含{result['total_topics']}个topic片段")
    else:
        print("分析失败，请检查数据文件和配置")

if __name__ == "__main__":
    main()