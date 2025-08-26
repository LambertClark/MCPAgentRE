#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的引用分析测试脚本
只处理JSON中的topic片段，避免TPM超限
"""

import json
import pandas as pd
import requests
import os
import time
from typing import Dict, List, Any

class SimpleCitationAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('SF_KEY')
        self.api_ep = 'https://api.siliconflow.cn/v1/chat/completions'
        self.model = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
        
    def load_data(self):
        """加载数据"""
        # 加载JSON
        with open('local_data/citation_results.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 加载CSV
        csv_df = pd.read_csv('local_data/副本正文引文内容（纯净版）.csv', encoding='utf-8')
        
        return json_data, csv_df
    
    def get_citations_from_csv(self, df: pd.DataFrame, row_index: int) -> Dict[int, str]:
        """从CSV中获取引文"""
        if row_index >= len(df):
            return {}
        
        row = df.iloc[row_index]
        citations_dict = {}
        
        for i in range(1, 21):
            col_name = f'引文{i}'
            if col_name in row and pd.notna(row[col_name]):
                citations_dict[i] = str(row[col_name])
        
        return citations_dict
    
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """调用API"""
        if not self.api_key:
            return {'success': False, 'error': '缺少SF_KEY', 'content': None}
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens': 500
        }
        
        try:
            response = requests.post(self.api_ep, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                return {'success': False, 'error': 'TPM超限', 'content': None}
            
            response.raise_for_status()
            result = response.json()
            return {'success': True, 'error': None, 'content': result['choices'][0]['message']['content']}
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': '超时', 'content': None}
        except Exception as e:
            return {'success': False, 'error': str(e), 'content': None}
    
    def analyze_topic(self, topic: str, citation_nums: List[int], citations_dict: Dict[int, str]) -> Dict[str, Any]:
        """分析单个topic"""
        # 构建简短的prompt
        prompt = f"""分析这段文字的引用是否合适：

文字：{topic}
使用的引用编号：{citation_nums}

对应引文：
"""
        for num in citation_nums:
            if num in citations_dict:
                cite_text = citations_dict[num][:100] + "..." if len(citations_dict[num]) > 100 else citations_dict[num]
                prompt += f"{num}. {cite_text}\n"
        
        prompt += "\n请简要分析引用是否合适，给出1-5分评分。"
        
        api_result = self.call_api(prompt)
        
        return {
            'topic': topic,
            'citations': citation_nums,
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': api_result['content']
        }
    
    def test_small_batch(self, num_topics: int = 5):
        """测试少量数据"""
        json_data, csv_df = self.load_data()
        
        print(f"测试前{num_topics}个topic...")
        results = []
        
        for i, topic_data in enumerate(json_data[:num_topics]):
            print(f"处理第{i+1}个topic...")
            
            # 获取引文
            rank = topic_data['rank']
            csv_row_index = rank - 1 if rank > 0 else 0
            citations_dict = self.get_citations_from_csv(csv_df, csv_row_index)
            
            # 分析
            result = self.analyze_topic(
                topic_data['topic'], 
                topic_data['citation'], 
                citations_dict
            )
            
            if result['api_success']:
                print(f"  成功")
            else:
                print(f"  失败: {result['api_error']}")
                if 'TPM' in str(result['api_error']):
                    print("  等待30秒...")
                    time.sleep(30)
            
            results.append(result)
            time.sleep(5)  # 长间隔避免TPM
        
        # 保存结果
        with open('local_data/simple_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"完成！处理了{len(results)}个topic")
        success_count = sum(1 for r in results if r['api_success'])
        print(f"成功: {success_count}, 失败: {len(results) - success_count}")

def main():
    analyzer = SimpleCitationAnalyzer()
    analyzer.test_small_batch(num_topics=3)  # 先测试3个

if __name__ == "__main__":
    main()