"""
方案1：完整prompt+引文分析脚本
使用完整的答案内容和对应引文来分析引用关系的准确性
"""

import pandas as pd
import requests
import json
import os
import re
from typing import Dict, List, Any
import time

class CitationAnalysisMethod1:
    def __init__(self):
        self.api_key = os.getenv('SF_KEY')
        self.api_ep = os.getenv('SF_EP', 'https://api.siliconflow.cn/v1/chat/completions')
        self.model = os.getenv('SF_MODEL', 'Qwen/Qwen3-235B-A22B-Instruct-2507')
        
        if not self.api_key:
            print("警告：未找到SF_KEY环境变量，无法调用API")
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"成功加载CSV数据：{len(df)}行")
            return df
        except Exception as e:
            print(f"加载CSV文件失败：{e}")
            return None
    
    def extract_citations(self, text: str) -> List[int]:
        """从文本中提取引用标记"""
        citations = []
        pattern = r'\[citation:(\d+)\]'
        matches = re.findall(pattern, text)
        for match in matches:
            citations.append(int(match))
        return sorted(list(set(citations)))
    
    def prepare_analysis_prompt(self, question: str, answer: str, citations_dict: Dict[int, str]) -> str:
        """准备分析prompt"""
        used_citations = self.extract_citations(answer)
        
        prompt = f"""请分析以下问答内容中引用与引文的匹配关系：  
  
问题：{question}  
  
回答：{answer}  
  
可用引文：  
"""
        for i, citation in citations_dict.items():
            if pd.notna(citation) and citation.strip():
                prompt += f"引文{i}：{citation}\n"
        prompt += f"""  
在回答中使用的引用标记：{used_citations}  
  
请分析：  每个引用标记[citation:X]是否存在错引用、漏引用或模型幻觉？
  
请给出分析结果。当引用有错误时：引用错误的原文以及错误原因分析；当引用正确时：只需要表示你分析了这一处引用。
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
            'max_tokens': 4000
        }
        
        try:
            print(f"正在调用API...【超时时间30秒】")
            response = requests.post(self.api_ep, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                return {
                    'success': False,
                    'error': 'Token Per Minute超限，请等待60秒后重试',
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
                'error': f'HTTP错误: {e.response.status_code} - {e.response.text[:200]}',
                'content': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'未知错误: {str(e)}',
                'content': None
            }
    
    def analyze_citation_quality(self, row: pd.Series) -> Dict[str, Any]:
        """分析单行数据的引用质量"""
        question = str(row['模型prompt'])
        answer = str(row['答案'])
        
        # 构建引文字典
        citations_dict = {}
        for i in range(1, 21):
            col_name = f'引文{i}'
            if col_name in row and pd.notna(row[col_name]):
                citations_dict[i] = str(row[col_name])
        
        # 生成分析prompt
        analysis_prompt = self.prepare_analysis_prompt(question, answer, citations_dict)
        
        # 调用API分析
        api_result = self.call_api(analysis_prompt)
        
        result = {
            'question': question,
            'answer_preview': answer[:100] + '...' if len(answer) > 100 else answer,
            'citations_used': self.extract_citations(answer),
            'citations_available': list(citations_dict.keys()),
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': api_result['content'] if api_result['success'] else None
        }
        
        return result
    
    def batch_analyze(self, csv_path: str, num_samples: int = 50) -> List[Dict[str, Any]]:
        """批量分析前N条数据"""
        df = self.load_data(csv_path)
        if df is None:
            return []
        
        # 取前N条数据
        sample_df = df.head(num_samples)
        results = []
        failed_count = 0
        success_count = 0
        
        print(f"开始分析前{num_samples}条数据...")
        print(f"注意：发生错误时会立即显示并继续下一条")
        
        for idx, row in sample_df.iterrows():
            print(f"\n=== 正在分析第{idx + 1}/{num_samples}条 ===")
            
            result = self.analyze_citation_quality(row)
            
            if result['api_success']:
                print(f"第{idx + 1}条分析成功")
                success_count += 1
            else:
                print(f"第{idx + 1}条分析失败: {result['api_error']}")
                failed_count += 1
                
                # 如果是TPM限制，等待更长时间
                if 'Token Per Minute' in str(result['api_error']):
                    print("等待60秒后重试...")
                    time.sleep(60)
            
            results.append({
                'index': idx + 1,
                **result
            })
            
            # 正常的API调用间隔
            if result['api_success']:
                time.sleep(2)  # 防止TPM超限
        
        print(f"\n=== 批量分析完成 ===")
        print(f"成功: {success_count}条, 失败: {failed_count}条")
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
    analyzer = CitationAnalysisMethod1()
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_method1_results.json"
    
    # 分析前50条数据
    results = analyzer.batch_analyze(csv_path, num_samples=50)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n方案1分析完成！共分析{len(results)}条数据")
        
        # 输出简要统计
        print("\n分析统计：")
        for i, result in enumerate(results[:3]):  # 显示前3条的概要
            print(f"第{result['index']}条 - 使用引用: {result['citations_used']}")
    else:
        print("分析失败，请检查数据文件和配置")

if __name__ == "__main__":
    main()