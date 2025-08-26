#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用百炼API的引用分析脚本
针对JSON中的topic片段进行引用关系分析
"""

import pandas as pd
import json
import requests
import os
import re
from typing import Dict, List, Any
import time
class BailianCitationAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('AL_KEY')
        self.api_ep = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
        self.model = 'qwen3-235b-a22b-instruct-2507'  # 百炼的模型名
        print(f"正在使用模型: {self.model}")
        
        if not self.api_key:
            print("警告：未找到AL_KEY环境变量，无法调用百炼API")

    def count_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 中文大约1.5字符=1token，英文约4字符=1token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return estimated_tokens
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载JSON数据：{len(data)}条topic")
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
    
    def prepare_analysis_prompt(self, topic: str, citation_numbers, citations_dict: Dict[int, str]) -> str:
        """准备分析prompt"""
        
        # 确保citation_numbers是列表
        if isinstance(citation_numbers, int):
            citation_numbers = [citation_numbers]
        elif not isinstance(citation_numbers, list):
            citation_numbers = list(citation_numbers) if citation_numbers else []
        
        prompt = f"""
你是专业的文献内容审查员，你的任务是严格审查【待分析文本】中的每一个信息点，并判断其是否得到了【对应的引文内容】的明确支持。

**审查标准：**
1.  **逐句分析**：将【待分析文本】视为一个独立的声明或句子，对其进行整体分析。
2.  **事实核对**：文本中的每一个具体信息、实体、数据或概念，都必须在引文中有明确、直接的文字对应。不允许任何形式的模糊推断或间接关联。
3.  **完整性校验**：如果文本中包含多个信息点（例如，列举了多种疾病），必须确保所有信息点都得到了引文的支持。只要有一个信息点在引文中找不到依据，就判定为不一致。

**输出格式：**
请严格按照以下JSON格式输出，不要添加任何额外的解释性文字。

```json
{{
  "is_consistent": boolean,
  "reason": "string"
}}
```

-   `is_consistent`:
    -   `true`: 当且仅当【待分析文本】中的**所有信息点**都能在【对应的引文内容】中找到**明确且直接**的支持时，为 `true`。
    -   `false`: 只要有**任何一个信息点**无法在引文中找到明确支持，或与引文内容相悖，即为 `false`。
-   `reason`:
    -   如果 `is_consistent` 为 `true`，此字段应为空字符串 `""`。
    -   如果 `is_consistent` 为 `false`，请在此字段中**详细说明不一致的原因**。必须明确指出是**哪个具体的信息点**在引文中找不到支持。

---

**审查示例：**

【待分析文本】
血液系统疾病：如白血病、淋巴瘤、再生障碍性贫血等，造血干细胞移植是治疗这些疾病的重要方法。

【对应的引文内容】
引文1：干细胞能够修复受损组织，治疗血液系统疾病...白血病患者常通过干细胞移植来替代病变的造血系统。
引文2：干细胞疗法可以治疗血液系统疾病...例如，干细胞疗法中的造血干细胞移植是治疗白血病的重要方法。

**输出示例：**
```json
{{
  "is_consistent": false,
  "reason": "【待分析文本】中提到的'淋巴瘤'和'再生障碍性贫血'在【对应的引文内容】中未得到支持。引文仅明确提到白血病作为血液系统疾病的例子。"
}}
```

---

**现在，请开始你的审查工作：**

【待分析文本】
{topic}

【文本中使用的引用编号】
{citation_numbers}

【对应的引文内容】
    
    def call_api(self, prompt: str) -> Dict[str, Any]:
        """调用百炼API"""
        if not self.api_key:
            return {
                'success': False,
                'error': '缺少AL_KEY环境变量',
                'content': None
            }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model,
            'input': {
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            },
            'parameters': {
                'temperature': 0.3,
                'max_tokens': 1000
            }
        }
        
        try:
            prompt_tokens = self.count_tokens(prompt)
            print(f"    调用百炼API... (请求Token: {prompt_tokens})")
            response = requests.post(self.api_ep, headers=headers, json=data, timeout=60)
            
            if response.status_code == 429:
                return {
                    'success': False,
                    'error': 'API调用频率超限',
                    'content': None,
                    'retry_after': 30
                }
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'API调用失败: {response.status_code} - {response.text[:200]}',
                    'content': None
                }
            
            result = response.json()
            
            if result.get('output') and result['output'].get('text'):
                content = result['output']['text']
                response_tokens = self.count_tokens(content)
                print(f"    响应Token: {response_tokens}")
                return {
                    'success': True,
                    'error': None,
                    'content': content
                }
            else:
                return {
                    'success': False,
                    'error': f'API返回格式异常: {result}',
                    'content': None
                }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': '网络超时(60秒)',
                'content': None
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': '网络连接失败',
                'content': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'未知错误: {str(e)}',
                'content': None
            }
    
    def analyze_topic(self, topic_data: Dict, citations_dict: Dict[int, str]) -> Dict[str, Any]:
        """分析单个topic的引用质量"""
        topic = topic_data['topic']
        rank = topic_data['rank']
        citation_numbers = topic_data['citation']
        
        # 确保citation_numbers是列表形式用于后续处理
        if isinstance(citation_numbers, int):
            citation_numbers_list = [citation_numbers]
        elif not isinstance(citation_numbers, list):
            citation_numbers_list = list(citation_numbers) if citation_numbers else []
        else:
            citation_numbers_list = citation_numbers
        
        # 如果没有任何引用，跳过分析
        if not citation_numbers_list:
            print("    跳过分析：topic中没有引用标记")
            return {
                'rank': rank,
                'topic': topic,
                'citation_numbers': citation_numbers,
                'citation_numbers_list': citation_numbers_list,
                'available_citations': [],
                'missing_citations': [],
                'api_success': True,  # 标记为成功但跳过
                'api_error': None,
                'analysis': {'is_consistent': True, 'reason': '跳过分析：topic中没有引用标记'},
                'skipped': True
            }
        
        # 生成分析prompt
        analysis_prompt = self.prepare_analysis_prompt(topic, citation_numbers, citations_dict)
        
        # 调用API分析
        api_result = self.call_api(analysis_prompt)
        
        analysis_json = {}
        if api_result['success']:
            try:
                # 提取json内容
                json_match = re.search(r'```json\s*(\{.*\})\s*```', api_result['content'], re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    analysis_json = json.loads(json_str)
                else:
                    # 如果没有找到代码块，尝试直接解析
                    analysis_json = json.loads(api_result['content'])
            except json.JSONDecodeError:
                api_result['success'] = False
                api_result['error'] = 'JSON解析失败'
                analysis_json = {'is_consistent': False, 'reason': f"返回内容不是有效的JSON格式: {api_result['content']}"}

        result = {
            'rank': rank,
            'topic': topic,
            'citation_numbers': citation_numbers,  # 保持原始格式
            'citation_numbers_list': citation_numbers_list,  # 列表格式用于处理
            'available_citations': [num for num in citation_numbers_list if num in citations_dict],
            'missing_citations': [num for num in citation_numbers_list if num not in citations_dict],
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': analysis_json,
            'skipped': False
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
        print("使用百炼API，调用间隔3秒")
        
        for i, topic_data in enumerate(json_data[:num_topics]):
            print(f"\n=== 第{i+1}/{num_topics}个topic ====")
            print(f"Rank: {topic_data['rank']}, Citations: {topic_data['citation']}")
            print(f"Topic: {topic_data['topic'][:80]}...")
            
            # 获取对应的引文内容
            rank = topic_data['rank']
            csv_row_index = rank - 1 if rank > 0 else 0
            citations_dict = self.get_citations_from_csv(csv_df, csv_row_index)
            
            print(f"    找到{len(citations_dict)}个可用引文")
            
            # 分析
            result = self.analyze_topic(topic_data, citations_dict)
            
            if result['api_success']:
                analysis_result = result.get('analysis', {})
                is_consistent = analysis_result.get('is_consistent', False)
                reason = analysis_result.get('reason', '无分析结果')
                
                if is_consistent:
                    print("    分析成功: ✅ 一致")
                else:
                    print(f"    分析成功: ❌ 不一致")
                    if reason:
                       print(f"      原因: {reason}")

                success_count += 1
            else:
                print(f"    分析失败: {result['api_error']}")
                failed_count += 1
                
                # 如果是频率限制，等待
                if '频率' in str(result['api_error']) or '429' in str(result['api_error']):
                    print("    等待30秒...")
                    time.sleep(30)
            
            results.append(result)
            
            # 调用间隔
            if result['api_success']:
                time.sleep(3)  # 成功后等3秒
            else:
                time.sleep(1)   # 失败后等1秒
        
        print(f"\n=== 分析完成 ===")
        print(f"总共处理: {len(results)}个topic")
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
    analyzer = BailianCitationAnalyzer()
    
    # 数据路径
    json_path = "local_data/citation_results.json"
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_bailian_results.json"
    
    # 分析前10个topic
    results = analyzer.batch_analyze(json_path, csv_path, num_topics=10)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n百炼API分析完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:2]):
                print(f"\n{i+1}. Topic (Rank {result['rank']}):")
                print(f"   引用: {result['citation_numbers']}")
                print(f"   内容: {result['topic'][:80]}...")
                if result['analysis']:
                    print(f"   分析: {result['analysis'][:100]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

if __name__ == "__main__":
    main()