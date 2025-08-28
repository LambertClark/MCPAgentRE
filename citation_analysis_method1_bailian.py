#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案1百炼版：使用完整答案内容+百炼API分析引用关系
"""

import pandas as pd
import json
import requests
import aiohttp
import asyncio
import os
import re
from typing import Dict, List, Any
import time

class Method1BailianAnalyzer:
    def __init__(self, concurrent_limit: int = 50):
        self.api_key = os.getenv('AL_KEY')
        self.api_ep = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
        self.model = 'qwen-plus-latest'  # 百炼的模型名
        self.concurrent_limit = concurrent_limit
        print(f"正在使用模型: {self.model}")
        print(f"并发限制: {self.concurrent_limit}条")
        
        if not self.api_key:
            print("警告：未找到AL_KEY环境变量，无法调用百炼API")
    
    def count_chars(self, text: str) -> int:
        """简单的字符计数估算token"""
        # 中文大约1.5字符=1token，英文约4字符=1token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return estimated_tokens
    
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
        """准备分析prompt（完整版本，不截断）"""
        used_citations = self.extract_citations(answer)
        
        prompt_start = f"""请分析以下问答内容中引用与引文的匹配关系：

【完整答案内容（包含思考过程和回答内容）】
{answer}

【答案中使用的引用标记】
{used_citations}

【可用引文内容】
"""
        
        citations_text = ""
        # 显示所有被使用的引文，完整内容
        for citation_num in used_citations:
            if citation_num in citations_dict:
                cite_text = citations_dict[citation_num]
                citations_text += f"引文{citation_num}：{cite_text}\n\n"
            else:
                citations_text += f"引文{citation_num}：（未找到对应内容）\n\n"
        
        analysis_requirements = '''【分析要求】
你是一个严谨的文本分析专家。

🚨 核心规则：只输出包含[citation:x]标记的句子！没有引用标记的句子绝对不能出现在JSON输出中！

你的任务：分析完整答案内容（包含思考过程和回答内容）中所有包含引用标记[citation:x]的句子，完全跳过没有引用标记的句子。

**重要规则（必须严格遵守）**：
- **引用边界识别**：引用标记[citation:x]的作用范围严格以句号（。）、换行符、段落分隔符为边界，绝对不能跨越这些边界
- **逐句独立分析**：必须将文本按句号（。）拆分为独立句子，每个句子单独判断是否包含引用标记
- **无引用标记=跳过**：如果一个句子内部没有[citation:x]标记，无论其前后句子是否有引用，都必须完全跳过该句子
- **严禁跨句关联**：绝对不能将前一句子的引用标记应用到后续没有引用标记的句子上

请遵循以下步骤和规则：

1.  **逐句拆分**：将【完整答案内容】拆分为独立的观点或句子，在思考过程和回答内容两部分中都要查找。
2.  **逐句分析**：对于每一个独立的观点或句子：
    a. **首先检查**：该句子是否包含引用标记 `[citation:x]`。如果没有任何引用标记，立即跳过该句子，不进行分析。
    b. **精确划定引用范围**：引用标记仅对引用标记列中位置**之前**的内容起作用。标点号后或新句子开始后的内容不在引用范围内。
    c. 如果有引用标记，在对应的 `引文x` 中查找支持性证据。
    d. **严格判断**：
        - **完全一致 (Consistent)**：当且仅当句子中的**所有信息点**（包括核心事实、数据、限定词、描述性词语）都能在引文中找到**明确、直接**的表述时，才能判定为“一致”。
        - **不一致 (Inconsistent)**：如果句子中包含任何引文中**没有明确提及**的信息（例如，具体的数字、夸大的描述、不同的概念），或者与引文内容相悖，则判定为“不一致”。这被视为一种“模型幻觉”。
3.  **输出格式**：
    - 请不要输出整体的分析报告，也不要在分析中对“整体”作任何分析，就像用放大镜挑刺一样。
    - 你的输出必须是一个JSON格式的列表 `[]`。
    - 列表中的每个对象代表对一个观点/句子的分析，格式如下：
      ```json
      {
        "topic": "被分析的句子或观点",
        "citation_numbers": [引用的编号列表],
        "consistency": "一致" 或 "不一致",
        "reason": "详细的判断理由。如果一致，请说明证据在哪。如果不一致，请明确指出是哪个信息点在引文中无法找到或存在矛盾。"
      }
4. **空引用情况（绝对重要）**：
    - **完全跳过规则**：如果一个句子没有[citation:x]标记，绝对不能出现在JSON输出中，连提及都不行
    - **错误做法1**（绝对禁止）：
      ```json
      {
        "topic": "火焰山的最佳游览时间是清晨7:00-9:00...",
        "citation_numbers": [],
        "consistency": "一致",
        "reason": "该句无引用标记，根据规则应跳过..."
      }
      ```
    - **错误做法2**（绝对禁止）：
      ```json
      {
        "topic": "优化操作设置是提升游戏体验的关键。",
        "citation_numbers": [],
        "consistency": "不一致", 
        "reason": "该句无引用标记..."
      }
      ```
    - **正确做法**：这些句子在输出JSON中完全不存在，就像它们从未出现过一样

**关键示例（引用边界识别）**：
错误的文本："根据[citation:6][citation:7][citation:8]，吐鲁番的最佳旅游时间是4-5月和9-10月，气温适宜。火焰山的最佳游览时间是清晨7:00-9:00或傍晚18:00-20:00，避开正午高温。"

**正确分析方法**：
1. 第一句："根据[citation:6][citation:7][citation:8]，吐鲁番的最佳旅游时间是4-5月和9-10月，气温适宜。" → **包含引用标记，需要分析**
2. 第二句："火焰山的最佳游览时间是清晨7:00-9:00或傍晚18:00-20:00，避开正午高温。" → **没有引用标记，必须跳过**

**错误做法**（绝对禁止）：将第二句也关联到[citation:6][citation:7][citation:8]，这是错误的跨句关联。

**示例输出**：
      ```json
      [
        {
          "topic": "根据[citation:6][citation:7][citation:8]，吐鲁番的最佳旅游时间是4-5月和9-10月，气温适宜",
          "citation_numbers": [6, 7, 8],
          "consistency": "一致",
          "reason": "引文6、7、8均支持该时间段和气温描述"
        }
      ]
      ```

注意：上例中"火焰山..."句子完全不出现在输出中，因为它没有引用标记。
'''
        
        return prompt_start + citations_text + analysis_requirements
    
    async def call_api_async(self, session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """异步调用百炼API，支持重试机制"""
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
                'temperature': 0.2,
                'max_tokens': 15000
            }
        }
        
        prompt_tokens = self.count_chars(prompt)
        
        # 重试循环
        last_error = None
        for attempt in range(max_retries):
            try:
                # 延长超时时间到180秒
                timeout = aiohttp.ClientTimeout(total=180)
                
                async with session.post(self.api_ep, headers=headers, json=data, timeout=timeout) as response:
                    # 检查HTTP状态码
                    if response.status == 200:
                        # 成功响应
                        result = await response.json()
                        
                        if result.get('output') and result['output'].get('text'):
                            content = result['output']['text']
                            response_tokens = self.count_chars(content)
                            return {
                                'success': True,
                                'error': None,
                                'content': content
                            }
                        else:
                            last_error = f'API返回格式异常: {result}'
                            # 格式异常通常不需要重试
                            break
                            
                    elif response.status == 429:
                        # 频率限制，需要等待后重试
                        last_error = 'API调用频率超限'
                        if attempt < max_retries - 1:  # 不是最后一次尝试
                            await asyncio.sleep(30)
                            continue
                        
                    elif response.status >= 500:
                        # 服务器错误，可以重试
                        response_text = await response.text()
                        last_error = f'服务器错误: {response.status} - {response_text[:200]}' 
                        if attempt < max_retries - 1:
                            await asyncio.sleep(10)
                            continue
                            
                    elif response.status in [401, 403]:
                        # 认证错误，不需要重试
                        response_text = await response.text()
                        return {
                            'success': False,
                            'error': f'认证错误: {response.status} - {response_text[:200]}',
                            'content': None
                        }
                        
                    else:
                        # 其他客户端错误，通常不需要重试
                        response_text = await response.text()
                        last_error = f'客户端错误: {response.status} - {response_text[:200]}'
                        break
            
            except asyncio.TimeoutError:
                last_error = '网络超时(180秒)'
                if attempt < max_retries - 1:
                    await asyncio.sleep(15)
                    continue
                    
            except aiohttp.ClientConnectionError:
                last_error = '网络连接失败'
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                    continue
                    
            except Exception as e:
                last_error = f'未知错误: {str(e)}'
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
        
        # 所有重试都失败了
        return {
            'success': False,
            'error': last_error or 'API调用失败',
            'content': None
        }
    
    def call_api(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """调用百炼API，支持重试机制"""
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
                'temperature': 0.2,
                'max_tokens': 15000
            }
        }
        
        prompt_tokens = self.count_chars(prompt)
        print(f"    调用百炼API... (估算请求Token: {prompt_tokens})")
        
        # 重试循环
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"    第{attempt + 1}次重试...")
                
                # 延长超时时间到180秒
                response = requests.post(self.api_ep, headers=headers, json=data, timeout=180)
                
                # 检查HTTP状态码
                if response.status_code == 200:
                    # 成功响应
                    result = response.json()
                    print(f"    API调用成功 (第{attempt + 1}次尝试)")
                    
                    if result.get('output') and result['output'].get('text'):
                        content = result['output']['text']
                        response_tokens = self.count_chars(content)
                        print(f"    估算响应Token: {response_tokens}")
                        return {
                            'success': True,
                            'error': None,
                            'content': content
                        }
                    else:
                        print(f"    API返回格式异常: {result}")
                        last_error = f'API返回格式异常: {result}'
                        # 格式异常通常不需要重试
                        break
                        
                elif response.status_code == 429:
                    # 频率限制，需要等待后重试
                    print(f"    API调用频率超限 (第{attempt + 1}次尝试)，等待30秒后重试")
                    last_error = 'API调用频率超限'
                    if attempt < max_retries - 1:  # 不是最后一次尝试
                        time.sleep(30)
                        continue
                    
                elif response.status_code >= 500:
                    # 服务器错误，可以重试
                    print(f"    服务器错误 {response.status_code} (第{attempt + 1}次尝试)，等待10秒后重试")
                    last_error = f'服务器错误: {response.status_code} - {response.text[:200]}'
                    if attempt < max_retries - 1:
                        time.sleep(10)
                        continue
                        
                elif response.status_code in [401, 403]:
                    # 认证错误，不需要重试
                    print(f"    认证错误 {response.status_code}，请检查API密钥")
                    return {
                        'success': False,
                        'error': f'认证错误: {response.status_code} - {response.text[:200]}',
                        'content': None
                    }
                    
                else:
                    # 其他客户端错误，通常不需要重试
                    print(f"    客户端错误 {response.status_code}")
                    last_error = f'客户端错误: {response.status_code} - {response.text[:200]}'
                    break
            
            except requests.exceptions.Timeout:
                print(f"    网络超时 (第{attempt + 1}次尝试，180秒)")
                last_error = '网络超时(180秒)'
                if attempt < max_retries - 1:
                    print(f"    等待15秒后进行第{attempt + 2}次尝试")
                    time.sleep(15)
                    continue
                    
            except requests.exceptions.ConnectionError:
                print(f"    网络连接失败 (第{attempt + 1}次尝试)")
                last_error = '网络连接失败'
                if attempt < max_retries - 1:
                    print(f"    等待10秒后进行第{attempt + 2}次尝试")
                    time.sleep(10)
                    continue
                    
            except Exception as e:
                print(f"    未知错误 (第{attempt + 1}次尝试): {str(e)}")
                last_error = f'未知错误: {str(e)}'
                if attempt < max_retries - 1:
                    print(f"    等待5秒后进行第{attempt + 2}次尝试")
                    time.sleep(5)
                    continue
        
        # 所有重试都失败了
        print(f"    API调用最终失败，已重试{max_retries}次")
        return {
            'success': False,
            'error': last_error or 'API调用失败',
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
        
        # 提取使用的引用
        citations_used = self.extract_citations(answer)
        
        print(f"    问题长度: {len(question)}字符")
        print(f"    答案长度: {len(answer)}字符")
        print(f"    可用引文: {len(citations_dict)}个")
        print(f"    实际引用: {citations_used}")
        
        # 如果没有任何引用，跳过分析
        if not citations_used:
            print("    跳过分析：答案中没有任何引用标记")
            return {
                'question': question,
                'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer,
                'citations_used': citations_used,
                'citations_available': list(citations_dict.keys()),
                'api_success': True,  # 标记为成功但跳过
                'api_error': None,
                'analysis': '跳过分析：答案中没有引用标记',
                'skipped': True
            }
        
        # 生成分析prompt
        analysis_prompt = self.prepare_analysis_prompt(question, answer, citations_dict)
        
        # 调用API分析
        api_result = self.call_api(analysis_prompt)
        
        analysis_content = None
        if api_result['success']:
            try:
                # 尝试解析API返回的JSON字符串
                analysis_content = json.loads(api_result['content'])
            except json.JSONDecodeError:
                # 如果解析失败，说明返回的不是合法的JSON，作为原始文本处理
                analysis_content = api_result['content']

        result = {
            'question': question,
            'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer,
            'citations_used': citations_used,
            'citations_available': list(citations_dict.keys()),
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': analysis_content,
            'skipped': False
        }
        
        return result
    
    async def analyze_citation_quality_async(self, session: aiohttp.ClientSession, row: pd.Series, rank: int) -> Dict[str, Any]:
        """异步分析单行数据的引用质量"""
        question = str(row['模型prompt'])
        answer = str(row['答案'])
        
        # 构建引文字典
        citations_dict = {}
        for i in range(1, 21):
            col_name = f'引文{i}'
            if col_name in row and pd.notna(row[col_name]):
                citations_dict[i] = str(row[col_name])
        
        # 提取使用的引用
        citations_used = self.extract_citations(answer)
        
        # 如果没有任何引用，跳过分析
        if not citations_used:
            return {
                'rank': rank,
                'question': question,
                'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer,
                'citations_used': citations_used,
                'citations_available': list(citations_dict.keys()),
                'api_success': True,  # 标记为成功但跳过
                'api_error': None,
                'analysis': '跳过分析：答案中没有引用标记',
                'skipped': True
            }
        
        # 生成分析prompt
        analysis_prompt = self.prepare_analysis_prompt(question, answer, citations_dict)
        
        # 调用异步API分析
        api_result = await self.call_api_async(session, analysis_prompt)
        
        analysis_content = None
        if api_result['success']:
            try:
                # 尝试解析API返回的JSON字符串
                analysis_content = json.loads(api_result['content'])
            except json.JSONDecodeError:
                # 如果解析失败，说明返回的不是合法的JSON，作为原始文本处理
                analysis_content = api_result['content']

        result = {
            'rank': rank,
            'question': question,
            'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer,
            'citations_used': citations_used,
            'citations_available': list(citations_dict.keys()),
            'api_success': api_result['success'],
            'api_error': api_result['error'],
            'analysis': analysis_content,
            'skipped': False
        }
        
        return result
    
    async def batch_analyze_concurrent(self, csv_path: str, num_samples: int = None, specific_rank: int = None, start_from: int = None) -> List[Dict[str, Any]]:
        """异步并发批量分析"""
        df = self.load_data(csv_path)
        if df is None:
            return []
        
        # 确定要处理的数据
        if specific_rank is not None:
            # 处理特定rank的单条数据
            if specific_rank <= 0 or specific_rank > len(df):
                print(f"错误：指定的rank {specific_rank} 超出数据范围 (1-{len(df)})")
                return []
            sample_df = df.iloc[[specific_rank - 1]]  # rank是1-based，转为0-based索引
            total_count = 1
            print(f"开始分析第{specific_rank}条数据...")
        elif start_from is not None:
            # 从指定位置开始处理指定数量的数据
            if start_from <= 0 or start_from > len(df):
                print(f"错误：起始位置 {start_from} 超出数据范围 (1-{len(df)})")
                return []
            start_idx = start_from - 1  # start_from是1-based，转为0-based索引
            if num_samples is None:
                # 从起始位置到结尾
                sample_df = df.iloc[start_idx:]
                total_count = len(df) - start_idx
                print(f"开始分析从第{start_from}条开始的所有数据（共{total_count}条）...")
            else:
                # 从起始位置开始指定数量
                end_idx = min(start_idx + num_samples, len(df))
                sample_df = df.iloc[start_idx:end_idx]
                total_count = len(sample_df)
                print(f"开始分析从第{start_from}条开始的{total_count}条数据...")
        elif num_samples is None:
            # 处理所有数据
            sample_df = df
            total_count = len(df)
            print(f"开始并发分析所有{total_count}条完整问答数据...")
        else:
            # 处理前num_samples条数据
            sample_df = df.head(num_samples)
            total_count = num_samples
            print(f"开始并发分析前{num_samples}条完整问答数据...")
        
        print(f"使用百炼API，并发限制: {self.concurrent_limit}条")
        
        # 创建信号量来控制并发数量
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def process_with_semaphore(session, row, rank):
            async with semaphore:
                result = await self.analyze_citation_quality_async(session, row, rank + 1)
                return result
        
        # 创建HTTP会话
        connector = aiohttp.TCPConnector(limit=100)  # 连接池大小
        async with aiohttp.ClientSession(connector=connector) as session:
            # 创建任务列表
            tasks = []
            for idx, row in sample_df.iterrows():
                task = process_with_semaphore(session, row, idx)
                tasks.append(task)
            
            # 执行所有任务并显示进度
            print(f"开始处理{len(tasks)}个任务...")
            start_time = time.time()
            
            completed_tasks = []
            for task in asyncio.as_completed(tasks):
                result = await task
                completed_tasks.append(result)
                
                # 显示进度
                progress = len(completed_tasks)
                elapsed = time.time() - start_time
                avg_time = elapsed / progress if progress > 0 else 0
                eta = avg_time * (total_count - progress)
                
                status = "✓" if result['api_success'] else "✗"
                print(f"[{progress}/{total_count}] {status} 第{result['rank']}条 "
                      f"(用时: {elapsed:.1f}s, ETA: {eta:.1f}s)")
        
        # 按rank排序结果
        completed_tasks.sort(key=lambda x: x['rank'])
        
        # 统计结果
        success_count = sum(1 for r in completed_tasks if r['api_success'])
        failed_count = len(completed_tasks) - success_count
        
        total_time = time.time() - start_time
        print(f"\n=== 并发分析完成 ===")
        print(f"总用时: {total_time:.1f}秒")
        print(f"平均每条: {total_time/len(completed_tasks):.2f}秒")
        print(f"成功: {success_count}条, 失败: {failed_count}条")
        
        return completed_tasks
    
    def batch_analyze(self, csv_path: str, num_samples: int = 10, specific_rank: int = None, start_from: int = None) -> List[Dict[str, Any]]:
        """批量分析数据，num_samples=None时处理所有数据"""
        df = self.load_data(csv_path)
        if df is None:
            return []
        
        # 确定要处理的数据
        if specific_rank is not None:
            # 处理特定rank的单条数据
            if specific_rank <= 0 or specific_rank > len(df):
                print(f"错误：指定的rank {specific_rank} 超出数据范围 (1-{len(df)})")
                return []
            sample_df = df.iloc[[specific_rank - 1]]  # rank是1-based，转为0-based索引
            total_count = 1
            print(f"开始分析第{specific_rank}条数据...")
        elif start_from is not None:
            # 从指定位置开始处理指定数量的数据
            if start_from <= 0 or start_from > len(df):
                print(f"错误：起始位置 {start_from} 超出数据范围 (1-{len(df)})")
                return []
            start_idx = start_from - 1  # start_from是1-based，转为0-based索引
            if num_samples is None:
                # 从起始位置到结尾
                sample_df = df.iloc[start_idx:]
                total_count = len(df) - start_idx
                print(f"开始分析从第{start_from}条开始的所有数据（共{total_count}条）...")
            else:
                # 从起始位置开始指定数量
                end_idx = min(start_idx + num_samples, len(df))
                sample_df = df.iloc[start_idx:end_idx]
                total_count = len(sample_df)
                print(f"开始分析从第{start_from}条开始的{total_count}条数据...")
        elif num_samples is None:
            # 处理所有数据
            sample_df = df
            total_count = len(df)
            print(f"开始分析所有{total_count}条完整问答数据...")
        else:
            # 处理前num_samples条数据
            sample_df = df.head(num_samples)
            total_count = num_samples
            print(f"开始分析前{num_samples}条完整问答数据...")
        
        results = []
        success_count = 0
        failed_count = 0
        
        print("使用百炼API，支持重试机制和超时延长")
        
        for idx, row in sample_df.iterrows():
            # 使用原始索引+1作为rank
            actual_rank = idx + 1
            print(f"\n=== 正在分析第{actual_rank}条数据 (DataFrame索引: {idx}) ===")
            
            result = self.analyze_citation_quality(row)
            
            if result['api_success']:
                print("    ✓ 分析成功")
                success_count += 1
            else:
                print(f"    ✗ 分析失败: {result['api_error']}")
                failed_count += 1
            
            results.append({
                'rank': actual_rank,
                **result
            })
            
            # 简化调用间隔，重试机制已经处理了大部分错误情况
            if result['api_success']:
                time.sleep(3)  # 成功后等3秒（原5秒）
            else:
                time.sleep(1)   # 失败后等1秒（原2秒），因为重试机制已经等待过了
        
        print(f"\n=== 方案1百炼版分析完成 ===")
        print(f"成功: {success_count}条, 失败: {failed_count}条")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存分析结果，增强错误处理"""
        try:
            # 创建目录（如果不存在）
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建输出目录：{output_dir}")
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✓ 结果已成功保存到：{output_path}")
            print(f"    文件大小：{os.path.getsize(output_path)} 字节")
            
        except PermissionError:
            print(f"✗ 保存失败：没有写入权限 - {output_path}")
        except FileNotFoundError:
            print(f"✗ 保存失败：路径不存在 - {output_path}")
        except Exception as e:
            print(f"✗ 保存失败：{e}")

def get_user_choice() -> dict:
    """获取用户选择的运行模式"""
    print("\n=== 引文分析脚本 ===")
    print("请选择运行模式：")
    print("1. 分析特定rank的单条数据（同步模式）")
    print("2. 从指定位置开始分析指定数量的数据（同步模式）") 
    print("3. 分析所有数据（并发模式）")
    print("4. 分析前N条数据（并发模式）")
    print("5. 分析特定rank的单条数据（并发模式）")
    print("6. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-6): ").strip()
            
            if choice == '1':
                rank = int(input("请输入要分析的rank (从1开始): "))
                return {"mode": "specific_rank", "specific_rank": rank}
                
            elif choice == '2':
                start_from = int(input("请输入起始位置 (从1开始): "))
                count_input = input("请输入要分析的数量 (留空表示分析到结尾): ").strip()
                num_samples = int(count_input) if count_input else None
                return {"mode": "start_from", "start_from": start_from, "num_samples": num_samples}
                
            elif choice == '3':
                return {"mode": "all", "num_samples": None}
                
            elif choice == '4':
                num_samples = int(input("请输入要分析的数据量: "))
                return {"mode": "head", "num_samples": num_samples}
                
            elif choice == '5':
                rank = int(input("请输入要分析的rank (从1开始): "))
                return {"mode": "specific_rank_async", "specific_rank": rank}
                
            elif choice == '6':
                print("退出程序")
                return {"mode": "exit"}
                
            else:
                print("无效选择，请重新输入")
                
        except ValueError:
            print("输入格式错误，请输入数字")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return {"mode": "exit"}

def main_unified():
    """统一主函数，根据用户选择决定同步或异步"""
    # 获取用户选择
    user_choice = get_user_choice()
    if user_choice["mode"] == "exit":
        return
        
    # 根据模式决定使用同步还是异步
    if user_choice["mode"] in ["specific_rank", "start_from"]:
        # 同步模式
        main_sync(user_choice)
    else:
        # 异步模式
        asyncio.run(main_async_impl(user_choice))

def main_sync(user_choice):
    """同步版本主函数"""
    analyzer = Method1BailianAnalyzer()
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    
    # 根据用户选择设置输出文件名
    if user_choice["mode"] == "specific_rank":
        output_path = f"local_data/citation_analysis_rank_{user_choice['specific_rank']}_sync_results.json"
    elif user_choice["mode"] == "start_from":
        if user_choice.get("num_samples"):
            output_path = f"local_data/citation_analysis_from_{user_choice['start_from']}_count_{user_choice['num_samples']}_sync_results.json"
        else:
            output_path = f"local_data/citation_analysis_from_{user_choice['start_from']}_to_end_sync_results.json"
    else:
        output_path = "local_data/citation_analysis_sync_results.json"

    print(f"开始同步分析...")
    print(f"输出文件：{output_path}")
    
    # 根据用户选择调用不同的分析方法
    if user_choice["mode"] == "specific_rank":
        results = analyzer.batch_analyze(csv_path, specific_rank=user_choice["specific_rank"])
    elif user_choice["mode"] == "start_from":
        results = analyzer.batch_analyze(csv_path, 
                                       num_samples=user_choice.get("num_samples"),
                                       start_from=user_choice["start_from"])
    else:
        results = analyzer.batch_analyze(csv_path)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n方案1百炼版分析完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:1]):
                print(f"\n{i+1}. 第{result['rank']}条数据:")
                print(f"   使用引用: {result['citations_used']}")
                print(f"   可用引文: {len(result['citations_available'])}个")
                if result['analysis']:
                    print(f"   分析片段: {result['analysis'][:150]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

async def main_async_impl(user_choice):
    """异步并发版本的主函数实现"""
    analyzer = Method1BailianAnalyzer(concurrent_limit=50)  # 50并发
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    
    # 根据用户选择设置输出文件名
    if user_choice["mode"] == "specific_rank_async":
        output_path = f"local_data/citation_analysis_rank_{user_choice['specific_rank']}_async_results.json"
    elif user_choice["mode"] == "head":
        output_path = f"local_data/citation_analysis_head_{user_choice['num_samples']}_results.json"
    else:
        output_path = "local_data/citation_analysis_method1_bailian_concurrent_results.json"

    print(f"🚀 启动高速并发分析模式！")
    print(f"输出文件：{output_path}")
    
    # 根据用户选择调用不同的分析方法
    if user_choice["mode"] == "specific_rank_async":
        results = await analyzer.batch_analyze_concurrent(csv_path, specific_rank=user_choice["specific_rank"])
    elif user_choice["mode"] == "head":
        results = await analyzer.batch_analyze_concurrent(csv_path, num_samples=user_choice["num_samples"])
    else:  # "all"
        results = await analyzer.batch_analyze_concurrent(csv_path, num_samples=None)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n🎉 并发分析完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:1]):
                print(f"\n{i+1}. 第{result['rank']}条数据:")
                print(f"   使用引用: {result['citations_used']}")
                print(f"   可用引文: {len(result['citations_available'])}个")
                if result['analysis']:
                    print(f"   分析片段: {result['analysis'][:150]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

# 保留原来的函数名作为别名        
async def main_async():
    """异步并发版本的主函数（兼容性保留）"""
    user_choice = {"mode": "all"}
    await main_async_impl(user_choice)

def main():
    """交互式主函数"""
    # 获取用户选择
    user_choice = get_user_choice()
    if user_choice["mode"] == "exit":
        return
        
    analyzer = Method1BailianAnalyzer()
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    
    # 根据用户选择设置输出文件名
    if user_choice["mode"] == "specific_rank":
        output_path = f"local_data/citation_analysis_rank_{user_choice['specific_rank']}_sync_results.json"
    elif user_choice["mode"] == "start_from":
        if user_choice.get("num_samples"):
            output_path = f"local_data/citation_analysis_from_{user_choice['start_from']}_count_{user_choice['num_samples']}_sync_results.json"
        else:
            output_path = f"local_data/citation_analysis_from_{user_choice['start_from']}_to_end_sync_results.json"
    elif user_choice["mode"] == "head":
        output_path = f"local_data/citation_analysis_head_{user_choice['num_samples']}_sync_results.json"
    else:
        output_path = "local_data/citation_analysis_method1_bailian_sync_results.json"

    print(f"开始同步分析...")
    print(f"输出文件：{output_path}")
    
    # 根据用户选择调用不同的分析方法
    if user_choice["mode"] == "specific_rank":
        results = analyzer.batch_analyze(csv_path, specific_rank=user_choice["specific_rank"])
    elif user_choice["mode"] == "start_from":
        results = analyzer.batch_analyze(csv_path, 
                                       num_samples=user_choice.get("num_samples"),
                                       start_from=user_choice["start_from"])
    else:  # "all" or "head"
        results = analyzer.batch_analyze(csv_path, num_samples=user_choice.get("num_samples"))
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n方案1百炼版分析完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:1]):
                print(f"\n{i+1}. 第{result['rank']}条数据:")
                print(f"   使用引用: {result['citations_used']}")
                print(f"   可用引文: {len(result['citations_available'])}个")
                if result['analysis']:
                    print(f"   分析片段: {result['analysis'][:150]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

if __name__ == "__main__":
    # 运行统一主函数，根据用户选择自动决定同步或异步
    main_unified()