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

【问题】
{question}

【回答内容】
{answer}

【回答中使用的引用标记】
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
你是一个严谨的文本分析专家。你的核心任务是：只分析包含引用标记[citation:x]的句子，完全跳过没有引用标记的句子。

**重要规则（必须严格遵守）**：
- 如果一个句子没有[citation:x]标记，立即跳过，不要在JSON输出中包含该句子
- 只分析和输出包含明确引用标记的句子
- 绝对不要分析没有引用标记的句子

请遵循以下步骤和规则：

1.  **逐句拆分**：将【回答内容】拆分为独立的观点或句子。
2.  **逐句分析**：对于每一个独立的观点或句子：
    a. **首先检查**：该句子是否包含引用标记 `[citation:x]`。如果没有任何引用标记，立即跳过该句子，不进行分析。
    b. 如果有引用标记，在对应的 `引文x` 中查找支持性证据。
    c. **严格判断**：
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
4. **空引用情况（重要）**：
    - 对于没有任何引用标记[citation:x]的句子，请完全跳过，不要在JSON输出中包含该句子。
    - 只分析包含明确引用标记的句子。
    - 错误示范（绝对不要这样做）：
      ```json
      {
        "topic": "优化操作设置是提升游戏体验的关键。",
        "citation_numbers": [],
        "consistency": "不一致", 
        "reason": "该句无引用标记..."
      }
      ```
    - 正确做法：完全跳过没有引用标记的句子，不在输出中包含。

**示例输出**：
      ```json
      [
        {
          "topic": "干细胞治疗成本高昂，单次治疗费用达20万–100万元。",
          "citation_numbers": [8],
          "consistency": "不一致",
          "reason": "引文8仅提及'生产此类细胞成本高昂'，但并未提供'20万–100万元'这一具体费用范围，该数字属于模型幻觉。"
        },
        {
          "topic": "干细胞具有修复、再生、调节三大核心功能。",
          "citation_numbers": [5],
          "consistency": "不一致",
          "reason": "引文5明确指出三大功能是'修复功能、调节代谢功能、自我更新功能'，并未提及'再生功能'，概念不完全匹配。"
        }
      ]
      ```
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
    
    async def analyze_citation_quality_async(self, session: aiohttp.ClientSession, row: pd.Series, index: int) -> Dict[str, Any]:
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
                'index': index,
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
            'index': index,
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
    
    async def batch_analyze_concurrent(self, csv_path: str, num_samples: int = None) -> List[Dict[str, Any]]:
        """异步并发批量分析"""
        df = self.load_data(csv_path)
        if df is None:
            return []
        
        # 确定要处理的数据
        if num_samples is None:
            sample_df = df
            total_count = len(df)
            print(f"开始并发分析所有{total_count}条完整问答数据...")
        else:
            sample_df = df.head(num_samples)
            total_count = num_samples
            print(f"开始并发分析前{num_samples}条完整问答数据...")
        
        print(f"使用百炼API，并发限制: {self.concurrent_limit}条")
        
        # 创建信号量来控制并发数量
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def process_with_semaphore(session, row, index):
            async with semaphore:
                result = await self.analyze_citation_quality_async(session, row, index + 1)
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
                print(f"[{progress}/{total_count}] {status} 第{result['index']}条 "
                      f"(用时: {elapsed:.1f}s, ETA: {eta:.1f}s)")
        
        # 按index排序结果
        completed_tasks.sort(key=lambda x: x['index'])
        
        # 统计结果
        success_count = sum(1 for r in completed_tasks if r['api_success'])
        failed_count = len(completed_tasks) - success_count
        
        total_time = time.time() - start_time
        print(f"\n=== 并发分析完成 ===")
        print(f"总用时: {total_time:.1f}秒")
        print(f"平均每条: {total_time/len(completed_tasks):.2f}秒")
        print(f"成功: {success_count}条, 失败: {failed_count}条")
        
        return completed_tasks
    
    def batch_analyze(self, csv_path: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """批量分析数据，num_samples=None时处理所有数据"""
        df = self.load_data(csv_path)
        if df is None:
            return []
        
        # 确定要处理的数据
        if num_samples is None:
            sample_df = df
            total_count = len(df)
            print(f"开始分析所有{total_count}条完整问答数据...")
        else:
            sample_df = df.head(num_samples)
            total_count = num_samples
            print(f"开始分析前{num_samples}条完整问答数据...")
        
        results = []
        success_count = 0
        failed_count = 0
        
        print("使用百炼API，支持重试机制和超时延长")
        
        for idx, row in sample_df.iterrows():
            print(f"\n=== 正在分析第{idx + 1}/{total_count}条 (原始索引: {idx}) ===")
            
            result = self.analyze_citation_quality(row)
            
            if result['api_success']:
                print("    ✓ 分析成功")
                success_count += 1
            else:
                print(f"    ✗ 分析失败: {result['api_error']}")
                failed_count += 1
            
            results.append({
                'index': idx + 1,
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

async def main_async():
    """异步并发版本的主函数"""
    analyzer = Method1BailianAnalyzer(concurrent_limit=50)  # 50并发
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_method1_bailian_concurrent_results.json"

    print("🚀 启动高速并发分析模式！")
    results = await analyzer.batch_analyze_concurrent(csv_path, num_samples=None)
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n🎉 并发分析完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:1]):
                print(f"\n{i+1}. 第{result['index']}条数据:")
                print(f"   使用引用: {result['citations_used']}")
                print(f"   可用引文: {len(result['citations_available'])}个")
                if result['analysis']:
                    print(f"   分析片段: {result['analysis'][:150]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

def main():
    """同步版本主函数（兼容性保留）"""
    analyzer = Method1BailianAnalyzer()
    
    # 数据路径
    csv_path = "local_data/副本正文引文内容（纯净版）.csv"
    output_path = "local_data/citation_analysis_method1_bailian_results.json"

    results = analyzer.batch_analyze(csv_path, num_samples=None)  # None表示处理所有数据
    
    if results:
        analyzer.save_results(results, output_path)
        print(f"\n方案1百炼版测试完成！")
        
        # 显示成功的结果预览
        success_results = [r for r in results if r['api_success']]
        if success_results:
            print(f"\n成功分析示例：")
            for i, result in enumerate(success_results[:1]):
                print(f"\n{i+1}. 第{result['index']}条数据:")
                print(f"   使用引用: {result['citations_used']}")
                print(f"   可用引文: {len(result['citations_available'])}个")
                if result['analysis']:
                    print(f"   分析片段: {result['analysis'][:150]}...")
    else:
        print("分析失败，请检查数据文件和AL_KEY配置")

if __name__ == "__main__":
    # 运行并发版本
    asyncio.run(main_async())