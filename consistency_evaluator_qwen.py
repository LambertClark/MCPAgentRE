#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用阿里云百炼API评估标注句子与引文内容一致性的增强版程序

功能：
1. 从citation_results.json读取标注句子数据
2. 从正文引文内容.xlsx读取引文内容数据
3. 调用阿里云百炼API评估一致性
4. 支持批量处理相同rank的数据
5. 支持断点续传和中间结果保存
6. 支持可配置的rank范围评测
7. 输出三个JSON格式结果文件

作者：AI Assistant
日期：2024
"""

import json
import pandas as pd
import requests
import time
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConsistencyEvaluatorQwen:
    def __init__(self, api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                 rank_start: int = 1, rank_end: int = 50):
        # 从环境变量读取API密钥
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("API密钥未设置。请设置DASHSCOPE_API_KEY环境变量或通过参数传入。")
        self.base_url = base_url
        self.max_input_length = 128000  # 128k字符限制

        # 创建checkpoints目录（如果不存在）
        self.checkpoint_dir = "../../results/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 生成包含时间戳和rank范围的检查点文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            f"qwen_evaluation_checkpoint_rank{rank_start}-{rank_end}_{timestamp}.json")

        # Token统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0

    def load_citation_data(self, file_path: str, rank_start: int = 1, rank_end: int = 50) -> List[Dict[str, Any]]:
        """
        从citation_results.json读取标注句子数据，筛选指定rank范围的记录

        Args:
            file_path: JSON文件路径
            rank_start: 起始rank值
            rank_end: 结束rank值

        Returns:
            筛选后的数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 筛选指定rank范围的数据
            filtered_data = [item for item in data if rank_start <= item['rank'] <= rank_end]
            logger.info(f"从{len(data)}条记录中筛选出{len(filtered_data)}条rank {rank_start}-{rank_end}的记录")

            return filtered_data

        except Exception as e:
            logger.error(f"读取citation数据失败: {e}")
            return []

    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        从Excel文件读取引文内容数据

        Args:
            file_path: Excel文件路径

        Returns:
            DataFrame对象
        """
        try:
            df = pd.read_excel(file_path)
            logger.info(f"成功读取Excel文件，共{len(df)}行数据")
            return df

        except Exception as e:
            logger.error(f"读取Excel数据失败: {e}")
            return pd.DataFrame()

    def group_data_by_rank(self, citation_data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        按rank值对数据进行分组

        Args:
            citation_data: 标注句子数据

        Returns:
            按rank分组的数据字典
        """
        grouped_data = defaultdict(list)
        for item in citation_data:
            grouped_data[item['rank']].append(item)

        logger.info(f"数据按rank分组完成，共{len(grouped_data)}个rank组")
        return dict(grouped_data)

    def prepare_batch_evaluation_data(self, rank_group: List[Dict[str, Any]], excel_df: pd.DataFrame, rank: int) -> \
    Dict[str, Any]:
        """
        为同一rank的数据准备批量评估格式

        Args:
            rank_group: 同一rank的数据列表
            excel_df: Excel数据
            rank: rank值

        Returns:
            批量评估数据
        """
        batch_data = {
            "rank": rank,
            "topics": [],
            "citations": {}
        }

        # 收集所有引文编号
        all_citations = set()

        for item in rank_group:
            # 处理citation字段，可能是单个数字或数组
            citation_raw = item['citation']
            if isinstance(citation_raw, list):
                citation_numbers = citation_raw
            elif isinstance(citation_raw, (int, str)):
                citation_numbers = [int(citation_raw)]
            else:
                logger.warning(f"未知的citation格式: {citation_raw}，跳过该条目")
                continue

            topic_data = {
                "topic": item['topic'],
                "citation_numbers": citation_numbers
            }
            batch_data["topics"].append(topic_data)

            # 收集引文编号
            for citation_num in citation_numbers:
                all_citations.add(int(citation_num))

        # 获取所有相关引文内容
        for citation_num in sorted(all_citations):
            col_name = f"引文{citation_num}"
            if col_name in excel_df.columns and rank <= len(excel_df):
                citation_content = excel_df.loc[rank - 1, col_name]  # rank从1开始，索引从0开始
                if pd.notna(citation_content):
                    batch_data["citations"][f"引文{citation_num}"] = str(citation_content)
                else:
                    batch_data["citations"][f"引文{citation_num}"] = "[引文内容为空]"
            else:
                batch_data["citations"][f"引文{citation_num}"] = "[引文列不存在或行号超出范围]"

        logger.debug(
            f"rank {rank} 批量数据准备完成: {len(batch_data['topics'])}个topic, {len(batch_data['citations'])}个引文")
        return batch_data

    def create_batch_prompt(self, batch_data: Dict[str, Any]) -> str:
        """
        创建批量评估的API提示词

        Args:
            batch_data: 批量评估数据

        Returns:
            格式化的提示词
        """
        rank = batch_data["rank"]
        topics = batch_data["topics"]
        citations = batch_data["citations"]

        # 构建引文内容部分
        citation_text = "\n\n".join(
            [f"{key}: {value[:500]}{'...' if len(value) > 500 else ''}" for key, value in citations.items()])

        # 构建待评估句子部分
        topics_text = ""
        for i, topic_data in enumerate(topics, 1):
            citation_nums_str = ", ".join([str(num) for num in topic_data['citation_numbers']])
            topics_text += f"\n{i}. 标注句子：{topic_data['topic']}\n   引用的引文编号：[{citation_nums_str}]\n"

        prompt = f"""【分析要求】
你是一个严谨的文本分析专家。你的核心任务是：评估以下标注句子与对应引文内容的一致性。


**重要规则（必须严格遵守）**：
- 评估时必须严格按照标注句子与引文的对应关系进行，不能随意交换位置或改变内容。
- 需要找到标注句子中引用的具体引文内容，不能根据标注句子的内容去匹配引文。
- 标注句子中引用的引文编号必须与引文内容中的编号一致。
- 每个标注句子的citation_topic字段中必须包含对应引文的具体内容，不能只包含引文编号
- 在评估之前必须先将标注句子与引文内容进行匹配，确保citation_topic字段中包含了引文的具体内容，最好是引文中有关联的完整的句子或者段落放入citation_topic字段
- 评估时需要时刻谨记评估标准，防止错评

评估标准：
1. 事实一致性
* 关键数据（年份、数值、统计结果）是否完全匹配引文。
* 专业术语定义是否与引文原文一致（如"造血干细胞移植"≠"干细胞疗法"）。
* 案例/事件描述是否无虚构或篡改（如引文未提"淋巴瘤治疗",AI不得添加）。
2. 内容完整性
* AI是否遗漏引文的关键限制条件（如"需配合化疗"被省略）。
* 是否擅自扩展引文范围（如引文仅支持"白血病",AI添加"再生障碍性贫血"）。
* 引文结论的适用边界是否被突破（如"部分有效"被改为"普遍有效"）。
3. 语义匹配度
* 核心论点逻辑链是否与引文一致（如"生成疾病细胞→研究机制"是否完整保留）。
* 引文中的因果关系是否被曲解（如"收入提升因数字技术"≠"因政策扶持"）。
* 引文中的否定表述是否被错误转换为肯定（如"未证明有效"≠"证明有效"）。
4. 引用规范性
* 引用的文献/期刊是否存在且未被虚构（如DOI验证失败或期刊已停刊）。
* 引用位置是否准确（如引文描述"细胞疗法"，AI误标为"基因治疗"）。
* 引用格式是否完整（缺失作者、出版年份、页码等关键信息）。
5. 逻辑连贯性
* 多个引文合并时是否产生矛盾（如citation:1与citation:6结论冲突）。
* 图表数据与正文分析是否一致（如正文称"全国数据"，图表仅含局部样本）。
* 是否出现反常识推论（如"量子计算可治愈癌症"无依据）。

输出要求：
请严格按照以下JSON格式输出，对每个标注句子分别评估，不要包含任何其他文字：
[
  {{
    "topic": "标注句子文本内容",
    "citation_topic": "对应引文文本内容,需要具体到句子或者段落",
    "consistency": "一致或者不一致",
    "reason": "一致或者不一致的具体原因说明，需要具体指出关键的一致或不一致之处",
    "rank": "{rank}",
    "citation_numbers": [引文编号列表]
  }},
  ...
]

示例：
[
    {{
    "topic": "提供成就证明（如国际奖项、高影响力作品、参展记录、媒体报道等）",
    "citation_topic": "提供相关成就的证明记录，如奖项、作品、国际影响力等，彰显专业实力与行业认可；",
    "consistency": "一致",
    "reason": "标注句子中的'成就证明'及具体示例（国际奖项、高影响力作品、参展记录、媒体报道）与引文1中'提供相关成就的证明记录，如奖项、作品、国际影响力等'完全匹配，且引文4也提及'获奖经历、媒体报道等相关证明'，语义和内容均一致。无虚构、篡改或遗漏关键信息。",
    "rank": "34",
    "citation_numbers": [
      1,
      4
    ]
    }},
    {{
    "topic": "补充新生细胞替代老化细胞，改善皮肤弹性（增加胶原蛋白）、骨密度及代谢功能",
    "citation_topic": "引文2: 干细胞能补充新生细胞，替代老化、受损细胞，改善组织代谢与营养供给。注入皮肤的干细胞，增加胶原蛋白合成，减少皱纹，让肌肤恢复弹性、光泽。 | 引文5: 干细胞可以起到调节代谢的功能，立体运用干细胞的多项分化能力，从而提高代谢的效率从而使机体的代谢废物排出体外，促进机体内的营养吸收，维持机体的正常生理功能；",
    "consistency": "不一致",
    "reason": "关键内容不匹配：1) 标注句子中'改善骨密度'在引文2和引文5中均未提及，属于擅自扩展引文范围；2) 引文2明确说明干细胞'改善组织代谢与营养供给'，而标注句子简化为'改善代谢功能'，虽语义相近但不够完整；3) 引文5提到调节代谢功能是'提高代谢效率，促进废物排出和营养吸收'，标注句子的'改善代谢功能'表述过于笼统，未完整反映引文细节",
    "rank": "1",
    "citation_numbers": [
      2,
      5
    ]
    }}
]


注意事项：
- 必须确保返回的是有效的JSON格式
- 每个标注句子都必须有对应的评估结果
- 评估理由要具体明确，说明为什么一致或不一致
- 如果标注句子与引文内容完全匹配或基本一致，标记为"一致"
- 如果存在事实错误、语义偏差或逻辑问题，标记为"不一致"
- consistency字段必须为"一致"或"不一致"

待评估内容：

Rank: {rank}

引文内容：
{citation_text}

待评估的标注句子：{topics_text}

请开始评估："""

        return prompt

    def call_qwen_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        调用阿里云百炼API

        Args:
            prompt: 提示词
            max_retries: 最大重试次数

        Returns:
            API响应内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "qwen3-32b",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 16000,
                "top_p": 0.8,
                "enable_thinking": False
            },

        }

        for attempt in range(max_retries):
            try:
                logger.debug(f"正在调用API (尝试 {attempt + 1}/{max_retries})...")

                # 检查输入长度
                if len(prompt) > self.max_input_length:
                    logger.warning(f"提示词长度({len(prompt)})超过限制({self.max_input_length})，进行截断")
                    prompt = prompt[:self.max_input_length]
                    data["input"]["messages"][0]["content"] = prompt

                response = requests.post(self.base_url, headers=headers, json=data, timeout=300)

                if response.status_code == 200:
                    result = response.json()

                    # 统计token使用情况
                    if 'usage' in result:
                        usage = result['usage']
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        total_tokens = usage.get('total_tokens', input_tokens + output_tokens)

                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_tokens += total_tokens
                        self.api_call_count += 1

                        logger.debug(f"Token使用: 输入={input_tokens}, 输出={output_tokens}, 总计={total_tokens}")

                    # 处理新的API响应格式
                    if 'output' in result:
                        if 'text' in result['output']:
                            content = result['output']['text']
                            logger.debug(f"API调用成功，响应长度: {len(content)}")
                            return content
                        elif 'choices' in result['output'] and len(result['output']['choices']) > 0:
                            content = result['output']['choices'][0]['message']['content']
                            logger.debug(f"API调用成功，响应长度: {len(content)}")
                            return content
                    else:
                        logger.error(f"API响应格式异常: {result}")

                elif response.status_code == 429:  # 速率限制
                    wait_time = 2 ** attempt * 2
                    logger.warning(f"API速率限制，等待{wait_time}秒后重试")
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"API调用超时，第{attempt + 1}次重试")
                time.sleep(2)

            except Exception as e:
                logger.error(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    delay = 2 * (attempt + 1)
                    logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)

        logger.error(f"API调用失败，已达到最大重试次数 {max_retries}")
        return None

    def parse_batch_api_response(self, response: str, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析批量API响应，转换为标准格式

        Args:
            response: API响应内容
            batch_data: 原始批量数据

        Returns:
            标准格式的评估结果列表
        """
        try:
            # 清理响应内容
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # 解析JSON
            results = json.loads(cleaned_response)

            if not isinstance(results, list):
                logger.error(f"API响应不是列表格式: {type(results)}")
                return []

            # 转换为标准格式
            standard_results = []
            for result in results:
                if not isinstance(result, dict):
                    continue

                required_keys = ['topic', 'citation_numbers', 'consistency', 'reason']
                if not all(key in result for key in required_keys):
                    logger.warning(f"结果格式不完整: {result}")
                    continue

                # 标准化consistency值
                consistency = result['consistency'].lower()
                if consistency in ['consistent', '一致']:
                    consistency_value = '一致'
                elif consistency in ['inconsistent', '不一致']:
                    consistency_value = '不一致'
                else:
                    logger.warning(f"未知的consistency值: {result['consistency']}")
                    continue

                standard_result = {
                    'topic': result['topic'],
                    'citation_topic': result['citation_topic'],
                    'consistency': consistency_value,
                    'reason': result['reason'],
                    'rank': batch_data['rank'],
                    'citation_numbers': result['citation_numbers']
                }
                standard_results.append(standard_result)

            logger.debug(f"成功转换 {len(standard_results)} 个结果为标准格式")
            return standard_results

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"响应内容: {response[:300]}...")
            return []
        except Exception as e:
            logger.error(f"解析API响应异常: {e}")
            return []

    def parse_batch_response(self, response: str, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析批量API响应并生成标准JSON输出

        Args:
            response: API响应内容
            batch_data: 原始批量数据

        Returns:
            标准格式的结果列表，解析失败返回空列表
        """
        try:
            # 清理响应内容，移除可能的markdown格式
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # 尝试直接解析JSON数组
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                results = json.loads(cleaned_response)

                # 验证结果格式
                if isinstance(results, list) and all(
                        isinstance(item, dict) and
                        all(key in item for key in ['topic', 'citation_topic', 'reason'])
                        for item in results
                ):
                    # 验证每个结果的consistency值
                    validated_results = []
                    for result in results:
                        if 'consistency' not in result:
                            logger.warning(f"结果缺少consistency字段: {result}")
                            continue
                        if result['consistency'] not in ['一致', '不一致']:
                            logger.warning(f"consistency值无效: {result['consistency']}")
                            continue
                        validated_results.append(result)

                    logger.info(f"成功解析 {len(validated_results)}/{len(batch_data['topics'])} 个评估结果")
                    return validated_results

            # 如果直接解析失败，尝试提取JSON数组部分
            import re
            json_match = re.search(r'\[[^\[\]]*"topic"[^\[\]]*\]', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                results = json.loads(json_str)
                if isinstance(results, list) and all(
                        isinstance(item, dict) and
                        all(key in item for key in ['topic', 'citation_topic', 'reason'])
                        for item in results
                ):
                    return results

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应前500字符: {response[:500]}")
        except Exception as e:
            logger.error(f"响应解析异常: {e}")

        # 解析失败，返回空列表
        logger.warning("批量API响应解析失败")
        return []

    def is_consistent(self, result: Dict[str, Any]) -> bool:
        """
        判断评估结果是否一致

        Args:
            result: 评估结果

        Returns:
            True表示一致，False表示不一致
        """
        # 直接使用consistency字段作为判断标准
        consistency = result.get('consistency', '').strip()

        # 如果consistency字段明确标注为"一致"，则返回True
        if consistency == '一致':
            return True
        # 如果consistency字段明确标注为"不一致"，则返回False
        elif consistency == '不一致':
            return False
        else:
            # 如果consistency字段为空或其他值，使用原有逻辑作为备用判断
            reason = result.get('reason', '').strip()
            consistent_keywords = ['一致', '相符', '支持', '正确', '无问题', '匹配']
            inconsistent_keywords = ['不一致', '不符', '不支持', '错误', '问题', '缺失', '未提及', '不匹配']

            # 检查不一致关键词
            for keyword in inconsistent_keywords:
                if keyword in reason:
                    return False

            # 检查一致关键词或reason为空
            if not reason or any(keyword in reason for keyword in consistent_keywords):
                return True

            # 默认根据reason长度判断：详细说明通常表示不一致
            return len(reason) < 20

    def save_checkpoint(self, processed_ranks: List[int], all_results: List[Dict[str, Any]]):
        """
        保存中间结果检查点

        Args:
            processed_ranks: 已处理的rank列表
            all_results: 所有结果
        """
        checkpoint_data = {
            "processed_ranks": processed_ranks,
            "all_results": all_results,
            "timestamp": time.time()
        }

        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"检查点已保存，已处理{len(processed_ranks)}个rank")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def load_checkpoint(self) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        加载中间结果检查点

        Returns:
            (已处理的rank列表, 所有结果)
        """
        if not os.path.exists(self.checkpoint_file):
            return [], []

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            processed_ranks = checkpoint_data.get('processed_ranks', [])
            all_results = checkpoint_data.get('all_results', [])

            logger.info(f"加载检查点成功，已处理{len(processed_ranks)}个rank，共{len(all_results)}条结果")
            return processed_ranks, all_results

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return [], []

    def print_token_statistics(self):
        """
        输出token使用统计信息
        """
        logger.info(f"\n=== Token使用统计 ===")
        logger.info(f"API调用次数: {self.api_call_count}")
        logger.info(f"输入Token总计: {self.total_input_tokens:,}")
        logger.info(f"输出Token总计: {self.total_output_tokens:,}")
        logger.info(f"Token总计: {self.total_tokens:,}")
        if self.api_call_count > 0:
            logger.info(f"平均每次调用输入Token: {self.total_input_tokens / self.api_call_count:.1f}")
            logger.info(f"平均每次调用输出Token: {self.total_output_tokens / self.api_call_count:.1f}")
            logger.info(f"平均每次调用总Token: {self.total_tokens / self.api_call_count:.1f}")

    def save_results(self, all_results: List[Dict[str, Any]], output_dir: str = "../../results"):
        """
        保存三个JSON结果文件

        Args:
            all_results: 所有评估结果
            output_dir: 输出目录
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 分类结果
            consistent_results = []
            inconsistent_results = []

            for result in all_results:
                if self.is_consistent(result):
                    consistent_results.append(result)
                else:
                    inconsistent_results.append(result)

            # 保存所有结果
            all_results_file = os.path.join(output_dir, "qwen_all_results.json")
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"所有结果已保存到: {all_results_file}")

            # 保存一致结果
            consistent_file = os.path.join(output_dir, "qwen_consistency_results.json")
            with open(consistent_file, 'w', encoding='utf-8') as f:
                json.dump(consistent_results, f, ensure_ascii=False, indent=2)
            logger.info(f"一致结果已保存到: {consistent_file}")

            # 保存不一致结果
            inconsistent_file = os.path.join(output_dir, "qwen_inconsistency_results.json")
            with open(inconsistent_file, 'w', encoding='utf-8') as f:
                json.dump(inconsistent_results, f, ensure_ascii=False, indent=2)
            logger.info(f"不一致结果已保存到: {inconsistent_file}")

            logger.info(
                f"结果统计: 总计{len(all_results)}条，一致{len(consistent_results)}条，不一致{len(inconsistent_results)}条")

            # 输出token统计
            self.print_token_statistics()

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def evaluate_batch(self, rank_group: List[Dict[str, Any]], excel_df: pd.DataFrame, rank: int) -> List[
        Dict[str, Any]]:
        """
        对同一rank的数据进行批量评估

        Args:
            rank_group: 同一rank的数据列表
            excel_df: Excel数据
            rank: rank值

        Returns:
            评估结果列表
        """
        logger.info(f"开始评估rank {rank}的数据，共{len(rank_group)}条")

        try:
            # 准备批量数据
            batch_data = self.prepare_batch_evaluation_data(rank_group, excel_df, rank)

            if not batch_data['topics']:
                logger.warning(f"rank {rank} 没有有效的topic数据")
                return []

            # 创建提示词
            prompt = self.create_batch_prompt(batch_data)
            logger.debug(f"rank {rank} 提示词长度: {len(prompt)}")

            # 调用API并处理响应解析失败的重试机制
            batch_results = []
            max_parse_retries = 3

            for parse_attempt in range(max_parse_retries):
                # 调用API
                api_response = self.call_qwen_api(prompt)

                if api_response:
                    # 尝试解析响应
                    parsed_results = self.parse_batch_api_response(api_response, batch_data)

                    if parsed_results:
                        batch_results = parsed_results
                        break
                    else:
                        logger.warning(f"rank {rank} 响应解析失败，重试第{parse_attempt + 1}次")
                        if parse_attempt < max_parse_retries - 1:
                            time.sleep(3)  # 等待后重试
                else:
                    logger.error(f"rank {rank} API调用失败")
                    break

            # 处理最终结果
            if batch_results:
                if len(batch_results) != len(batch_data['topics']):
                    logger.warning(
                        f"rank {rank} 结果数量不匹配: 期望{len(batch_data['topics'])}，实际{len(batch_results)}")

                logger.info(f"rank {rank} 评估完成，获得{len(batch_results)}个结果")
                return batch_results
            else:
                # 所有重试都失败，为该rank的每个topic添加失败记录
                failure_results = []
                for topic_data in batch_data["topics"]:
                    failure_result = {
                        "topic": topic_data["topic"],
                        "citation_topic": "评估失败",
                        "consistency": "不一致",
                        "reason": "API调用失败或响应解析失败，无法获取评估结果",
                        "rank": rank,
                        "citation_numbers": topic_data["citation_numbers"]
                    }
                    failure_results.append(failure_result)

                logger.error(f"rank {rank} 评估失败，已添加{len(failure_results)}条失败记录")
                return failure_results

        except Exception as e:
            logger.error(f"rank {rank} 批量评估异常: {e}")
            return []

    def evaluate_consistency(self, citation_file: str, excel_file: str, rank_start: int = 1, rank_end: int = 50,
                             resume: bool = True):
        """
        主评估流程

        Args:
            citation_file: citation_results.json文件路径
            excel_file: Excel文件路径
            rank_start: 起始rank值
            rank_end: 结束rank值
            resume: 是否从检查点恢复
        """
        logger.info(f"开始一致性评估流程，rank范围: {rank_start}-{rank_end}")

        # 1. 加载数据
        citation_data = self.load_citation_data(citation_file, rank_start, rank_end)
        if not citation_data:
            logger.error("无法加载citation数据，程序退出")
            return

        excel_df = self.load_excel_data(excel_file)
        if excel_df.empty:
            logger.error("无法加载Excel数据，程序退出")
            return

        # 2. 按rank分组数据
        grouped_data = self.group_data_by_rank(citation_data)

        # 3. 加载检查点（如果启用恢复）
        processed_ranks = []
        all_results = []
        if resume:
            processed_ranks, all_results = self.load_checkpoint()

        # 4. 逐rank批量评估
        total_ranks = len(grouped_data)
        processed_count = 0

        for rank in sorted(grouped_data.keys()):
            if rank in processed_ranks:
                logger.info(f"跳过已处理的rank {rank}")
                continue

            processed_count += 1
            logger.info(f"正在评估rank {rank} ({processed_count}/{total_ranks})")

            # 使用批量评估方法
            batch_results = self.evaluate_batch(grouped_data[rank], excel_df, rank)

            # 添加结果
            if batch_results:
                all_results.extend(batch_results)
                processed_ranks.append(rank)
                logger.info(f"rank {rank} 处理完成，获得{len(batch_results)}条结果")
            else:
                processed_ranks.append(rank)
                logger.warning(f"rank {rank} 处理完成，但未获得有效结果")

            # 保存检查点
            self.save_checkpoint(processed_ranks, all_results)

            # 添加延迟避免API限制
            time.sleep(2)

        # 5. 保存最终结果
        self.save_results(all_results)

        # 6. 清理检查点文件
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("检查点文件已清理")

        logger.info(f"评估完成，总共处理了{len(all_results)}条数据")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='使用阿里云百炼API评估标注句子与引文内容一致性')
    parser.add_argument('--api-key', help='阿里云百炼API密钥（可选，默认从DASHSCOPE_API_KEY环境变量读取）')
    parser.add_argument('--rank-start', type=int, default=10, help='起始rank值')
    parser.add_argument('--rank-end', type=int, default=20, help='结束rank值')
    parser.add_argument('--citation-file', default='../../results/citation_results.json',
                        help='citation_results.json文件路径')
    parser.add_argument('--excel-file', default='../../data/正文引文内容.xlsx', help='Excel文件路径')
    parser.add_argument('--no-resume', action='store_true', help='不从检查点恢复，重新开始评估')

    args = parser.parse_args()

    print("=== 标注句子与引文内容一致性评估程序（阿里云百炼版） ===")
    print(f"评估rank范围: {args.rank_start}-{args.rank_end}")
    print(f"断点续传: {'禁用' if args.no_resume else '启用'}\n")

    # 创建评估器
    evaluator = ConsistencyEvaluatorQwen(api_key=args.api_key, rank_start=args.rank_start, rank_end=args.rank_end)

    # 开始评估
    evaluator.evaluate_consistency(
        citation_file=args.citation_file,
        excel_file=args.excel_file,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        resume=not args.no_resume
    )

    print("\n=== 评估完成 ===")


if __name__ == "__main__":
    main()