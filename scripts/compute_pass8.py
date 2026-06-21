import json
import glob
import os
from collections import defaultdict

def main():
    # 找到最新生成的 samples jsonl 文件
    sample_files = glob.glob("samples_mixed_pass8_*.jsonl")
    if not sample_files:
        # 有可能是在 output_path 对应的目录里
        sample_files = glob.glob("results_mixed_pass8.json/*samples_mixed_pass8_*.jsonl")
        
    if not sample_files:
        print("未找到对应的 samples jsonl 日志文件，请确认 --log_samples 是否生效或评测是否成功结束。")
        return

    # 获取最新的文件
    latest_sample_file = max(sample_files, key=os.path.getctime)
    print(f"解析样本文件: {latest_sample_file}")

    # 按 doc_id 聚合结果
    pass_counts = defaultdict(int)
    total_counts = defaultdict(int)
    groundtruths = {}      # doc_id -> groundtruth (来自原始 output)
    all_responses = defaultdict(list)  # doc_id -> [resp1, resp2, ...]
    
    with open(latest_sample_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过第 {line_num} 行（JSON 解析失败）: {e}")
                continue
            
            doc_id = data.get("doc_id")
            if doc_id is None:
                continue
            
            # 提取 groundtruth（samples 文件中的 target 对应原始数据的 output）
            if doc_id not in groundtruths:
                if "target" in data:
                    groundtruths[doc_id] = data["target"]
                elif "doc" in data and "output" in data["doc"]:
                    groundtruths[doc_id] = data["doc"]["output"]
            
            # 收集模型的生成结果（原始输出）
            if "resps" in data and isinstance(data["resps"], list):
                for resp_list in data["resps"]:
                    if isinstance(resp_list, list):
                        all_responses[doc_id].extend(resp_list)
                    else:
                        all_responses[doc_id].append(resp_list)
            
            # 评测标准：exact_match 字段
            is_correct = 0
            if "exact_match" in data:
                is_correct = int(bool(data["exact_match"]))
            elif "metrics" in data and "exact_match" in data["metrics"]:
                is_correct = int(bool(data["metrics"]["exact_match"]))
            
            pass_counts[doc_id] += is_correct
            total_counts[doc_id] += 1
    
    original_data_path = "data/mixed_10.jsonl"
    output_data_path = "data/mixed_pass8.jsonl"
    
    if not os.path.exists(original_data_path):
        print(f"找不到原始文件: {original_data_path}")
        return

    print("开始向原数据注入 groundtruth、pass@8 和模型生成结果并保存到新文件...")
    with open(original_data_path, 'r', encoding='utf-8') as fin, \
         open(output_data_path, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 使用对应 doc_id 的测试结果，如果没有就默认为 0/8
            passes = pass_counts.get(idx, 0)
            total = total_counts.get(idx, 8)
            denominator = max(total, 8)
            
            item["pass@8"] = f"{passes}/{denominator}"
            
            # 优先使用从 samples 中提取的 groundtruth，否则回退到原始数据的 output 字段
            item["groundtruth"] = groundtruths.get(idx, item.get("output", ""))
            
            # 注入该题目对应的 8 次模型生成结果（如果收集到的话）
            if idx in all_responses:
                item["responses"] = all_responses[idx]
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"处理完毕！结果已保存至 {output_data_path}")
    print(f"共处理 {len(pass_counts)} 个有效 doc_id")

if __name__ == "__main__":
    main()
