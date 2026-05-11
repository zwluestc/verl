import json
import os

def analyze_results(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    results = []
    correct_count = 0
    incorrect_count = 0
    invalid_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # 获取题目 ID
            q_id = data.get("doc", {}).get("ID", "Unknown")
            
            # 获取正确答案
            correct_ans = data.get("target", "Unknown")
            
            # 获取模型提取的答案
            filtered_resps = data.get("filtered_resps", ["[None]"])
            model_ans = filtered_resps[0] if filtered_resps else "[None]"
            
            # 判断是否正确
            is_correct = data.get("exact_match", 0.0) == 1.0
            
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
                if model_ans == "[invalid]":
                    invalid_count += 1

            results.append({
                "ID": q_id,
                "doc_id": data.get("doc_id", -1),
                "Correct": correct_ans,
                "Model": model_ans,
                "IsCorrect": is_correct
            })

    # 按照 AIME 题目 ID 进行排序，方便查看
    results.sort(key=lambda x: (x["ID"]))

    print(f"{'题目 ID':<15} | {'标准答案':<15} | {'模型答案':<15} | {'是否正确'}")
    print("-" * 65)
    for r in results:
        match_str = "✅ 正确" if r['IsCorrect'] else "❌ 错误"
        print(f"{r['ID']:<15} | {str(r['Correct']):<15} | {str(r['Model']):<15} | {match_str}")

    print("\n" + "="*65)
    print("总计分析:")
    print(f"已评估题目总数 : {len(results)}")
    print(f"正确数量       : {correct_count}")
    print(f"错误数量       : {incorrect_count} (其中包含 {invalid_count} 个 '[invalid]' 格式解析失败或无限循环)")
    print("="*65)

if __name__ == "__main__":
    file_path = "/Users/zwlustc/Documents/verl/data/test/lm-evaluation.jsonl"
    analyze_results(file_path)
