import json
import os
from datasets import load_dataset

# 整理好的各个年份在 Hugging Face 上的优质开源数据源
# 格式: "保存文件名": [("HF仓库名", "对应split")]
AIME_REPOS = {
    "aime_2023": [
        ("MathArena/aime_2023_I", "train"),
        ("MathArena/aime_2023_II", "train")
    ],
    "aime_2024": [
        ("HuggingFaceH4/aime_2024", "train")
    ],
    "aime_2025": [
        ("math-ai/aime25", "train") # 2025年最新开源的集合
    ]
}

def find_key(row, candidate_keys):
    """自动匹配不同数据集里不规则的列名"""
    for key in candidate_keys:
        if key in row:
            return row[key]
        # 有些仓库列名首字母大写
        if key.capitalize() in row:
            return row[key.capitalize()]
    return None

def process_and_save():
    current_dir = os.getcwd()
    
    # 常见的题目和答案列名特征
    problem_keys = ['problem', 'question', 'raw_problem']
    answer_keys = ['answer', 'ground_truth', 'gold_answer', 'final_answer']

    for out_name, repos in AIME_REPOS.items():
        out_path = os.path.join(current_dir, f"{out_name}.jsonl")
        merged_data = []
        
        print(f"========== 正在处理 {out_name} ==========")
        for repo_id, split in repos:
            print(f"正在从 Hugging Face 拉取: {repo_id} ({split})...")
            try:
                ds = load_dataset(repo_id, split=split)
                
                for row in ds:
                    problem = find_key(row, problem_keys)
                    answer = find_key(row, answer_keys)
                    
                    if problem and answer is not None:
                        # 确保答案统一处理为纯字符串（去掉可能带有的多余空格）
                        ans_str = str(answer).strip()
                        
                        merged_data.append({
                            "problem": problem,
                            "ground_truth": ans_str,
                            "source": repo_id
                        })
            except Exception as e:
                print(f"拉取 {repo_id} 失败: {e}")
                
        # 写入 JSONL 文件
        if merged_data:
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in merged_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✅ 成功保存 {len(merged_data)} 条数据至 -> {out_path}\n")
        else:
            print(f"❌ {out_name} 没有解析出任何有效数据。\n")

if __name__ == "__main__":
    process_and_save()