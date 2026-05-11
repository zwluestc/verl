import json
import os

def extract_correct_responses(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    output_dir = os.path.join(os.path.dirname(file_path), "correct_outputs")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # 获取题目 ID
            q_id = data.get("doc", {}).get("ID", "Unknown")
            
            # 检查是否回答正确
            is_correct = data.get("exact_match", 0.0) == 1.0
            
            if is_correct:
                count += 1
                # 提取模型原始的完整输出
                raw_response = data.get("resps", [[""]])[0][0]
                target_ans = data.get("target", "Unknown")
                filtered_resps = data.get("filtered_resps", ["[None]"])
                model_ans = filtered_resps[0] if filtered_resps else "[None]"
                
                # 保存到 Markdown 文件中
                out_file = os.path.join(output_dir, f"{q_id}_correct.md")
                with open(out_file, "w", encoding="utf-8") as out_f:
                    out_f.write(f"# 题目 ID: {q_id}\n\n")
                    out_f.write(f"**正确答案 (Target):** {target_ans}\n")
                    out_f.write(f"**模型提取答案 (Model):** {model_ans}\n\n")
                    out_f.write("---\n\n")
                    out_f.write("## 模型完整生成的正确解答过程：\n\n")
                    out_f.write(raw_response)
                
                print(f"已导出 [{q_id}] 的正确过程至: {out_file}")
                
    if count == 0:
        print("没有找到正确的输出记录。")
    else:
        print(f"\n提取完成，共导出了 {count} 个正确解答。你可以直接在 VS Code 中点开上述 Markdown 文件阅读。")

if __name__ == "__main__":
    file_path = "/Users/zwlustc/Documents/verl/data/test/lm-evaluation.jsonl"
    extract_correct_responses(file_path)
