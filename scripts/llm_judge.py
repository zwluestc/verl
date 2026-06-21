import json
import glob
import os
import re
import math
import concurrent.futures
from openai import OpenAI

def extract_boxed(text):
    """Attempt to extract \boxed{...} from text. Returns the content inside boxed, or the full text if not found."""
    # A simple regex for \boxed{...}. Note: it doesn't handle nested braces well, 
    # but it's a rough heuristic if LLM needs help. We'll let LLM do the heavy lifting anyway.
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def judge_response(client, model, target, response):
    """Use LLM to judge if the response matches the target."""
    system_prompt = (
        "You are an expert mathematics and logic evaluator. "
        "You will be given the Ground Truth Output (which might contain a complex derivation and a final boxed answer) "
        "and a Submitted Answer (which is the model's extracted final answer).\n"
        "Your task is to determine whether the Submitted Answer is mathematically equivalent to the exact final answer in the Ground Truth.\n"
        "Output ONLY a valid JSON string (without markdown formatting blocks) with the following structure:\n"
        "{\n"
        "  \"is_correct\": true/false,\n"
        "  \"reason\": \"short explanation\"\n"
        "}"
    )
    
    user_prompt = f"Ground Truth Output:\n{target}\n\nSubmitted Answer:\n{response}"
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content
        if not content:
            return False
            
        # 兼容一些模型返回时可能会带有 markdown code block 标记
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        res = json.loads(content.strip())
        return bool(res.get("is_correct", False))
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return False

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")
    
    if not api_key:
        print("未检测到 OPENAI_API_KEY，跳过 LLM Judge。")
        return
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    sample_files = glob.glob("samples_mixed_pass8_*.jsonl")
    if not sample_files:
        sample_files = glob.glob("results_mixed_pass8.json/*samples_mixed_pass8_*.jsonl")
        
    if not sample_files:
        print("未找到对应的 samples jsonl 日志文件进行 LLM Judge。")
        return

    latest_sample_file = max(sample_files, key=os.path.getctime)
    # Check if the latest file is already judged (to avoid infinite looping if run multiple times)
    if "llm_judged" in latest_sample_file:
        print(f"文件已被判卷: {latest_sample_file}")
        return
        
    print(f"使用 LLM {judge_model} 对生成结果进行判卷: {latest_sample_file}")
    
    judged_file_name = latest_sample_file.replace(".jsonl", "_llm_judged.jsonl")
    
    lines = []
    with open(latest_sample_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    def process_line(line_str):
        if not line_str.strip():
            return None
        try:
            data = json.loads(line_str)
        except json.JSONDecodeError:
            return None
            
        target = data.get("target", "")
        # If there's no target, try extract from doc
        if not target and "doc" in data and "output" in data["doc"]:
            target = data["doc"]["output"]
            
        # Get the actual model answer. Prioritize filtered_resps as it extracts <final_answer>
        response = ""
        if "filtered_resps" in data and data["filtered_resps"]:
            response = data["filtered_resps"][0]
        elif "resps" in data and data["resps"]:
            response = data["resps"][0]
            
        # If it's a list, flatten it (sometimes resps is [\"answer\"])
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
            
        if not str(response).strip():
            data["exact_match"] = False
            return data

        # Basic exact match shortcut: if string matches perfectly or exact_match was already set to true originally
        # it might save API calls, but the user specifically asked LLM to judge.
        # Let's just use LLM for all.
        
        is_matched = judge_response(client, judge_model, target, str(response))
        data["exact_match"] = is_matched
        
        # 兼容 lm-eval 的 metrics 数据结构，有可能是 list 或者 dict
        if "metrics" not in data:
            data["metrics"] = {"exact_match": is_matched}
        elif isinstance(data["metrics"], dict):
            data["metrics"]["exact_match"] = is_matched
        # 如果是 list，这里就不强行修改了，依赖 external 脚本读取 "exact_match" 的逻辑
        
        return data
        
    # Use ThreadPoolExecutor to speed up API calls
    judged_data = []
    print(f"正在启动 LLM 判卷流程 (Total lines: {len(lines)})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_line, lines))
        
    with open(judged_file_name, 'w', encoding='utf-8') as fout:
        for res in results:
            if res is not None:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                
    print(f"LLM 判卷完成！结果保存至: {judged_file_name}")

if __name__ == "__main__":
    main()
