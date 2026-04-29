import re
from typing import Any

def _normalize_answer(text: Any) -> str:
    """强化版归一化：处理数学公式、空格、LaTeX 噪音"""
    if text is None:
        return ""
    text = str(text).lower().strip()
    
    # 替换 LaTeX 常见写法
    text = text.replace(r"\\dfrac", r"\\frac")
    text = text.replace(r"\\left(", "(").replace(r"\\right)", ")")
    text = text.replace(r"\\left[", "[").replace(r"\\right]", "]")
    text = text.replace(r"\\left\{", "{").replace(r"\\right\\}", "}")
    
    # 去除所有空格和 LaTeX 间距符号
    text = re.sub(r"\s+", "", text)
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "").replace(r"\!", "")
    
    # 去除所有大括号（数学意义上通常不影响对比）
    text = text.replace("{", "").replace("}", "")
    # 去除末尾标点
    text = text.rstrip(".")
    
    return text

def extract_answer(text: str) -> list[str]:
    """
    符合 GRPO 习惯的答案提取逻辑
    按优先级：<answer> 标签 > \boxed{} 内容 > 最后一行的非空字符串
    """
    text = str(text)
    candidates = []
    
    # 1. 提取 <answer> 标签内容 (GRPO 后的模型常用格式)
    tag_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(tag_matches)
    
    # 2. 提取 \boxed{} 内容 (数学任务标准格式)
    start = 0
    while True:
        idx = text.find(r"\boxed{", start)
        if idx == -1: break
        count = 1
        for i in range(idx + 7, len(text)):
            if text[i] == '{': count += 1
            elif text[i] == '}': count -= 1
            if count == 0:
                candidates.append(text[idx + 7:i])
                start = i
                break
        if count != 0: break # 括号不匹配则退出

    # 3. 如果没提取到，取最后一行
    if not candidates:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            candidates.append(lines[-1])
            
    return candidates

# ================= verl 调用的标准函数入口 =================
def compute_score(solution_str, ground_truth, **kwargs):
    """
    verl 调用此函数时会传入 solution_str (模型生成的 response)
    和 ground_truth (数据集中的原答案)
    """
    # 1. 预处理 Ground Truth
    # 数据集里的答案有时也带 \boxed，先尝试提取，提不到就直接归一化
    gt_list = extract_answer(ground_truth)
    gt_val = _normalize_answer(gt_list[0]) if gt_list else _normalize_answer(ground_truth)
    
    # 2. 提取模型回答中的候选答案
    pred_candidates = extract_answer(solution_str)
    
    # 3. 判定逻辑：只要有一个候选答案与 GT 一致，即得 1 分
    for cand in pred_candidates:
        if _normalize_answer(cand) == gt_val:
            return 1.0  # 答案正确
            
    return 0.0  # 答案错误