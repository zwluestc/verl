import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="查看指定题目的8次采样回答及提取的答案")
    parser.add_argument("n", type=int, help="要查看的题目序号 (例如 1 代表第一题)")
    parser.add_argument("--file", type=str, default="sh/output.jsonl", help="Jsonl结果文件路径")
    parser.add_argument("--full", action="store_true", help="是否打印完整的模型回答 (默认只打印最后1000字防止刷屏)")
    args = parser.parse_args()

    if args.n < 1:
        print("题目序号必须大于等于 1")
        sys.exit(1)

    target_idx = args.n - 1
    item = None

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == target_idx:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"❌ 第 {args.n} 行数据损坏 (JSONDecodeError)，可能是生成时被截断。")
                        sys.exit(1)
                    break
            else:
                print(f"查无此题。文件中共只有 {i+1} 行数据。")
                sys.exit(1)
    except FileNotFoundError:
        print(f"找不到文件: {args.file}")
        sys.exit(1)

    # 打印原题信息
    doc = item.get("doc", {})
    input_text = doc.get("input", "")
    target_text = doc.get("output", doc.get("target", ""))
    
    print(f"==================================================")
    print(f"                 第 {args.n} 题 详情")
    print(f"==================================================")
    print(f"【原题输入 (前500字)】:\n{input_text[:500]}{'...' if len(input_text)>500 else ''}\n")
    print(f"【原题标答文本 (后500字)】:\n...{target_text[-500:]}\n")
    
    resps = item.get("resps", [[]])[0]
    filtered_resps = item.get("filtered_resps", [[]])
    if isinstance(filtered_resps, list) and len(filtered_resps) > 0 and isinstance(filtered_resps[0], list):
        filtered_resps = filtered_resps[0] # lm-eval有可能是嵌套列表

    print(f"本题共进行了 {len(resps)} 次采样生成。\n")

    for i in range(len(resps)):
        print(f"------------------- 采样 {i+1} -------------------")
        
        # 打印模型原始回答
        ans = resps[i]
        if args.full or len(ans) <= 1000:
            print(f"【模型原始生成】:\n{ans}")
        else:
            print(f"【模型原始生成 (截断后1000字)】:\n...{ans[-1000:]}")
            
        # 打印经过 filter 后提取的简短答案 (处理部分lm-eval版本将过滤后结果聚合的情况)
        extracted = None
        if isinstance(filtered_resps, list) and i < len(filtered_resps):
            extracted = filtered_resps[i]
        elif isinstance(filtered_resps, list) and len(filtered_resps) == 1:
            # 有时 take_first 这个 filter 会把所有结果折叠成一个，只保留第一个
            extracted = filtered_resps[0] if i == 0 else "同上/被聚合"
            
        print(f"\n【提取出的答案】: >>> {extracted} <<< \n")

if __name__ == "__main__":
    main()
