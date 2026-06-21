# Reward Scoring Logic — `qwen3_0p6b_deepseek_reward.py`

## 主流程 `compute_score()`

```
预测文本
  │
  ├─ 提取答案：<final>…</final> 或 \boxed{}  → max_reward = 1.0
  └─ 都没有：取末尾 500 字符                 → max_reward = 0.8（惩罚未遵循格式）
        └─ 还是空 → 直接返回 0.0

ground_truth → _extract_official_answer()

↓
Step 1: _local_rule_score()（纯规则，无 API）
  ├─ wildsci 特判：先走选择题字母匹配 (A/B/C/D)
  ├─ normalized 精确匹配（去 LaTeX 符号、空格、大小写等）
  ├─ 数值等价（浮点，相对误差 1e-6）
  ├─ 多元组拆分匹配（逗号/分号分隔，顺序无关）
  └─ SymPy 符号化等价（化简后差为 0）

↓ 规则得分 > 0 → 直接返回（乘以 max_reward）

↓ 规则得分 = 0
Step 2: 检查是否在 LOCAL_RULE_ONLY_SOURCES
  {"deepmath", "metamath", "openr1math", "sciinstruct", "wildsci"}
  └─ 在集合里 → 直接返回 0.0，不调 API

↓ 不在集合里
Step 3: LLM Judge（DeepSeek API）→ 返回 0/1
↓ API 失败
Step 4: 兜底精确匹配
```

## 各数据集适配分析（不用 API）

对于 `LOCAL_RULE_ONLY_SOURCES` 里的五个数据集，代码**硬性关闭了 LLM Judge API 调用**，完全依赖本地规则判分。

| 数据集 | 答案类型 | 本地规则覆盖率 | 适合程度 |
|--------|---------|---------------|---------|
| **WildSci** | 选择题 (A/B/C/D) | 高：专门写了 `_extract_choice` 逻辑 | 适合 |
| **DeepMath** | 数学竞赛，答案多为数值或简单解析式 | 中高：numeric + sympy 能覆盖大多数 | 基本适合 |
| **Metamath** | 形式化证明/定理，答案格式极规范 | 中：标准形式精确匹配率高，复杂符号式 sympy 可能解析失败 | 部分适合 |
| **OpenR1Math** | 数学题，答案多为数值/分数/根式 | 中高：numeric + sympy 覆盖主流，复杂表达式有盲区 | 基本适合 |
| **SciInstruct** | 科学计算，含数值和公式 | 中：依赖 numeric + sympy，复杂物理公式可能漏判 | 部分适合 |

## 潜在盲区

1. **复杂符号式解析失败**：如 `\sin^2 x + \cos^2 x` vs `1` 这类恒等式，`_latex_to_sympy_text` 转换失败时 sympy 无法判断，最终落到精确匹配，**误判为错**。

2. **Metamath/DeepMath 高阶表达式**：如 $\frac{\sqrt{2}}{2} \cdot e^{i\pi}$，LaTeX 解析链条较脆，parse 抛异常后直接返回 0.0。

3. **WildSci 非选择题**：若混有填空题，choice 字母提取不到后退回 normalized 匹配，精度下降。

4. **多元组顺序假设**：`_split_top_level` 做无序匹配，但依赖逗号/分号为分隔符；若 ground truth 用其他分隔符，拆分会失效。

## 环境变量

| 变量 | 作用 | 默认值 |
|------|------|-------|
| `LLM_API_KEY` / `DEEPSEEK_API_KEY` | API 鉴权 Key | 无（必填，否则跳过 LLM Judge） |
| `LLM_API_BASE_URL` / `BASE_URL` | API Base URL | `https://api.deepseek.com` |
| `LLM_JUDGE_MODEL` / `DEEPSEEK_MODEL` | 判分模型名 | `deepseek-v4-flash` |
| `LLM_JUDGE_MAX_TOKENS` | Judge 响应最大 token 数 | `128` |
| `REWARD_DEBUG_LOG` | 调试日志路径（JSONL） | 不写日志 |
| `REWARD_DEBUG_LIMIT` | 调试日志最大条数 | `1000` |
