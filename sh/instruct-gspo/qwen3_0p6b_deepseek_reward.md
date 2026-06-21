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

## 代码级问题（不用 API 时的具体风险）

### 问题 1：`_normalize()` 会破坏分数表达式（高风险）

`_normalize()` 的处理顺序是：先把 `\dfrac` 替换成 `\frac`，最后把所有 `{` `}` 删掉（L387）。
结果：`\frac{1}{2}` → `\frac12`，而不是 `1/2` 或 `0.5`。

```
# 模型答案用斜线：1/2   → _normalize → "1/2"
# gt 用 LaTeX 分数：\frac{1}{2} → _normalize → "\frac12"
# 两者不等 → 精确匹配失败
```

字符串精确匹配对不同写法的分数完全失效，**唯一兜底是 sympy**。
一旦 sympy 未安装或解析失败，分数形式的答案就会误判为错。

### 问题 2：含 `=`/`<`/`>` 的答案绕过 sympy（中风险）

`_parse_sympy_expr()` 第 507 行：

```python
if re.search(r"[<>=]", expr_text):
    return None
```

任何含等号的表达式（如 `x = 3`、区间 `x ≥ 1`）都不走 sympy。
这时只靠 `_normalize()` 做字符串比较——空格变化、`\,` 等细微差异都会导致匹配失败，
且对不同但等价的写法（`x=3` vs `x = 3`）依赖 normalize 去空格，而对 `3` vs `x=3` 这种情况完全无法判断。

### 问题 3：sympy 不可用时覆盖率大幅下降（高风险）

代码用 `try/except` 导入 sympy（L63–76），失败时 `_sympy = None`，`_sympy_equal` 始终返回 `False`。
此时对 LOCAL_RULE_ONLY_SOURCES 只剩两条路：

- 精确字符串匹配（形式敏感，如上述分数问题）
- 纯数值比较（仅 `[-+]?\d+(\.\d*)?(e...)?%?` 格式，不含任何字母）

`\sqrt{2}/2`、`\pi/4`、`e^{i\pi}+1` 等所有符号答案在 sympy 缺失时全部返回 0.0。

### 问题 4：WildSci 选项提取有退路但不保险（低-中风险）

`_extract_choice()` 提取逻辑：
1. 先走 `_extract_official_answer()` 拿到"最终答案"
2. 对结果做 `fullmatch(r"\(?\s*([A-Z])\s*\)?")` 或 `search(r"answer/option/choice is X")`

若模型输出长推理链，`_extract_official_answer` 会返回完整原文（无 `<final>` / `\boxed{}` / `####` 时的兜底），
这时 fullmatch 大概率失败，keyword search 也可能找不到。
`_extract_choice` 返回 `""`，分支 `if pred_choice and gt_choice` 不成立，
回退到 normalized 精确匹配——对 MCQ 来说效果极差。

### 问题 5：角度单位不一致（中风险，SciInstruct/WildSci）

`_normalize()` 第 385 行：`^\\circ` 被直接删除（`30°` → `30`）。
但 `\pi/6` 经过 normalize 后仍为 `\pi/6`，与 `30` 不等；
sympy 也不做度-弧度转换，`sympy.simplify(30 - pi/6)` ≠ 0。
含角度的物理/科学题若答案一侧用度、另一侧用弧度，**本地规则无法识别等价**。

### 问题 6：`_extract_official_answer` 对 gt 的兜底返回原始全文（中风险）

若 ground_truth 不含 `<final>`、`\boxed{}`、`####`、"answer is" 等标志，
函数最终返回完整原始字符串（L427 `return raw`）。
一旦 gt 原文很长，normalized 比较和 sympy 解析均会失败，
LOCAL_RULE_ONLY_SOURCES 中直接返回 0.0，形成**系统性漏判**。

---

## 各数据集风险汇总

| 数据集 | 主要答案类型 | 核心风险 | 整体评估 |
|--------|------------|---------|---------|
| **WildSci** | 选择题 A/B/C/D | 推理链过长导致选项提取失败；角度单位不一致 | 基本适合，MCQ 格式规范时可靠 |
| **DeepMath** | 数值、多项式、根式 | 分数写法不一致（`\frac` vs `/`）；sympy 不可用时大量漏判 | 依赖 sympy，有风险 |
| **Metamath** | 形式化证明结论 | gt 格式若非标准 boxed/tagged，兜底返回全文必然失败 | 风险较高，需确认 gt 格式 |
| **OpenR1Math** | 竞赛数值/分数/根式 | 同 DeepMath；含 `=` 的方程解跳过 sympy | 基本适合，纯数值答案可靠 |
| **SciInstruct** | 物理公式、数值 | 角度单位 + 含等号表达式 + 复杂物理量符号解析 | 风险较高，复合公式易漏判 |

## 环境变量

| 变量 | 作用 | 默认值 |
|------|------|-------|
| `LLM_API_KEY` / `DEEPSEEK_API_KEY` | API 鉴权 Key | 无（必填，否则跳过 LLM Judge） |
| `LLM_API_BASE_URL` / `BASE_URL` | API Base URL | `https://api.deepseek.com` |
| `LLM_JUDGE_MODEL` / `DEEPSEEK_MODEL` | 判分模型名 | `deepseek-v4-flash` |
| `LLM_JUDGE_MAX_TOKENS` | Judge 响应最大 token 数 | `128` |
| `REWARD_DEBUG_LOG` | 调试日志路径（JSONL） | 不写日志 |
| `REWARD_DEBUG_LIMIT` | 调试日志最大条数 | `1000` |
