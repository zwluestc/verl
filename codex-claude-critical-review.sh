#!/usr/bin/env bash

set -Eeuo pipefail

# Codex -> Codex/Claude critical review -> Codex fix
# Run this script from anywhere inside a clean Git repository.

if [[ $# -lt 1 ]]; then
  echo '用法：'
  echo '  codex-claude-critical-review.sh "你的需求"'
  exit 2
fi

TASK="$*"

if ! command -v git >/dev/null 2>&1; then
  echo "错误：找不到 git。"
  exit 1
fi
if ! command -v codex >/dev/null 2>&1; then
  echo "错误：找不到 codex，请先确认 Codex CLI 已安装并登录。"
  exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "错误：找不到 claude，请先确认 Claude Code CLI 已安装并登录。"
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "错误：当前目录不在 Git 仓库内。"
  exit 1
fi
cd "$ROOT"

# Keep workflow artifacts and an untracked copy of this script out of Git status.
mkdir -p "$ROOT/.git/info" "$ROOT/.ai-review"
touch "$ROOT/.git/info/exclude"
grep -qxF '.ai-review/' "$ROOT/.git/info/exclude" 2>/dev/null || \
  echo '.ai-review/' >> "$ROOT/.git/info/exclude"

SCRIPT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
case "$SCRIPT_ABS" in
  "$ROOT"/*)
    SCRIPT_REL="${SCRIPT_ABS#"$ROOT"/}"
    if ! git ls-files --error-unmatch -- "$SCRIPT_REL" >/dev/null 2>&1; then
      grep -qxF "$SCRIPT_REL" "$ROOT/.git/info/exclude" 2>/dev/null || \
        echo "$SCRIPT_REL" >> "$ROOT/.git/info/exclude"
    fi
    ;;
esac

if [[ -n "$(git status --porcelain)" ]]; then
  echo "错误：工作区开始前必须干净，避免把旧修改混入本次审查。"
  echo
  git status --short
  echo
  echo "请先提交，或执行 git stash -u；完成后再运行本脚本。"
  exit 1
fi

RUN_ID="$(date '+%Y%m%d-%H%M%S')"
RUN_DIR="$ROOT/.ai-review/$RUN_ID"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_ID" "$ROOT/.ai-review/latest"
LOG_FILE="$RUN_DIR/workflow.log"
CURRENT_STAGE="初始化"

# Terminal formatting. Falls back cleanly when output is not a TTY.
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'
  BLUE=$'\033[34m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  RED=$'\033[31m'
  RESET=$'\033[0m'
else
  BOLD=''; BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

say() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

stage() {
  CURRENT_STAGE="$1"
  say ""
  say "${BOLD}${BLUE}============================================================${RESET}"
  say "${BOLD}${BLUE}[$(date '+%H:%M:%S')] $1${RESET}"
  say "${BOLD}${BLUE}============================================================${RESET}"
}

on_error() {
  local code=$?
  say ""
  say "${BOLD}${RED}流程失败：${CURRENT_STAGE}（退出码 $code）${RESET}"
  say "日志：$LOG_FILE"
  exit "$code"
}
trap on_error ERR

cat > "$RUN_DIR/task.md" <<EOF
# 原始需求

$TASK
EOF

stage "STEP 0/3：环境检查"
say "项目：$ROOT"
say "Codex：$(codex --version 2>/dev/null | head -n 1 || echo 已安装)"
say "Claude：$(claude --version 2>/dev/null | head -n 1 || echo 已安装)"
say "运行目录：$RUN_DIR"
say "审查策略：只检查影响结果的重大错误；忽略风格、命名、排版和非阻塞优化。"

stage "STEP 1/3：Codex 实现需求"
{
  codex exec \
    --cd "$ROOT" \
    --sandbox workspace-write \
    --output-last-message "$RUN_DIR/01-codex-implementation.md" \
    - <<EOF
你是本次任务的实现者。请在当前 Git 仓库实现以下需求：

$TASK

执行约束：
1. 先阅读 AGENTS.md、README、相关源码和测试，只探索与需求直接相关的区域。
2. 只做满足需求所需的最小修改，不进行无关重构、格式化或依赖升级。
3. 保持现有公开接口和行为兼容，除非需求明确要求改变。
4. 补充或修改必要测试；优先运行与改动直接相关的最小测试集。
5. 不执行 git commit、git push、git reset、git clean 或删除用户文件。
6. 完成后用简短 Markdown 汇报：修改文件、核心行为、测试结果、已知风险。
EOF
} 2>&1 | tee "$RUN_DIR/01-codex-execution.log"

if [[ -z "$(git status --porcelain)" ]]; then
  say "${YELLOW}警告：Codex 没有产生任何工作区修改。仍继续审查其结果。${RESET}"
fi

git status --short > "$RUN_DIR/changed-files-after-step1.txt"
git diff --stat > "$RUN_DIR/diff-stat-after-step1.txt"

stage "STEP 2A/3：Codex 独立审查重大错误（只读）"
{
  codex exec \
    --cd "$ROOT" \
    --sandbox read-only \
    --output-last-message "$RUN_DIR/02a-codex-review.md" \
    - <<EOF
你是独立代码审查员。原始需求在：
$RUN_DIR/task.md

请只读审查当前工作区相对 HEAD 的修改。先看 git diff --stat、git diff --name-only 和相关 diff；只在必要时读取调用方或类型定义。

只报告会明显影响最终结果的阻塞性问题：
- 实现不满足原始需求或核心逻辑错误
- 会导致崩溃、数据损坏、严重状态不一致或可靠复现的错误
- 明确的安全漏洞或权限绕过
- 破坏公开接口、协议、存储格式或关键兼容性
- 测试/构建必然失败，或关键路径完全没有覆盖且很可能出错

必须忽略：代码风格、命名、注释、文档措辞、格式化、轻微性能、重复代码、纯理论边界、非阻塞优化和个人偏好。
不要修改任何文件。最多报告 5 个问题；没有重大问题就只输出 PASS。

每个问题只用以下四行：
- 严重度：CRITICAL 或 HIGH
- 位置：文件:行号
- 影响：具体失败场景
- 修复：最小修复方向
EOF
} 2>&1 | tee "$RUN_DIR/02a-codex-review.log"

stage "STEP 2B/3：Claude 独立审查重大错误（只读）"
CLAUDE_PROMPT="$(cat <<EOF
你是独立代码审查员。原始需求在：$RUN_DIR/task.md

只读审查当前工作区相对 HEAD 的修改。先看 git diff --stat、git diff --name-only 和相关 diff；只在必要时读取调用方或类型定义。

只报告会明显影响最终结果的阻塞性问题：
- 实现不满足原始需求或核心逻辑错误
- 会导致崩溃、数据损坏、严重状态不一致或可靠复现的错误
- 明确的安全漏洞或权限绕过
- 破坏公开接口、协议、存储格式或关键兼容性
- 测试/构建必然失败，或关键路径完全没有覆盖且很可能出错

必须忽略：代码风格、命名、注释、文档措辞、格式化、轻微性能、重复代码、纯理论边界、非阻塞优化和个人偏好。
不要修改任何文件。最多报告 5 个问题；没有重大问题就只输出 PASS。

每个问题只用以下四行：
- 严重度：CRITICAL 或 HIGH
- 位置：文件:行号
- 影响：具体失败场景
- 修复：最小修复方向
EOF
)"

{
  claude \
    --print \
    --permission-mode plan \
    --model "${CLAUDE_REVIEW_MODEL:-sonnet}" \
    --effort "${CLAUDE_REVIEW_EFFORT:-medium}" \
    --max-turns "${CLAUDE_REVIEW_MAX_TURNS:-8}" \
    --no-session-persistence \
    --output-format text \
    --allowedTools "Read,Grep,Glob,Bash(git status *),Bash(git diff *),Bash(git show *),Bash(git log *)" \
    "$CLAUDE_PROMPT"
} 2>&1 | tee "$RUN_DIR/02b-claude-review.md"

stage "STEP 3/3：Codex 验证审查结果并修复"
{
  codex exec \
    --cd "$ROOT" \
    --sandbox workspace-write \
    --output-last-message "$RUN_DIR/03-codex-fix-summary.md" \
    - <<EOF
你是最终修复者。原始需求及两份独立审查报告位于：
- $RUN_DIR/task.md
- $RUN_DIR/02a-codex-review.md
- $RUN_DIR/02b-claude-review.md

请执行：
1. 逐条核实两份报告，不要盲目接受；同一问题合并处理。
2. 只修复确认成立、会影响正确性/安全性/数据/兼容性的 CRITICAL 或 HIGH 问题。
3. 对风格、命名、文档措辞、轻微性能或非阻塞建议不要改。
4. 保持修改范围最小，并补充能防止回归的必要测试。
5. 优先运行相关的最小测试集、类型检查或构建检查；不要无理由运行昂贵的全量测试。
6. 不执行 git commit、git push、git reset、git clean 或删除用户文件。
7. 在项目根目录创建或更新 CODE_STRUCTURE.md，保持简洁，必须包含：
   - 项目/本次功能概述
   - 与本次功能相关的目录树（不要输出 node_modules、.git、缓存或无关目录）
   - 本次新增/修改文件及职责
   - 关键调用链或数据流
   - 相关测试文件与运行命令
8. 最终简短汇报：接受并修复的问题、拒绝的问题及原因、测试结果、剩余风险。
EOF
} 2>&1 | tee "$RUN_DIR/03-codex-fix-execution.log"

stage "完成：输出结果"

git diff --check | tee "$RUN_DIR/git-diff-check.txt"
git status --short | tee "$RUN_DIR/final-git-status.txt"
git diff --stat | tee "$RUN_DIR/final-diff-stat.txt"

if [[ ! -f "$ROOT/CODE_STRUCTURE.md" ]]; then
  say "${YELLOW}警告：Codex 未生成 CODE_STRUCTURE.md，正在生成基础版。${RESET}"
  {
    echo '# Code Structure'
    echo
    echo '## 本次需求'
    echo
    echo "$TASK"
    echo
    echo '## 修改文件'
    echo
    echo '```text'
    git status --short
    echo '```'
    echo
    echo '## 相关目录（最多三层）'
    echo
    echo '```text'
    find . -maxdepth 3 \
      \( -path './.git' -o -path './.ai-review' -o -path './node_modules' -o -path './.venv' -o -path './venv' -o -path './dist' -o -path './build' \) -prune \
      -o -type f -print | sed 's#^./##' | sort | head -n 300
    echo '```'
  } > "$ROOT/CODE_STRUCTURE.md"
fi

say ""
say "${BOLD}${GREEN}三阶段流程已完成。${RESET}"
say "代码结构：$ROOT/CODE_STRUCTURE.md"
say "Codex 审查：$RUN_DIR/02a-codex-review.md"
say "Claude 审查：$RUN_DIR/02b-claude-review.md"
say "最终修复摘要：$RUN_DIR/03-codex-fix-summary.md"
say "完整日志：$LOG_FILE"
say "最新运行快捷路径：$ROOT/.ai-review/latest"
say ""
say "请人工检查 git diff 后再决定是否提交。"
