#!/usr/bin/env bash
# ==============================================================================
# 用**真模型**跑全部评测，产出"真数字"（README §五 挂了很久的那一项）。
#
#   PY=/path/to/python3.13 bash scripts/run_real_evals.sh
#
# ━━━ 为什么是脚本而不是手打几条命令 ━━━
#
# 1. **可复现**：哪次跑用了真模型、跑了哪些脚本，一条命令写死在这里。
# 2. **缓存兜底**：`agent/llm.py` 的 LLMCache 命中就不联网 —— 重跑**不花钱**，
#    而且保证同一 prompt 结果一致。所以这个脚本跑第二遍是免费的。
# 3. **落盘**：每个脚本的输出进 `reports/real/<脚本>.txt`，和 mock 那批分开存，
#    免得又出现"这个数字是 mock 的还是真模型跑的"分不清。
#
# ⚠️ 会**覆盖**两个文件，跑之前先备份（脚本会自动备）：
#      data/decisions.jsonl        （eval_decision_signal 的产物）
#      reports/coverage_eval.json  （eval_coverage 的产物）
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export ET_REAL_LLM=1                 # ← 唯一的开关，见 agent/generator.py

PY="${PY:-python}"
OUTDIR="reports/real"
mkdir -p "$OUTDIR"

# ---- 备份会被覆盖的产物（只备第一次，别把真数字备份覆盖掉）----
[ -f data/decisions.jsonl ] && [ ! -f data/decisions.mock.bak.jsonl ] \
    && cp data/decisions.jsonl data/decisions.mock.bak.jsonl
[ -f reports/coverage_eval.json ] && [ ! -f reports/coverage_eval.mock.bak.json ] \
    && cp reports/coverage_eval.json reports/coverage_eval.mock.bak.json

n_cache_before=$("$PY" -c "import json;from pathlib import Path;p=Path('data/cache/llm_cache.json');print(len(json.loads(p.read_text(encoding='utf-8'))) if p.exists() else 0)")

echo "=============================================================================="
echo "真模型评测 ｜ 生成器 = deepseek-chat ｜ 缓存起点 $n_cache_before 条"
echo "输出 → $OUTDIR/"
echo "=============================================================================="

FAILED=()
for s in eval_select eval_decision_signal eval_attribution eval_coverage \
         eval_session eval_verify eval_consistency; do
    echo
    echo "▶ $s"
    if "$PY" "scripts/$s.py" > "$OUTDIR/$s.txt" 2>&1; then
        echo "   ✓ 退出码 0 → $OUTDIR/$s.txt"
        tail -3 "$OUTDIR/$s.txt" | sed 's/^/     /'
    else
        echo "   ❌ 失败（退出码 $?）—— 见 $OUTDIR/$s.txt"
        tail -6 "$OUTDIR/$s.txt" | sed 's/^/     /'
        FAILED+=("$s")
    fi
done

n_cache_after=$("$PY" -c "import json;from pathlib import Path;p=Path('data/cache/llm_cache.json');print(len(json.loads(p.read_text(encoding='utf-8'))) if p.exists() else 0)")

echo
echo "=============================================================================="
echo "跑完 ｜ 缓存 $n_cache_before → $n_cache_after 条（新增 $((n_cache_after - n_cache_before)) 次真实调用）"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ 失败：${FAILED[*]}"
    exit 1
fi
echo "✅ 全部完成。真数字在 $OUTDIR/ —— 和 mock 那批（reports/*.txt）分开存，别混。"
echo "=============================================================================="
