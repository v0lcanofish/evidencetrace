# -*- coding: utf-8 -*-
"""
块 E2 自测 —— 端到端跑通「提问 → 查资料 → 写答案 → 审计」。

四条判据：
  ① 一条命令跑出 report.md + audit.json
  ② ⭐ 账本-报告闭包断言通过（追不到出处就报错）
  ③ 断网可跑（mock 不碰网络）
  ④ 同 seed 跑两次，账本 JSON 完全一致（可复现）

外加两条"断言本身是活的"的验证：
  ⑤ 故意给坏引用 → 闭包必须拦住
  ⑥ 证据不足 → 必须说"答不了"，不能瞎编

跑法（纯 CPU，零 GPU）：
  cd 代码库/projects/EvidenceTrace
  PYTHONIOENCODING=utf-8 D:/anaconda/python.exe scripts/run_mock.py
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))


from agent.research_agent import ResearchAgent, AgentConfig, CITE_RE   # noqa: E402
from agent.mocks import MockRetriever, MockLLM                         # noqa: E402
from ledger.ledger import LedgerError                                  # noqa: E402

OUT_DIR = PROJECT / "data" / "runs"
QUESTION = "Can I take ibuprofen while I'm on warfarin?"


def build(bad_citation: bool = False) -> ResearchAgent:
    return ResearchAgent(
        llm=MockLLM(bad_citation=bad_citation),
        retriever=MockRetriever(),
        config=AgentConfig(max_steps=8, top_k=5, context_budget=3),
    )


def main() -> int:
    print("=" * 78)
    print("块 E2 · 编排引擎自测（零 API / 零 GPU / 断网可跑）")
    print("=" * 78)
    print(f"\n问题：{QUESTION}\n")

    errs = []

    # ---------------- ① 端到端
    print("① 端到端跑一遍")
    print("-" * 78)
    lg = build().run(QUESTION, run_id="mock-1", seed=42)

    print("账本结构：")
    for s in lg.steps:
        extra = ""
        if s.action == "retrieve":
            extra = (f"  拿到 {len(s.retrieved)} 篇 ｜ 进报告={s.used_in_report}"
                     f" ｜ 丢弃原因={s.reason_dropped}")
        print(f"  step {s.step}  {s.action:<10}{extra}")

    print(f"\n报告（{len(lg.claims)} 条带引用的论断）：")
    for line in lg.report_md.split(". "):
        if line.strip():
            print("   " + line.strip()[:96])
    print()

    if not lg.report_md:
        errs.append("没有产出报告")
    if not lg.claims:
        errs.append("没有解析出任何 claim")

    # ---------------- ② 闭包
    print("② 账本-报告闭包断言")
    print("-" * 78)
    try:
        lg.check_closure()
        print("   ✅ 通过：回答里每条引用都能在账本里追到出处")
    except LedgerError as e:
        errs.append(f"闭包失败: {e}")
        print(f"   ❌ {e}")

    # ---------------- ③ 断网
    print("\n③ 断网可跑")
    print("-" * 78)
    import socket
    _orig = socket.socket

    class _NoNet(socket.socket):
        def __init__(self, *a, **kw):
            raise OSError("网络被测试禁用了")

    socket.socket = _NoNet
    try:
        lg_off = build().run(QUESTION, run_id="mock-offline", seed=42)
        print("   ✅ 禁用 socket 后仍然跑通（说明 mock 真没碰网络）")
    except Exception as e:                                     # noqa: BLE001
        errs.append(f"断网跑不通: {type(e).__name__}: {e}")
        print(f"   ❌ {type(e).__name__}: {e}")
    finally:
        socket.socket = _orig

    # ---------------- ④ 可复现
    print("\n④ 同 seed 跑两次，结果是否完全一致")
    print("-" * 78)
    a = build().run(QUESTION, run_id="rep", seed=42).to_dict()
    b = build().run(QUESTION, run_id="rep", seed=42).to_dict()
    if a == b:
        print("   ✅ 完全一致（账本 JSON 逐字节相同）")
    else:
        errs.append("同 seed 两次结果不一致")
        print("   ❌ 不一致")

    # ---------------- ⑤ 坏引用必须被拦
    print("\n⑤ 故意给坏引用，闭包必须拦住")
    print("-" * 78)
    try:
        build(bad_citation=True).run(QUESTION, run_id="bad", seed=42)
        errs.append("坏引用没被拦住 —— 闭包断言是摆设")
        print("   ❌ 没拦住！")
    except LedgerError as e:
        print(f"   ✅ 拦住了：{str(e).splitlines()[0][:70]}")

    # ---------------- ⑥ 证据不足
    print("\n⑥ 检索不到证据时，不能瞎编")
    print("-" * 78)
    agent = ResearchAgent(llm=MockLLM(), retriever=lambda q, k: [], config=AgentConfig())
    lg_empty = agent.run("What is the price of tea in China?", run_id="empty", seed=42)
    txt = lg_empty.report_md.lower()
    ok = ("insufficient" in txt or "cannot" in txt or "unable" in txt)
    print(f"   {'✅' if ok else '❌'} 回答：{lg_empty.report_md[:90]}")
    if not ok:
        errs.append("证据不足时没有明确说答不了")

    # ---------------- 存盘
    print("\n" + "=" * 78)
    if errs:
        print("❌ 自检未通过：")
        for e in errs:
            print("   -", e)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mock_report.md").write_text(lg.report_md, encoding="utf-8")
    lg.save(OUT_DIR / "mock_ledger.json")

    print("✅ E2 自检全绿")
    print(f"   · 端到端出报告（{len(lg.claims)} 条带引用论断，{len(lg.steps)} 步账本）")
    print("   · 闭包断言通过 ｜ 断网可跑 ｜ 同 seed 两次一致")
    print("   · 坏引用被拦住 ｜ 证据不足不瞎编")
    print(f"\n产物 → {OUT_DIR}/mock_report.md ｜ mock_ledger.json")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
