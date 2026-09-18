# -*- coding: utf-8 -*-
"""
检索器工厂 —— **所有脚本都从这里拿检索器，别再各自 `new` 了**。

━━━ 为什么要有这个文件（不是过度封装）━━━

2026-09-18 实测：`eval_agent.py` 里 **14 处**各自写死

    r = BM25Retriever(labels, use_locator=True)

于是块 E6 把检索层修好（`HybridRetriever`：三路候选 + 自适应融合 + 域外弃权门限）之后，
**agent 侧一行都没变，还在用那个有 bug 的旧实现** —— E7 / E8 的全部数字
都是在"结构先验只能重排序、引不进候选"的检索器上跑出来的。

> 这就是 [[lesson-wired-but-never-called]] 那条：**写了但没接进流程。**
> 它的危险在于**不报错** —— 新代码有测试、有数字，只有主路径没变。
> 而 14 个写死的调用点是**同一个洞的 14 个入口**，改一个漏一个。
> 收成一个工厂之后，"换检索器"是**一处**的事。

━━━ 切换方式 ━━━

    make_retriever(labels)                      # 默认 hybrid
    make_retriever(labels, kind="bm25")         # 老实现（做对照 / 复现历史数字）
    ET_RETRIEVER=bm25 python scripts/xxx.py     # 环境变量切换，方便整脚本对照
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from retrieval.hybrid import HybridRetriever
from retrieval.retriever import BM25Retriever

# 向量缓存的**唯一**位置 —— 散在各脚本里的话，换个脚本就要重编码一遍（40 秒）。
EMB_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "emb"
RERANK_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "rerank_scores.json"


def make_retriever(labels: List[Dict[str, Any]], kind: str = None,
                   dense_cache: Any = None, **kw):
    """
    Args:
        kind: "hybrid"（默认）｜ "bm25"。省略时读环境变量 `ET_RETRIEVER`。
        **kw: 透传给检索器（如 `use_locator=False` 做消融）。

    ⚠️ `HybridRetriever` 的构造会把 1156 个 chunk 编码一遍（CPU 40 秒），
       **但有磁盘缓存**：第二次起是秒级。缓存键含语料指纹，
       语料变了（比如 E5 扩语料）会自动重算 —— 不会静默用错向量。
    """
    kind = (kind or os.environ.get("ET_RETRIEVER") or "hybrid").lower()
    if kind == "bm25":
        return BM25Retriever(labels, **kw)
    if kind == "hybrid":
        kw.setdefault("dense_cache", dense_cache if dense_cache is not None else EMB_CACHE)
        # 重排分数也落盘：否则每次重建检索器都要重跑一遍交叉编码器（实测 60 条题 121 秒）。
        kw.setdefault("rerank_cache", RERANK_CACHE)
        kw.setdefault("verbose", False)
        return HybridRetriever(labels, **kw)
    raise ValueError(f"未知的检索器类型 {kind!r}（可选：hybrid / bm25）")


__all__ = ["make_retriever", "EMB_CACHE"]
