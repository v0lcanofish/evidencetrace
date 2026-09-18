# -*- coding: utf-8 -*-
"""
块 E6 · 交叉编码器重排 —— 融合负责**召回**，重排负责**排序**。

    from retrieval.rerank import CrossReranker

━━━ 为什么值得加（先量了才加的，不是"别人都有所以我也加"）━━━

探针 test 60 上，三路融合结果的 Recall@k 曲线：

    @1  0.433     @5  0.717  ← 现在交付给 agent 的就到这里
    @10 0.867     @20 0.933
    @50 1.000

**@5 和 @20 之间差 0.216** —— gold 已经在候选里了，只是名次不够。
这正是重排能干的活。（如果 @20 ≈ @5，说明漏在**召回**，那该动的是候选生成，重排白搭。）

━━━ 为什么它是"重排"而不是"再加一路候选"━━━

    BM25 / 稠密   双塔式 —— query 和 passage **各自编码**，从没见过面。
                  所以它能从 1156 个块里快速捞出 100 个，但**分不清那 100 个谁更对**。
    交叉编码器     query 和 passage **拼在一起**过一遍模型 —— 慢，但能看清细节。

⚠️ 所以它**必须在候选已经收窄之后才用**（这里是 top-20）。
   对全库跑交叉编码器 = 1156 次前向/查询，CPU 上不可能。

━━━ 缓存 ━━━

矩阵是 O(查询数 × 候选数) 次前向。本地 CPU 实测约 50ms/对 →
60 条题 × 20 候选 = 1200 对 ≈ 1 分钟。评测会反复跑，所以**结果落盘**：
键 = (query, chunk_id)，语料变了 chunk_id 会变，**不会串**。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class CrossReranker:
    """
    交叉编码器重排。

    Args:
        model_name: 默认 bge-reranker-base（278M 参数）。
                     ⚠️ 实测在本机 CPU 上**能用但不算快** —— 所以千万别拿它筛全库。
        top_n:      只对融合结果的**前 N 个**重排，剩下的排在后面（保持相对顺序）。
                     N=20 的依据见模块 docstring 的 Recall@k 曲线。
        batch_size: CPU 上 8 比较稳（再大内存涨、速度不涨）。
        max_length: 512。chunk 中位 ~110 token，512 是模型的训练长度，不截断更安全。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 top_n: int = 20,
                 batch_size: int = 8,
                 max_length: int = 512,
                 cache_path: Optional[Path] = None,
                 verbose: bool = False):
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self._tok = None
        self._model = None
        self._torch = None

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                # ⚠️ 缓存坏了就丢掉重算 —— **不要让一个坏缓存把整条评测卡死**。
                #    但也不静默：说一声。
                print(f"      [rerank cache] ⚠️ {self.cache_path.name} 读不出来，忽略并重建")
                self._cache = {}
        self._dirty = False

    # ------------------------------------------------------------ 模型
    def _lazy_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name).eval()
            self._torch = torch
        return self._tok, self._model

    @staticmethod
    def _key(query: str, chunk_id: str) -> str:
        h = hashlib.sha1()
        h.update(query.encode("utf-8"))
        h.update(b"\x1f")                 # 分隔符：不加的话 ("ab","c") 和 ("a","bc") 会撞
        h.update(chunk_id.encode("utf-8"))
        return h.hexdigest()[:20]

    def score_pairs(self, query: str,
                    items: Sequence[Tuple[str, str]]) -> List[float]:
        """items = [(chunk_id, text), ...] → 每个的分数（越大越相关）。"""
        todo = [(cid, txt) for cid, txt in items
                if self._key(query, cid) not in self._cache]
        if todo:
            tok, model = self._lazy_model()
            torch = self._torch
            out: List[float] = []
            with torch.no_grad():
                for i in range(0, len(todo), self.batch_size):
                    batch = todo[i:i + self.batch_size]
                    enc = tok([query] * len(batch), [t for _c, t in batch],
                              padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")
                    logits = model(**enc).logits.view(-1)
                    out.extend(float(x) for x in logits)
            for (cid, _t), sc in zip(todo, out):
                self._cache[self._key(query, cid)] = sc
            self._dirty = True
            if self.verbose:
                print(f"      [rerank] 新算 {len(todo)} 对（缓存命中 "
                      f"{len(items) - len(todo)}）")
        return [self._cache[self._key(query, cid)] for cid, _t in items]

    def flush(self):
        """把新算的分数写盘。⚠️ 由调用方决定什么时候调 —— 每条查询都写一次太费。"""
        if self.cache_path and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            self._dirty = False

    # ------------------------------------------------------------ 重排
    def rerank(self, query: str,
               ranked: List[int],
               chunks: Sequence[Any],
               allowed: Optional[Sequence[int]] = None) -> List[int]:
        """
        对候选重排，**原有的相对顺序作为兜底**。

        Args:
            allowed: ⭐⭐ **已知的硬约束** —— 只在这些下标之间重排，
                     其余保持在后面且相对顺序不变。**这个参数是这块的关键。**

        ⭐⭐ 为什么必须有 `allowed`（2026-09-18 实测，这一条是本块最值钱的发现）：

            不加约束（纯全局重排）实测**是负收益**：

                检索集 60    融合 1.000 → 重排 **0.850**（−0.150）
                探针   test  融合 0.717 → 重排 **0.700**（−0.017）

            查了失败样例，机制很清楚：

                Q: How much allopurinol should I take?
                   重排前 top-5 药名: [allopurinol ×5]
                   重排后 top-5 药名: [rosuvastatin, risperidone, allopurinol, …]
                60 条里 **46 条** 重排后混进了**别的药**的块。

            **交叉编码器只看「这段文字像不像在回答『该吃多少』」——
              它看不出「这段文字讲的是不是这个药」。**
            ⇒ 它把**药名当成了软信号**（语义相似度的一部分），
              而在这个任务里，**药名是硬约束**。

            这和块 E8 那条教训是同一个：
              **事实问题进硬判据，程度问题进分数。**
              「是不是这份药」是**事实**，不该交给一个只会打相似度分的模型去猜。

            把硬约束喂进去之后（只在定位到的药/章节内重排）：

                探针 test    0.717 → **0.783**（+0.066）
                探针 dev     0.700 → **0.833**（+0.133）
                检索集       1.000 → **1.000**（不掉）

            ⇒ **重排没错，错的是让它去猜它猜不到的东西。**
        """
        if not ranked:
            return ranked
        if allowed:
            head = [i for i in ranked if i in set(allowed)][:self.top_n]
        else:
            head = ranked[:self.top_n]
        tail = [i for i in ranked if i not in set(head)]

        if len(head) <= 1:
            return ranked          # 约束内只有一块，重排没有意义 —— 别白跑一遍模型

        items = [(chunks[i].chunk_id, chunks[i].text) for i in head]
        scores = self.score_pairs(query, items)
        # ⚠️ 排序键带原排名做 tie-break：分数打平时保持融合的名次，
        #    否则同一个分数下顺序取决于 Python 的排序实现，**不可复现**。
        order = sorted(range(len(head)), key=lambda j: (-scores[j], j))
        return [head[j] for j in order] + tail


__all__ = ["CrossReranker"]
