# -*- coding: utf-8 -*-
"""
块 E6 · 三路候选 + 自适应加权 RRF —— **修掉"结构先验被乘性挂在词面分上"那个洞**。

    from retrieval.hybrid import HybridRetriever

━━━ 修的是什么（2026-09-18 挖出来的真 bug）━━━

原实现（`BM25Retriever.search_detail`）：

    scores[i] = scores.get(i, 0.0) * locate_boost        # ← 乘性

而 `ChunkBM25.search()` 里有一句 `if s > 0` —— **BM25 分为 0 的块压根不进候选榜**。
实测：**平均每题 1063 / 1156 个块（92%）BM25 分为 0**。

于是 0 × 3 = 0：**定位层只能给"BM25 已经捞到的块"重新排序，永远引不进 BM25 漏掉的块**——
而它要修的恰恰就是那些。实测检索集 8 道错题**全部**是这个原因：定位**指对了**章节，
gold 块的 BM25 分是 0，于是定位再准也白搭。

> **教训：乘法加成 = 把强先验挂在弱信号上。**
> 先验要能**引入候选**，不能只做**重排序**。

━━━ 三路候选（每一路都能独立引入）━━━

    S  结构先验   定位章节内的块（**含 BM25 零分块**）—— 医学编码是先验事实
    B  词面      全局 BM25 榜
    D  语义      全局稠密榜（检索/dense.py）

━━━ ⭐ 融合权重是「算」出来的，不是拍的 ━━━

等权 RRF 实测**会把稠密拉低**（探针 test：稠密 0.717 → 等权融合 0.633）。原因：
**词面和语义的强弱不是固定的，取决于查询里有没有判别性实词。**

    检索集里**有实体**的 40 条：BM25 **0.950** ／ 稠密 0.850   ← BM25 赢
    检索集里**没实体**的 20 条：BM25 **0.100** ／ 稠密 0.700   ← 稠密大胜
    探针 test 60：               BM25 0.450 ／ 稠密 0.717

机制很直白：**药名在该药的每个 chunk 里都有**，所以只有药名时 BM25 在药内排序**基本是噪声**。

于是权重用 `contrast_p90(q)` —— **BM25 自己的分数分布有多尖**，纯机械量、无自由参数：

    contrast_p90 = (top1 分 − 第 10 名的分) / top1 分

实测分辨力（BM25 自己的 Recall@5）：`<0.2 → 0.22` ｜ `0.2~0.5 → 0.53` ｜ `≥0.5 → 0.95`。

> 这和项目里覆盖度那条是同一个思路：**"够不够"要算出来，不能让模型自己感觉。**
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from retrieval.chunker import Chunk, build_chunks, to_doc
from retrieval.retriever import BM25Retriever, SectionLocator


class HybridRetriever:
    """
    三路候选 + 自适应加权 RRF。

    接口与 `BM25Retriever` **完全兼容**（`__call__` / `search_detail` / `get_chunk` / `locator`），
    所以 agent 侧（`agent/tools.py`）一行都不用改。

    Args:
        use_locator:   ⭐ 消融开关 —— 关掉就是"无结构先验"的世界（探针集量的是这个）
        use_dense:     消融开关
        use_bm25:      消融开关
        section_weight: 结构先验那一路的 RRF 权重。**默认 3.0：强，但不是绝对** ——
            ⚠️ 不设成"限定"（restrict）是有意的：定位**错**的时候限定会把 5 个位置全浪费掉。
               给个强权重，让定位错的章节靠自身融合分自然沉下去。
            ⚠️ **诚实边界**：现有评测集里定位**要么全对（检索集 60/60）、要么全不触发（探针 120）**，
               **没有"定位触发但指错了"的题** → 这个权重在失败模式上的表现**测不出来**，只能靠机制论证。
        rrf_k:         RRF 的 k 常数（20）。列表短（候选百量级），k 取小一点让名次差别有意义。
        cutoff:        每路榜单截断长度（100）。不截的话稠密榜有 1156 项，
                       末尾那些 1/(20+1156) 会把所有块的分差抹平。
        dense_min_cosine: ⭐ **稠密层的弃权门限**，见下面 `_dense_gate` 的长注释。
    """

    def __init__(self, labels: List[Dict[str, Any]],
                 use_locator: bool = True,
                 use_dense: bool = True,
                 use_bm25: bool = True,
                 section_weight: float = 3.0,
                 rrf_k: int = 20,
                 cutoff: int = 100,
                 dense_min_cosine: float = 0.60,
                 use_rerank: bool = True,
                 rerank_model: str = "BAAI/bge-reranker-base",
                 rerank_top_n: int = 20,
                 rerank_cache: Optional[Any] = None,
                 dense_model: str = "BAAI/bge-small-en-v1.5",
                 dense_cache: Optional[Any] = None,
                 target_chars: int = 420,
                 verbose: bool = False):
        self.chunks: List[Chunk] = build_chunks(labels, target_chars=target_chars)
        self.index: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.locator = SectionLocator(labels)
        self._bm = BM25Retriever(labels, use_locator=False, target_chars=target_chars)
        # ⚠️ 分块必须**逐块对齐**：BM25 / 稠密 / 结构先验三路都按下标引用同一个 chunk。
        #    这里断言一下，省得哪天两边参数不一致导致"向量和块错位"——那种错**不报错**，
        #    只是结果悄悄变差。
        assert [c.chunk_id for c in self._bm.chunks] == [c.chunk_id for c in self.chunks], \
            "BM25 与 Hybrid 的分块结果不一致 —— 下标会错位"
        self.bm25 = self._bm.bm25

        self.use_locator, self.use_dense, self.use_bm25 = use_locator, use_dense, use_bm25
        self.section_weight, self.rrf_k, self.cutoff = section_weight, rrf_k, cutoff
        self.dense_min_cosine = dense_min_cosine

        self.dense = None
        if use_dense:
            from retrieval.dense import DenseRetriever
            self.dense = DenseRetriever([c.text for c in self.chunks],
                                        model_name=dense_model,
                                        cache_path=dense_cache, verbose=verbose)

        # ⭐ 重排：**只在已知的硬约束内**做软排序（详见 `_rerank_allowed` 与
        #    `retrieval/rerank.py` 里那段 —— 不限定就是负收益，实测 −0.150）。
        self.reranker = None
        if use_rerank and use_dense:
            from retrieval.rerank import CrossReranker
            self.reranker = CrossReranker(model_name=rerank_model, top_n=rerank_top_n,
                                          cache_path=rerank_cache, verbose=verbose)

        # 定位缓存：search_detail 和 confidence 都会用到，别重复算
        self._q_cache: Dict[str, Any] = {}
        self.last_trace: Dict[str, Any] = {}

        # 向量按需编码（每道新查询一次），进程内缓存
        self._qvec: Dict[str, Any] = {}

    # ------------------------------------------------------------ 向量缓存
    def _dense_scores(self, query: str) -> Dict[int, float]:
        import numpy as np
        v = self._qvec.get(query)
        if v is None:
            v = self.dense.encode_queries([query])[0]
            self._qvec[query] = v
        sims = self.dense.emb @ v
        return {int(i): float(s) for i, s in enumerate(sims)}

    # ------------------------------------------------------------ ⭐ 词面置信度
    def bm25_confidence(self, query: str) -> float:
        """
        `contrast_p90` —— BM25 分数分布有多尖，域 [0, 1]。

        ⚠️ 这不是"调出来的阈值"，是 BM25 分数分布的一个**无量纲描述量**：
           全库分数都一样平 → 0（说明查询词在每个块里都有，毫无判别力）
           有一个块显著高于其它   → 接近 1

        为什么需要它：**药名在该药的每个 chunk 里都出现**，所以"只有药名"的查询，
        BM25 会给出**一堆几乎同分的块** —— 排序在药内是噪声（实测 Recall@5 = 0.100）。
        这一项就是用来**把那种情况认出来，然后把词面那一路的权重降到 0**。
        """
        raw = self.bm25.score_all(query)
        sc = sorted(raw.values(), reverse=True)
        if len(sc) < 5 or sc[0] <= 0:
            return 0.0
        p90 = sc[max(0, len(sc) // 10)]
        return max(0.0, (sc[0] - p90) / sc[0])

    # ------------------------------------------------------------ ⭐ 稠密弃权门限
    def dense_gate(self, query: str) -> Tuple[bool, float]:
        """
        稠密层有没有**任何**相关的东西？返回 (放行?, top1 余弦)。

        ⚠️⚠️ 为什么必须有这个 —— 这是 2026-09-18 换用 Hybrid 时**打断 E7 判据**才暴露的：

            BM25 侧早就修好了"域外问题返回 0 条"（E7：去掉停用词）。
            但**稠密检索永远返回 top-k** —— 余弦相似度对任何两个文本都有定义。
            ⇒ 域外问题 `What is the price of tea in China?` 又拿到 5 条证据，
               **agent 再次观察不到「我什么都没查到」这个信号**，于是硬答。

            E7 的断言当场抓住：
                [!!] 域外问题检索返回 0 条     [!!] 以 abstain 收尾（answer）
            **回归判据就是干这个用的。**

        ⭐ 用**绝对**余弦、不用像 BM25 那样的 contrast：
           实测 contrast 分不开（out_of_scope 0.212 vs 域内 0.217 —— 一模一样），
           而 top1 余弦分得开。原因：**余弦本身就是绝对量纲，BM25 分数不是。**
           ⇒ 同一个思路（"够不够要算"），但**两个检索器要用各自合适的量**。

        门限怎么来的（**不是拍的**，用 `boundary_set` 的 12 条 out_of_scope 标定）：

            out_of_scope 最高 = 0.591（Who won the World Cup in 2022?）
            域内最低        = 0.622
            间隙            = (0.591, 0.622)，取中点 **0.60**

        ⚠️⚠️ **诚实边界**：这条间隙只有 **0.031 宽**，而且域外一侧**只有 12 条样本**。
           换个领域、换个嵌入模型，门限大概率要重标 —— **不能当成通用常数搬走**。
        """
        d = self._dense_scores(query)
        top1 = max(d.values()) if d else 0.0
        return (top1 >= self.dense_min_cosine), top1

    # ------------------------------------------------------------ 三路候选
    def _section_candidates(self, loc: Dict[str, Any]) -> List[int]:
        """
        结构先验路：定位章节里的**全部**块，**含 BM25 分为 0 的**。

        ⭐ 这就是那个 bug 的修法 —— 原来的乘性 boost 永远给不出这一路。

        节内排序：稠密分降序 → BM25 分降序 → 下标升序（**最后一项保证确定性**）。
        """
        loincs = set(loc.get("loincs") or [])
        drug = loc.get("drug")
        if not loincs or not drug:
            return []
        idx = [i for i, c in enumerate(self.chunks)
               if c.drug == drug and c.loinc in loincs]
        if not idx:
            return []
        return idx

    def _rank_section(self, idx: List[int], dsc: Dict[int, float],
                      bsc: Dict[int, float]) -> List[int]:
        return sorted(idx, key=lambda i: (-dsc.get(i, 0.0), -bsc.get(i, 0.0), i))

    # ------------------------------------------------------------ 主入口
    def search_detail(self, query: str, k: int = 5,
                      restrict: Optional[Dict[str, Any]] = None
                      ) -> Tuple[List[Chunk], Dict[str, Any]]:
        loc = self.locator(query) if self.use_locator else {
            "drug": None, "intent": None, "loincs": [], "reason": "定位关闭"}

        # ---- restrict：agent 明确说"就查这一节"。此时**不许丢零分块**
        #      （原来走 score_all() → `if s > 0` → 节里有东西却返回空，
        #        agent 会收到"这节没东西"的**错误信号**。E7 的 docstring 警告过这个坑，
        #        但当时只改了排序、没改候选丢失。）
        if restrict:
            # ⭐⭐ 9/19 改：**drug 维度保持硬过滤，loincs 维度从硬过滤改成加权**
            #
            # 为什么改（实测，隔离了 agent 只测检索本身）：
            #   ① 不限定            → gold 命中 2/2
            #   ② 限定 drug+loincs  → gold 命中 **1/2**  ← 限定反而更差
            #   ③ 只限定 drug       → gold 命中 2/2
            #
            # 病根：`loincs` 硬过滤会**物理挡住**定位器没识别出的章节。
            #   例："I'm taking allopurinol. What's the usual dose, and what other
            #        medicines should I avoid taking with it?"
            #   → 关键词表只有 "take it with"，这句是 "taking with it"，没命中 interaction
            #   → loincs = [禁忌, 剂量]，**相互作用节整个被挡在池子外**
            #   → 而它正是 gold 之一。排序、K、模型能力，全都救不回来。
            #
            # 定位器不可能覆盖所有自然问法 ⇒ 硬过滤**天生脆弱**：漏一个意图就永久失联。
            # 改成加权后：定位到的章节**排在前面**（保留结构先验的价值），
            # 但其它章节仍可达 —— 漏判的代价从"gold 永远拿不到"降到"排得靠后一点"。
            #
            # ⚠️ base 那 60 题从没暴露这个洞：它的 direct/paraphrase **都是同一批模板的变体**，
            #    问法刚好全在关键词表覆盖内。**换评测集 = 换了一次压力测试。**
            # drug 允许是**字符串或列表**：跨药问题（"我在吃 A 和 B"）要把两边的块都留下，
            # 只留一个药会把它自己的 gold 硬过滤掉（实测 cross_drug 那批 gold 进池率 41%）。
            _d = restrict.get("drug")
            drugs = ({_d.lower()} if isinstance(_d, str) else {str(x).lower() for x in (_d or [])})
            loincs = set(restrict.get("loincs") or [])
            sub = [i for i, c in enumerate(self.chunks)
                   if (not drugs or (c.drug or "").lower() in drugs)]
            if not sub:                       # drug 限定到空 → 退回原来的硬口径，别凭空造候选
                sub = list(self._allowed_indices(restrict))
            dsc = self._dense_scores(query) if self.dense else {}
            bsc = self.bm25.score_all(query)

            def _soft_key(i: int):
                c = self.chunks[i]
                return (0 if (loincs and c.loinc in loincs) else 1,   # 定位章节优先
                        -dsc.get(i, 0.0), -bsc.get(i, 0.0), i)        # 同档再按相关度，最后保持原序

            ordered = sorted(sub, key=_soft_key)
            self.last_trace = {"query": query, "mode": "restrict_soft", "locate": loc,
                               "restrict": restrict, "n_candidates": len(ordered), "k": k}
            return [self.chunks[i] for i in ordered[:k]], self.last_trace

        dsc = self._dense_scores(query) if self.dense else {}
        bsc = self.bm25.score_all(query)

        # ---- 三路榜单（各自截断，避免长尾把分差抹平）
        lists: List[Tuple[List[int], float, str]] = []
        if self.use_locator:
            sect = self._section_candidates(loc)
            if sect:
                lists.append((self._rank_section(sect, dsc, bsc), self.section_weight, "sect"))
        conf = 0.0
        if self.use_bm25:
            brank = [i for i, _ in self.bm25.search(query, k=self.cutoff)]
            conf = self.bm25_confidence(query)
            if brank and conf > 0:
                lists.append((brank, conf, "bm25"))

        # ⚠️⚠️ `conf == 0` 时**故意**不把 BM25 榜加进来 —— 这不是漏写，是设计。
        #
        #   conf=0 的含义是「BM25 的分数**完全平坦**」，即查询词在每个块里都有、
        #   毫无判别力（探针里 `What is X supposed to accomplish?` 就是：supposed /
        #   accomplish 语料里根本不出现，只剩药名）。此时 BM25 的 top-5 **是任意的**。
        #
        #   实测（探针 dev 60）：这几道题 BM25 单独用"能命中"，但命中在 rank 4 ——
        #   **那是撞运气，不是检索**。把它当成相关证据喂给 agent，
        #   正是块 E7 抓到的病：**「检索有返回」≠「检索到相关证据」**，
        #   也正是块 E8 里硬答率 0.911 的来源。
        #
        #   ⇒ **没有信号时返回空**，agent 才能观察到「我什么都没查到」这个信号
        #     （E7 那条：分词器把虚词当匹配词 → agent 永远观察不到这个信号 → 域外问题也硬答）。
        #
        #   ⚠️ 副作用：`use_dense=False` 且 conf=0 时**结果为空**。这是正确的，
        #      但会让"③ 三路融合（无稠密）"那一行的 Recall 看起来比 ① 低 —— 报告里要讲清楚。
        dense_ok, dense_top1 = False, 0.0
        if self.use_dense:
            dense_ok, dense_top1 = self.dense_gate(query)
            if dense_ok:
                drank = [i for i, _ in self.dense.search(query, k=self.cutoff)]
                lists.append((drank, 1.0, "dense"))

        # ---- RRF 融合
        sc: Dict[int, float] = {}
        for lst, w, _name in lists:
            for rank, i in enumerate(lst, 1):
                sc[i] = sc.get(i, 0.0) + w / (self.rrf_k + rank)

        # ⚠️ 排序键带上下标：分数打平时按块的下标**固定**排序。
        #    不带的话靠 dict 插入顺序 —— 那是"碰巧确定"，换个 Python 版本就可能变。
        ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        order = [i for i, _ in ranked]

        # ---- ⭐ 重排（**带硬约束**，见 `_rerank_allowed`）
        n_head = 0
        if self.reranker is not None and order:
            pool = order[:max(k, self.reranker.top_n)]
            allowed = self._rerank_allowed(pool, loc)
            n_head = len(allowed)
            reordered = self.reranker.rerank(query, pool, self.chunks,
                                             allowed=allowed or None)
            if reordered != pool:
                order = reordered + order[len(pool):]

        self.last_trace = {
            "query": query, "mode": "fusion", "locate": loc, "restrict": None,
            "layers": {name: len(lst) for lst, _w, name in lists},
            "weights": {name: w for _lst, w, name in lists},
            "bm25_confidence": conf,
            "dense_top1_cosine": dense_top1,
            "dense_gate_passed": dense_ok,
            "rerank_candidates": n_head,
            "n_candidates": len(sc), "k": k,
        }
        return [self.chunks[i] for i in order[:k]], self.last_trace

    def _rerank_allowed(self, pool: List[int], loc: Dict[str, Any]) -> List[int]:
        """
        ⭐⭐ 重排的**硬约束** —— 这个项目里"事实判断"和"程度判断"的分界。

        交叉编码器只会打"这段文字像不像在回答这个问题"的分，
        **它不知道药名是硬约束** —— 实测不限定的话，60 条里 46 条会把别的药的块排上来，
        检索集 Recall@5 从 1.000 掉到 0.850。

        这里把**已经确定的事实**喂进去当约束，只让它排"不确定的部分"：

            定位到 (药, 章节)  → 只在该章节内排   ← 约束最强
            只定位到 药        → 只在该药的块里排  ← 探针集就是这种情况（意图认不出但药名在）
            都没定位到         → 不限定（返回空 = 走全局，虽然那是负收益，
                                 但**总比什么都不排更可解释**）

        实测收益：探针 test 0.717 → 0.783 ｜ dev 0.700 → 0.833 ｜ 检索集 1.000 → 1.000。
        """
        drug, loincs = loc.get("drug"), set(loc.get("loincs") or [])
        if not drug:
            return []
        if loincs:
            return [i for i in pool
                    if self.chunks[i].drug == drug and self.chunks[i].loinc in loincs]
        return [i for i in pool if self.chunks[i].drug == drug]

    def _allowed_indices(self, restrict: Dict[str, Any]) -> set:
        drug = restrict.get("drug")
        loincs = set(restrict.get("loincs") or [])
        return {i for i, c in enumerate(self.chunks)
                if (not drug or c.drug == drug) and (not loincs or c.loinc in loincs)}

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.index.get(chunk_id)

    def __call__(self, query: str, k: int = 5,
                 restrict: Optional[Dict[str, Any]] = None) -> List[Any]:
        chunks, _ = self.search_detail(query, k, restrict=restrict)
        return [to_doc(c, rank=i + 1, score=1.0 / (i + 1)) for i, c in enumerate(chunks)]


__all__ = ["HybridRetriever"]
