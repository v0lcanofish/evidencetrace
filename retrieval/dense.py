# -*- coding: utf-8 -*-
"""
块 E6 · 稠密检索层 —— 语义匹配，用来补**结构先验失效**时的空缺。

    from retrieval.dense import DenseRetriever

━━━ 为什么需要它（2026-09-18 实测的定位）━━━

    检索集 60 条：BM25 0.667 → +章节定位 0.867。**但定位 60/60 全对，指标饱和。**
    探针 120 条（先验失效）：BM25 **0.458**，定位贡献 **+0.000**。

    ⇒ 稠密检索的活干在哪，是量出来的，不是猜的：
      **结构先验够不着的地方**。词面没重合、意图词不认识、章节定位不了 —— 剩下全靠语义。

━━━ 两个实现决策（都不是随手选的）━━━

① **不装 sentence-transformers**

   本机只有 `transformers` + `torch(CPU)`。sentence-transformers 是**一层薄封装**，
   核心就两件事：mean pooling + L2 归一化。自己写十几行，**少一个依赖、少一层版本地狱**。
   ⚠️ 反过来，自己写就必须保证 pooling 方式和模型训练时一致 —— 不一致会**静默变差**
      （向量还算得出来，只是不好用）。bge 系列用 mean pooling，已按此实现。

② **向量落盘缓存，但缓存必须"认得自己是什么"**

   1156 个 chunk 编码一次要几分钟 CPU。缓存键 = (模型名 + 文本的 sha1)。
   文本一变（比如 E5 那样扩语料）→ 指纹变 → **自动重算**。
   ⚠️ 这是"静默失真"的高发区：缓存不校验指纹的话，改了语料却拿旧向量跑，
      数字会**悄悄地不对**，而且看不出来。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ⚠️ 环境事实（2026-09-18 实测）：huggingface.co **直连超时**，hf-mirror.com 通（1.4s）。
#    不设这个的话，第一次跑会在 from_pretrained 上卡到超时，看起来像"代码坏了"。
#    用 setdefault —— **用户显式设过就听用户的**，不覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# bge 系列的**查询指令前缀**。官方说明：短查询做检索时加上它，长段落不加。
# ⚠️ 只加在 query 侧、不加在 passage 侧 —— 两边都加会削弱效果。
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _mean_pool(last_hidden: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
    """按 attention_mask 做 mean pooling。"""
    m = mask[..., None].astype(np.float32)
    return (last_hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


class DenseRetriever:
    """
    稠密检索：把 chunk 和 query 都编码成向量，用余弦相似度排序。

    Args:
        texts:       语料（与 BM25 用**同一批 chunk 文本、同一个顺序** —— 下标必须能对齐）
        model_name:  HF 模型名，默认 bge-small-en-v1.5（33.4M 参数 / 384 维，CPU 够用）
        cache_path:  向量缓存目录
        batch_size:  CPU 上 32 比较稳
        max_length:  截断长度。DailyMed 的 chunk 中位 ~429 字符 ≈ 100 token，512 足够
    """

    def __init__(self, texts: Sequence[str],
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_path: Optional[Path] = None,
                 batch_size: int = 32,
                 max_length: int = 512,
                 verbose: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self.texts = list(texts)
        self._tok = None
        self._model = None

        self.emb = self._load_or_build(texts, cache_path)

    # ------------------------------------------------------------ 模型
    def _lazy_model(self):
        """延迟加载 —— 只读缓存时根本不用把模型读进内存。"""
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            # ⚠️ 本机 CPU 版 torch。线程数交给 torch 自己定，
            #    手动写死会在别的机器上变成"跑得莫名其妙地慢"。
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
            self._torch = torch
        return self._tok, self._model

    def _encode(self, texts: Sequence[str]) -> "np.ndarray":
        tok, model = self._lazy_model()
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = tok(batch, padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt")
                hidden = model(**enc).last_hidden_state
                vec = _mean_pool(hidden.numpy(), enc["attention_mask"].numpy())
                out.append(vec)
                if self.verbose and (i // self.batch_size) % 10 == 0:
                    print(f"      编码 {i + len(batch)}/{len(texts)} ...", flush=True)
        v = np.concatenate(out, axis=0).astype(np.float32)
        # L2 归一化 → 内积即余弦
        v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        return v

    # ------------------------------------------------------------ 缓存
    @staticmethod
    def _fingerprint(texts: Sequence[str]) -> str:
        h = hashlib.sha1()
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")          # 分隔符：不加的话 ["ab","c"] 和 ["a","bc"] 指纹相同
        return h.hexdigest()[:16]

    def _load_or_build(self, texts: Sequence[str],
                       cache_path: Optional[Path]) -> "np.ndarray":
        import re
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.model_name)
        fp = self._fingerprint(texts)
        if cache_path is None:
            return self._encode(texts)

        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        vec_file = cache_path / f"{safe}.npy"
        meta_file = cache_path / f"{safe}.meta.json"
        meta = {"model": self.model_name, "fingerprint": fp, "n": len(texts)}

        if vec_file.exists() and meta_file.exists():
            old = json.loads(meta_file.read_text(encoding="utf-8"))
            if old == meta:
                arr = np.load(vec_file)
                if arr.shape[0] == len(texts):
                    if self.verbose:
                        print(f"      [cache] 复用 {vec_file.name} "
                              f"（{arr.shape[0]}×{arr.shape[1]}，指纹 {fp}）")
                    return arr
                # ⚠️ 条数对不上 = 缓存坏了。**不悄悄重建、也不悄悄用**，先说清楚。
                print(f"      [cache] ⚠️ 缓存条数 {arr.shape[0]} ≠ 语料 {len(texts)}，重建")
            else:
                # ⭐ 这是最危险的一种：文件在、但内容对不上（语料改了 / 换了模型）
                if self.verbose:
                    print(f"      [cache] 指纹不符（缓存 {old.get('fingerprint')} "
                          f"≠ 当前 {fp}）→ 重建（语料或模型变过了）")

        arr = self._encode(texts)
        np.save(vec_file, arr)
        meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.verbose:
            print(f"      [cache] 已存 {vec_file.name}（{arr.shape[0]}×{arr.shape[1]}）")
        return arr

    # ------------------------------------------------------------ 检索
    def encode_queries(self, queries: Sequence[str]) -> "np.ndarray":
        return self._encode([QUERY_PREFIX + q for q in queries])

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """返回 [(chunk 下标, 余弦), ...]，降序。"""
        qv = self.encode_queries([query])
        sims = (self.emb @ qv[0])
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def score_all(self, query: str) -> Dict[int, float]:
        qv = self.encode_queries([query])
        sims = self.emb @ qv[0]
        return {int(i): float(s) for i, s in enumerate(sims)}


__all__ = ["DenseRetriever", "QUERY_PREFIX"]
