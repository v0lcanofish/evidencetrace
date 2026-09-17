# -*- coding: utf-8 -*-
"""
块 E6 · 分块 + **引用契约**。

    from retrieval.chunker import build_chunks, to_doc

━━━ 为什么检索的粒度是 chunk 而不是整节 ━━━

一份说明书的一节可能有两千多字符。整节塞进上下文：
  ① 浪费 token（用户只问"能不能一起吃"，不需要整节禁忌）
  ② 排序更粗（一节里只有一段相关，整节都得进 top-k）

所以切成段落级 chunk，检索到哪一段就引哪一段。

━━━ ⭐ 引用契约（这是本项目的地基）━━━

调研时看到一句话，直接决定了这里的设计：

    **"引用不该是模型临时生成的字符串，而该是摄取阶段就建立的契约。"**
    —— 开源项目 Source-grounded RAG

所以每个 chunk 在**建立索引时**就带齐这些字段：

    chunk_id      {setid}#{loinc}#{序号}       唯一标识
    doc_id        setid                        哪份说明书
    loinc         章节码                        哪一节
    char_span     (start, end)                 在原文的精确位置
    heading_path  章节路径                      给人读
    content_hash  sha1 前 12 位                 ⭐ 防篡改

**这样"引用核验"就是机械判断**：查这个 chunk_id 在不在索引里、
content_hash 对不对得上 —— **不需要 LLM 当裁判**。

⚠️ 诚实边界：content_hash 只能证明"这句话来自这里"，
   **不能证明"这个推论是对的"**。后者仍需人看。这条写进报告。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# 句子切分：英文句末标点 + 后面的空白
_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+")

# 目标 chunk 长度（字符）。太小会切碎语义，太大会浪费上下文。
TARGET_CHARS = 420
OVERLAP_SENTS = 1          # 相邻 chunk 重叠的句子数（防切断语义）


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class Chunk:
    """一个可检索的片段 —— **带完整引用契约**。"""

    chunk_id: str
    doc_id: str
    loinc: str
    section: str
    drug: str
    text: str
    char_span: Tuple[int, int]
    heading_path: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "loinc": self.loinc,
            "section": self.section, "drug": self.drug, "text": self.text,
            "char_span": list(self.char_span), "heading_path": self.heading_path,
            "content_hash": self.content_hash,
        }


def split_sentences(text: str) -> List[Tuple[str, int]]:
    """切句，并记住每句在原文里的起始位置（后面要算 char_span）。"""
    out: List[Tuple[str, int]] = []
    pos = 0
    for piece in _SENT_SPLIT.split(text):
        if not piece.strip():
            pos += len(piece) + 1
            continue
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        out.append((piece, idx))
        pos = idx + len(piece)
    return out


def build_chunks(labels: List[Dict[str, Any]],
                 target_chars: int = TARGET_CHARS,
                 overlap_sents: int = OVERLAP_SENTS) -> List[Chunk]:
    """
    把说明书章节切成 chunk。

    做法：**按句子贪心合并到目标长度**（而不是按固定字符数硬切）——
    硬切会把句子拦腰截断，检索出来的片段读不通，引用也无意义。

    相邻 chunk 之间**重叠 1 句**：一句话里的关键信息可能跨句
    （"Metformin is contraindicated in patients with: / Renal disease ..."），
    不重叠的话两个 chunk 各丢一半。
    """
    chunks: List[Chunk] = []
    for L in labels:
        drug, setid = L["drug"], L["setid"]
        # ⚠️ 计数键是 **(setid, loinc)**，不是"每节从 1 重新数"。
        #
        #    同一份说明书里可能有**多节共用同一个 LOINC 码** —— 扩语料实测：
        #    4 份药有这情况（omeprazole 的 43685-7 ×2、methotrexate 的 34073-7 ×3……）。
        #    每节重置计数器 → chunk_id 撞车 → `{chunk_id: chunk}` 索引里**后者静默覆盖前者**
        #    （实测 1111 个 chunk 建完索引只剩 1101，**丢了 10 个**，不报任何错）。
        #    后果：被覆盖的那些块**永远检索不到**，而且看不出来。
        #
        #    ⭐ 为什么用这个改法：对**没有重复节的药**，两种计数结果完全一样
        #       （每份 (setid,loinc) 只有一个节 → 计数器行为等价）→
        #       **现有 chunk_id 一个都不变**，评测集不受影响。
        n_by_loinc: Counter = Counter()
        for sec in L.get("sections", []):
            loinc = sec["loinc"]
            section = sec["section"]
            text = (sec.get("text") or "").strip()
            if not text:
                continue

            sents = split_sentences(text)
            i = 0
            while i < len(sents):
                buf, start = [], sents[i][1]
                cur_len = 0
                j = i
                while j < len(sents) and (cur_len < target_chars or j == i):
                    buf.append(sents[j][0])
                    cur_len += len(sents[j][0]) + 1
                    j += 1
                end = sents[j - 1][1] + len(sents[j - 1][0])
                body = " ".join(buf).strip()
                if body:
                    n_by_loinc[loinc] += 1
                    chunks.append(Chunk(
                        chunk_id=f"{setid}#{loinc}#{n_by_loinc[loinc]}",
                        doc_id=setid, loinc=loinc, section=section, drug=drug,
                        text=body, char_span=(start, end),
                        heading_path=f"{drug} / {section}",
                        content_hash=_sha(body),
                    ))
                # 下一块从"上一块最后 overlap_sents 句"开始（制造重叠）
                i = max(i + 1, j - overlap_sents)
    return chunks


def to_doc(chunk: Chunk, rank: int = 0, score: float = 0.0):
    """
    Chunk → 账本里的 `Doc`。

    ⚠️ `Doc` 是**检索结果**的载体（有 rank/score），`Chunk` 是**索引里的**静态数据。
       分开是有意的：同一个 chunk 在不同查询下的 rank/score 不同，
       但它的 chunk_id / content_hash 永远不变 —— 这正是"契约"的含义。

    ⭐ 把 chunk_id 和 content_hash 编码进 `title`，
       这样账本里存下来的 Doc 依然带着完整的可核验信息。
    """
    from ledger.ledger import Doc

    return Doc(
        doc_id=chunk.doc_id,
        section=chunk.section,
        loinc=chunk.loinc,
        drug=chunk.drug,
        text=chunk.text,
        title=f"{chunk.chunk_id}|{chunk.content_hash}|{chunk.heading_path}",
        rank=rank,
        score=score,
        authority=1.0,      # DailyMed 官方说明书，权威性拉满
        recency=1.0,
        verifiable=True,
    )


def verify_chunk(chunk: Chunk, chunks_index: Dict[str, Chunk]) -> Tuple[bool, str]:
    """
    ⭐ **引用核验** —— 机械判断，不问 LLM。

    两道关：
      ① chunk_id 在索引里（这条引用指向真实存在过的片段）
      ② content_hash 对得上（内容**没被改过**）

    返回 (是否通过, 原因)。
    """
    got = chunks_index.get(chunk.chunk_id)
    if got is None:
        return False, f"chunk_id 不在索引里：{chunk.chunk_id}"
    if got.content_hash != chunk.content_hash:
        return False, (f"content_hash 不匹配：索引里是 {got.content_hash}，"
                       f"引用里写的是 {chunk.content_hash} —— 内容被改过")
    return True, "ok"


__all__ = ["Chunk", "build_chunks", "to_doc", "verify_chunk", "split_sentences"]
