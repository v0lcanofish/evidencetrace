# -*- coding: utf-8 -*-
"""块 E6 · 检索层：分块 + 引用契约 + BM25 + 章节定位。"""
from retrieval.chunker import Chunk, build_chunks, to_doc, verify_chunk
from retrieval.retriever import BM25Retriever, ChunkBM25, SectionLocator

__all__ = ["Chunk", "build_chunks", "to_doc", "verify_chunk",
           "BM25Retriever", "ChunkBM25", "SectionLocator"]
