# -*- coding: utf-8 -*-
"""块 E6 · 检索层：分块 + 引用契约 + BM25 + 章节定位 + 稠密 + 三路融合。"""
from retrieval.chunker import Chunk, build_chunks, to_doc, verify_chunk
from retrieval.retriever import BM25Retriever, ChunkBM25, SectionLocator
from retrieval.hybrid import HybridRetriever
from retrieval.factory import make_retriever, EMB_CACHE

__all__ = ["Chunk", "build_chunks", "to_doc", "verify_chunk",
           "BM25Retriever", "ChunkBM25", "SectionLocator", "HybridRetriever", "make_retriever", "EMB_CACHE"]
