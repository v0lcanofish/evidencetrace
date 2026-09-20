# -*- coding: utf-8 -*-
"""
生成层的**唯一开关** —— mock（给断言）还是真模型（给数字）。

━━━ 为什么要有这个（2026-09-19）━━━

`agent/llm.py` 里 `make_llm()`（真 DeepSeek，还带 LLM 缓存）早就写好了、
有文档、有参数，但**分析脚本里一行都没用过** —— 8 个脚本全部硬编码
`GroundedMockLLM()`。于是 E8–E13 的所有数字都是**生成器的性质**，
README §五「生成层换真模型」也就一直挂着"未做"。

这和 observe 那边的退化谱是同一个形状：**实现了、没接进流程**。
这里把它收成一个开关，而不是散在 8 个文件里各改各的。

━━━ 两个生成器不是二选一，是两个用途 ━━━

    GroundedMockLLM   确定性、离线、可复现   → **给断言用**（回归必须逐字节可复现）
    make_llm()        真 DeepSeek（带缓存）   → **给数字用**（报出去的数必须是真跑出来的）

所以**默认仍然是 mock** —— 断言不该因为换了模型而漂。
要真数字时显式打开，并且**启动会打印用的是哪个**：

    ET_REAL_LLM=1 python scripts/eval_select.py --quick    # 先小步验通路
    ET_REAL_LLM=1 python scripts/eval_select.py            # 再跑全量

⚠️ 打印这件事不是装饰：这个项目反复栽在"参数没接上 → 静默走默认值 →
跑完才发现"（`--length-norm` 没透传、`--max-seq-tokens` 写死 8192…）。
**生效的配置必须在启动时自己喊出来。**
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

# 真模型那一档的默认超参。temperature=0 是为了可复现（和缓存配合）。
REAL_KWARGS: dict[str, Any] = {"model": "deepseek-chat", "temperature": 0.0}

_ON, _OFF = {"1", "true", "yes", "y", "on"}, {"0", "false", "no", "n", "off", ""}

_printed = False


def real_enabled() -> bool:
    """`ET_REAL_LLM` 是否要求用真模型。取值不认识就**直接炸**，不猜。"""
    raw = os.environ.get("ET_REAL_LLM", "").strip().lower()
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ValueError(f"ET_REAL_LLM={raw!r} 看不懂 —— 要么留空，要么 1/0。")


def generator_name(real: bool) -> str:
    return "deepseek-chat(真模型)" if real else "GroundedMockLLM"


def make_generator(real: bool | None = None, **kw) -> Callable[[str], str]:
    """
    造生成层。`real=None` 时看环境变量 `ET_REAL_LLM`。

    Args:
        real: 显式指定；None = 读 ET_REAL_LLM（默认 False = mock）
        kw:   传给 make_llm 的超参（offline / cache_path / verbose …）；
              mock 档会忽略它们
    """
    global _printed
    real = real_enabled() if real is None else real

    if real:
        from agent.llm import make_llm
        opts = {**REAL_KWARGS, **kw}
        llm = make_llm(**opts)
    else:
        from agent.mocks import GroundedMockLLM
        llm = GroundedMockLLM()

    if not _printed:
        _printed = True
        tag = generator_name(real)
        print(f"[生成层] {tag}"
              + (f"  {REAL_KWARGS if real else ''}" if real else "  （确定性·离线·给断言用）"),
              file=sys.stderr)
        if not real:
            print("[生成层] ⚠️ 现在是 **mock** —— 这批数字是生成器的性质，不是系统的性质。"
                  "\n           要报出去的真数字请加 ET_REAL_LLM=1。", file=sys.stderr)
    return llm


__all__ = ["make_generator", "generator_name", "real_enabled", "REAL_KWARGS"]
