# -*- coding: utf-8 -*-
"""
LLM 客户端 —— 和 `agent/mocks.py` 里的假模型**同一个签名**，可直接互换。

    from agent.llm import make_llm
    llm = make_llm()                 # 真模型（DeepSeek）
    llm = make_llm(offline=True)     # 不联网，缺 key 直接报错而不是偷偷降级

━━━ 三个刻意的设计 ━━━

    ① **temperature = 0**
       本项目要量的是"策略带来的差异"，不是"采样带来的差异"。
       温度一开，三档策略的对照里就混进了随机性，数字解释不了。

    ② **磁盘缓存（prompt → 回答）**
       同一段 prompt 第二次问不再花钱、不再联网。
       两个直接好处：
         · 复现性 —— 重跑一遍脚本，数字必须一模一样
         · 省钱     —— 调循环逻辑时反复跑，不重复付同一笔账
       ⚠️ 缓存键是 prompt 原文。改了 prompt = 换了一把钥匙，自然重新问 —— 这是对的。

    ③ **缺 key 就报错，不静默降级**
       静默降级成 mock 是最坏的情况：你以为在跑真模型，其实在跑假的，
       然后拿着一堆假数字下结论。宁可当场炸。

━━━ 为什么不用 mocks.py 那个 MockLLM 跑真实验 ━━━

    MockLLM 是给**断言**用的（确定性、离线、可复现），
    它的回答是按关键词硬编码的 —— 拿它跑真实语料，
    它只会对"ibuprofen + warfarin"那道题有反应，别的题一律回"证据不足"。
    这不是模型不行，是**假模型的适用范围就那么大**。
    所以：断言用 mock，真数字用真模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT / "data" / "cache" / "llm_cache.json"
ENV_FILE = PROJECT.parents[1] / ".env"          # 代码库/.env

DEEPSEEK_BASE = "https://api.deepseek.com"


# ---------------------------------------------------------------- 缓存


class LLMCache:
    """prompt → 回答 的磁盘缓存。**键是 prompt 的 sha1。**"""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.path = Path(path) if path else DEFAULT_CACHE
        self.enabled = enabled
        self.data: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}          # 缓存坏了不值得炸，重建就是

    @staticmethod
    def key(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        if not self.enabled:
            return None
        v = self.data.get(self.key(prompt))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, prompt: str, answer: str) -> None:
        if not self.enabled:
            return
        self.data[self.key(prompt)] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self.data)}


# ---------------------------------------------------------------- key


def load_api_key(env_file: Path = ENV_FILE) -> str:
    """从环境变量或 代码库/.env 里取 key。**取不到就抛，不返回空串。**"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                key = line.split("=", 1)[-1].strip().strip('"').strip("'")
                break
    if not key or key.startswith("sk-请") or len(key) < 12:
        raise RuntimeError(
            f"没找到可用的 DEEPSEEK_API_KEY（找过环境变量和 {env_file}）。\n"
            "要么把 key 填进 .env，要么显式用 mock：make_llm(offline=True) 会直接报这个错。")
    return key


# ---------------------------------------------------------------- 客户端


def make_llm(provider: str = "deepseek",
             model: str = "deepseek-chat",
             temperature: float = 0.0,
             cache_path: Optional[Path] = None,
             use_cache: bool = True,
             offline: bool = False,
             max_retries: int = 3,
             verbose: bool = False) -> Callable[[str], str]:
    """
    造一个 `LLM = Callable[[str], str]`。

    Args:
        offline: True → 只读缓存，**不联网**；缓存里没有就抛错。
                 批量跑批时用它保证"一分钱不多花"，也保证复现。
    """
    cache = LLMCache(cache_path, enabled=use_cache)
    client = None

    if not offline:
        from openai import OpenAI                     # 延迟 import：离线时不依赖它
        key = load_api_key() if provider == "deepseek" else os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(f"provider={provider} 没有可用的 key")
        base = DEEPSEEK_BASE if provider == "deepseek" else None
        client = OpenAI(api_key=key, base_url=base)

    def llm(prompt: str) -> str:
        hit = cache.get(prompt)
        if hit is not None:
            if verbose:
                print(f"   [cache hit] {prompt[:40]}...")
            return hit
        if offline:
            raise RuntimeError(
                f"offline 模式下缓存里没有这条 prompt（{prompt[:60]}...）。\n"
                "要么先联网跑一次把它填进缓存，要么别开 offline。")

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}])
                text = (resp.choices[0].message.content or "").strip()
                cache.put(prompt, text)
                return text
            except Exception as e:                     # noqa: BLE001  网络错就重试
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用连续失败 {max_retries} 次：{last_err}")

    llm.cache = cache                                  # 便于脚本读命中率
    llm.model = model
    llm.offline = offline
    return llm


__all__ = ["make_llm", "LLMCache", "load_api_key", "DEFAULT_CACHE"]
