# -*- coding: utf-8 -*-
"""
E4 第 1 步 · 从 DailyMed 抓真说明书，按【章节】切成结构化证据单元。

为什么按章节切：
    说明书不是一坨文本 —— 它有 HL7 SPL 规定的标准章节，每节带 **LOINC 编码**。
    「回答这个问题该引哪一节」因此是**机械可判**的 → gold 证据集能半自动构造。

⚠️ 几个踩过的点：
    · LOINC 在 <code code="34073-7" codeSystem="2.16.840.1.113883.6.1"> 里
      —— 属性顺序固定，正则要按实际顺序写
    · **section 是嵌套的**，不能用正则切；必须用 ElementTree 走树
    · SPL 有命名空间 urn:hl7-org:v3，遍历时要带上
    · 同一个药名会命中一堆厂家 —— 优先取**处方药**（有 DRUG INTERACTIONS 节的）

跑法（纯 CPU，要联网）：
  cd 代码库/projects/EvidenceTrace
  PYTHONIOENCODING=utf-8 python scripts/fetch_labels.py
"""

import copy
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
NS = {"hl7": "urn:hl7-org:v3"}
HEADERS = {"User-Agent": "Mozilla/5.0 (EvidenceTrace research)"}

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "data" / "labels.json"

# 我们关心的章节（LOINC → 短名）。只留能支撑问答的那些。
WANTED = {
    "34067-9": "Indications and Usage",
    "34070-3": "Contraindications",
    "34073-7": "Drug Interactions",
    "34071-1": "Warnings",
    "34068-7": "Dosage and Administration",
    "43685-7": "Warnings and Precautions",
    "34072-5": "Adverse Reactions",
    "34069-5": "Precautions",
}

DRUGS = ["warfarin", "metformin", "atorvastatin", "lisinopril",
         "levothyroxine", "omeprazole"]


def get(url: str, timeout: int = 30) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=timeout).read()


def find_setid(drug: str) -> str:
    """按药名找 setid —— 优先挑真有 DRUG INTERACTIONS 的处方药。

    ⚠️ 药名必须 **URL 编码**：「insulin glargine」「valproic acid」带空格，
       不编码会直接抛 InvalidURL（`urllib` 不容忍裸空格）。
       扩语料时实测踩到 —— 两个药因为一个空格被静默跳过。
    """
    q = urllib.parse.quote(drug)
    d = json.loads(get(f"{BASE}/spls.json?drug_name={q}&pagesize=20").decode())
    for item in d.get("data", []):
        sid = item.get("setid")
        if not sid:
            continue
        try:
            xml = get(f"{BASE}/spls/{sid}.xml").decode("utf-8", "replace")
        except Exception:                                     # noqa: BLE001
            continue
        if "34073-7" in xml:          # 有 DRUG INTERACTIONS 节 → 处方药
            return sid
    # 退而求其次：只要能拿到的第一份
    data = d.get("data", [])
    return data[0]["setid"] if data else ""


def section_text(el) -> str:
    """
    把 <section> 自己的文字抠出来。

    ⚠️ 必须先摘掉嵌套的 <section> —— 否则父节的文字里会混进整个子节，重复且串味。
    """
    clone = copy.deepcopy(el)
    for sub in clone.findall(".//hl7:section", NS):
        parent = _find_parent(clone, sub)
        if parent is not None:
            parent.remove(sub)
    parts = ["".join(t.itertext()) for t in clone.iter("{urn:hl7-org:v3}text")]
    if not parts:
        parts = ["".join(clone.itertext())]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _find_parent(root, target):
    for p in root.iter():
        for c in list(p):
            if c is target:
                return p
    return None


def parse_sections(xml: str):
    """走树抽章节。返回 [{loinc, section, text}]；嵌套节会被展平。"""
    root = ET.fromstring(xml)
    out = []

    # ⚠️ 必须用 iter() 遍历【全树】——
    #    section 不是 <document> 的直接子节点（路径是
    #    document → component → structuredBody → component → section），
    #    findall("hl7:section") 会返回 0 个。
    for sec in root.iter("{urn:hl7-org:v3}section"):
        code = sec.find("hl7:code", NS)
        loinc = code.get("code") if code is not None else None
        disp = code.get("displayName") if code is not None else None
        if loinc not in WANTED:
            continue
        txt = section_text(sec)
        if len(txt) > 60:                            # 太短的丢掉
            out.append({"loinc": loinc, "section": WANTED[loinc],
                        "display": disp, "text": txt[:4000]})
    return out


def main() -> int:
    print("=" * 76)
    print("E4 · 从 DailyMed 抓处方药说明书（按章节切）")
    print("=" * 76)

    labels = []
    for drug in DRUGS:
        print(f"\n[{drug}] 找 setid ...")
        try:
            sid = find_setid(drug)
        except Exception as e:                                # noqa: BLE001
            print(f"   失败：{type(e).__name__}: {e}")
            continue
        if not sid:
            print("   没有可用结果")
            continue
        try:
            xml = get(f"{BASE}/spls/{sid}.xml").decode("utf-8", "replace")
        except Exception as e:                                # noqa: BLE001
            print(f"   取全文失败：{e}")
            continue
        secs = parse_sections(xml)
        print(f"   setid={sid[:18]}...  抽到 {len(secs)} 个章节")
        for s in secs:
            print(f"      {s['loinc']}  {s['section']:<26} {len(s['text']):>5} 字符")
        if secs:
            labels.append({"drug": drug, "setid": sid, "sections": secs})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    n_sec = sum(len(l["sections"]) for l in labels)
    print("\n" + "=" * 76)
    print(f"✅ 抓到 {len(labels)} 份说明书，共 {n_sec} 个章节")
    print(f"   已存 → {OUT}")
    print("=" * 76)
    return 0 if labels else 1


if __name__ == "__main__":
    raise SystemExit(main())
