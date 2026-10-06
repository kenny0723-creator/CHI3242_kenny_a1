from __future__ import annotations

import html
import re
from pathlib import Path

import jieba
import pandas as pd
from opencc import OpenCC
from qhchina import load_stopwords
from qhchina.analytics.collocations import find_collocates


ROOT = Path(__file__).resolve().parent
TEXT_PATH = ROOT / "data" / "novel.txt"
OUTPUT_DIR = ROOT / "output"
# 更換 TARGET 可改分析角色；EXPERIMENTS 定義要比較的共現語境。
TARGET = "伯夷"
EXPERIMENTS = [
    ("window_h3", "window", 3),
    ("window_h5", "window", 5),
    ("window_h10", "window", 10),
    ("window_h15", "window", 15),
    ("sentence", "sentence", None),
]


def prepare_sentences(text: str) -> tuple[list[list[str]], str]:
    """先統一為簡體，再分句、分詞，回傳詞元句列與正規化全文。"""
    # OpenCC 必須在分句與 jieba 分詞前執行，確保字形處理順序一致。
    simplified = OpenCC("t2s").convert(text)
    # 固定人物姓名邊界，避免被 jieba 和相鄰字合併而漏算。
    jieba.add_word(TARGET, freq=10_000_000)
    jieba.add_word("叔齐", freq=10_000_000)
    # 以句末標點或換行切分；保留句子列表，供 sentence/window 方法共用。
    raw_sentences = re.split(r"(?<=[。！？!?；;])|\n+", simplified)
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = [
            token.strip()
            for token in jieba.lcut(sentence, HMM=True)
            if token.strip() and re.search(r"[\u4e00-\u9fff]", token)
        ]
        if tokens:
            sentences.append(tokens)
    return sentences, simplified


def render_html(tables: list[tuple[str, pd.DataFrame]], stats: dict[str, int]) -> None:
    """將各次實驗的表格整合到同一頁，方便比較不同語境設定。"""
    sections = []
    for name, frame in tables:
        title = name.replace("_", " ")
        if frame.empty:
            table_html = '<p class="empty">沒有符合篩選條件的顯著搭配詞。</p>'
        else:
            table_html = frame.to_html(index=False, border=0, classes="results")
        sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<p>{len(frame)} 個搭配詞 · CSV: output/collocates_{html.escape(name)}.csv</p>"
            f"{table_html}</section>"
        )

    page = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>伯夷搭配詞比較</title>
<style>
body {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #202b32; font: 16px/1.65 "Segoe UI", sans-serif; background: #f5f6f2; }}
h1 {{ margin-bottom: 4px; font-size: 30px; }}
.meta {{ color: #59676b; margin: 0 0 28px; }}
section {{ margin: 28px 0; padding-top: 16px; border-top: 2px solid #bf5a3c; }}
h2 {{ text-transform: capitalize; font-size: 21px; margin-bottom: 4px; }}
section > p {{ color: #59676b; margin-top: 0; }}
.results {{ border-collapse: collapse; width: 100%; font-size: 14px; background: white; }}
.results th, .results td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #dce1dc; }}
.results th {{ background: #e8ede6; position: sticky; top: 0; }}
.results td {{ overflow-wrap: anywhere; }}
.empty {{ padding: 18px; background: white; }}
@media (max-width: 700px) {{ body {{ margin-top: 20px; }} .results {{ font-size: 12px; }} .results th, .results td {{ padding: 6px; }} }}
</style>
</head>
<body>
<h1>伯夷：搭配詞結果比較</h1>
<p class="meta">《采薇》｜簡體化後 {stats['characters']:,} 個字元、{stats['tokens']:,} 個詞、{stats['sentences']:,} 個分句；目標詞 {stats['target_tokens']} 次。{len(EXPERIMENTS)} 種設定均使用 Fisher 單尾檢定、BH 校正、min_word_length ≥ 2。</p>
{''.join(sections)}
</body>
</html>
"""
    (OUTPUT_DIR / "results.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = TEXT_PATH.read_text(encoding="utf-8")
    sentences, simplified = prepare_sentences(original)
    token_count = sum(map(len, sentences))
    target_count = sum(sentence.count(TARGET) for sentence in sentences)
    if target_count == 0:
        raise ValueError(f"Target word {TARGET!r} was not found after tokenization")

    stopwords = sorted(load_stopwords("zh_sim"))
    # 同時要求原始 p 值及 BH 校正 p 值低於 .05，並排除停用詞與單字詞。
    filters = {
        "stopwords": stopwords,
        "min_word_length": 2,
        "max_p": 0.05,
        "max_adjusted_p": 0.05,
    }
    stats = {
        "characters": len(simplified),
        "tokens": token_count,
        "sentences": len(sentences),
        "target_tokens": target_count,
    }
    tables = []

    for name, method, horizon in EXPERIMENTS:
        # greater 檢驗「過度共現」；fdr_bh 校正同時測試多個候選詞的影響。
        options = {
            "sentences": sentences,
            "target_words": TARGET,
            "method": method,
            "measures": ["log_dice", "t_score"],
            "filters": filters,
            "correction": "fdr_bh",
            "alternative": "greater",
            "sort_by": "adjusted_p_value",
            "ascending": True,
            "return_type": "dataframe",
            "max_sentence_length": None,
        }
        if horizon is not None:
            options["horizon"] = horizon
        frame = find_collocates(**options)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        if not frame.empty:
            # 額外採嚴格小於 .05，確保輸出不會含等於門檻的結果。
            frame = frame.loc[
                (frame["p_value"] < 0.05)
                & (frame["adjusted_p_value"] < 0.05)
            ].copy()
        csv_path = OUTPUT_DIR / f"collocates_{name}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        tables.append((name, frame))
        print(f"{name}: {len(frame)} significant collocates -> {csv_path.relative_to(ROOT)}")
        if not frame.empty:
            print(frame.head(12).to_string(index=False))

    render_html(tables, stats)
    print(
        f"Corpus: {stats['characters']} characters, {stats['tokens']} tokens, "
        f"{stats['sentences']} sentences, {stats['target_tokens']} target tokens"
    )
    print("Wrote output/results.html")


if __name__ == "__main__":
    main()