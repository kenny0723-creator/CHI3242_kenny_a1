# Character Space Through Collocations

This repository studies Boyi (伯夷) in Lu Xun's short story "Caiwei" (〈采薇〉), included in *Old Tales Retold* (《故事新編》). The UTF-8 corpus is the simplified-Chinese text supplied in `data/novel.txt`; OpenCC converts it to simplified Chinese again before sentence segmentation and tokenization so that the processing sequence is explicit and reproducible.

- Work: "Caiwei" (〈采薇〉), *Old Tales Retold* (《故事新編》)
- Author: Lu Xun (魯迅)
- Work and contents reference: [Wenshuoge, *Old Tales Retold*](https://www.wenshuoge.com/book/221/%E6%95%85%E4%BA%8B%E6%96%B0%E7%BC%96) (lists "Caiwei" among the eight stories)
- Analysis corpus: the local UTF-8 text in `data/novel.txt`; it was not fetched directly from Wenshuoge, and this project does not claim a character-for-character match with its online text.
- Target character: Boyi (伯夷)

## Reproduce

Use Python 3.10 or newer. Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
python analyze.py
```

The script converts the full text with OpenCC before splitting sentences or calling jieba. It runs five qhchina experiments: `method="window"` with `horizon=3`, `5`, `10`, and `15`, plus `method="sentence"` without a horizon. All use `qhchina.load_stopwords("zh_sim")`, a minimum token length of two, Fisher's one-sided `alternative="greater"`, and Benjamini-Hochberg correction. CSV output retains only rows whose raw and adjusted p-values are both strictly below .05.

Generated files are `output/collocates_*.csv` and the comparison page `output/results.html`. The report is provided as both `report.md` and `report.pdf`.

