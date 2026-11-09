# Table 2: where the judgments are and what they cover

**The judgments are automated, not manual.** Gemini 3 Flash (`gemini-3-flash-preview`) decides whether a predicted description and the GT description describe the same person. Before this rerun they were saved only for the 109 older images (`code/same_person_eval_gemini.jsonl`, `code/same_person_eval_kimi.jsonl`, `code/gemini_same_person_eval_kimi.jsonl`), and nearly all of those entries were empty. LineFinder's judgments were never saved.

**Now:** every judgment used for the new Table 2 is saved in `results/2026-11-09/table2/judgments_gemini3flash.jsonl` (on the HF dataset). Each line holds the prompt version, both descriptions, the answer, the model and the time. `code/evaluation/eval_table2_all.py` reads it, so recomputing costs nothing.

**Coverage:** none of the 427 images lacks a judgment. Every endpoint needing one got one, with 0 failed judge calls, and every method has a prediction for all 427 images. LineFinder's descriptions are new, from the seeded rerun's start/end picks (`results/2026-11-09/official/start_end_person_descriptions.csv`, GPT-5.4 mini, 427 images, 0 errors).

**Two scorings were in use:**

| | VLM scripts (`eval_start_end_person_<model>[_new].py`) | LineFinder script (`eval_start_end_person.py`) |
|---|---|---|
| Judge prompt | strict: "Be conservative: if uncertain, return false", temperature 0 | lenient: "Return true if the two descriptions are likely describing the same person", default temperature |
| Rule | both descriptions missing → correct; one missing → wrong; else judge | both endpoints not visible → correct; visibility disagrees → wrong; else as left |

The paper used each script's own scoring. `tables_paper_vs_new.md` gives both that and one strict scoring for all four methods.

**Check against the paper (pre-correction GT):** pooled by image group, the recomputed values match the recorded logs for Gemini (Luna/Olivia 11.27% = 11.27%) and GPT (13.89% vs 13.73%; the recorded run had 323 images). The older-image values are about 1.7–2.1 points higher (20.64% vs 18.98% for GPT, 22.02% vs 19.91% for Gemini), partly because IMG_6728 is now included (109 images, not 108). Kimi does not reproduce: the paper's 0.1957 implies about 20.7% on Luna/Olivia, while `kimi_line_luna_olivia.jsonl` gives 29.0%. The paper's Kimi Table 2 value was probably computed from a different prediction file; `kimi_line_luna.jsonl` exists but is empty.
