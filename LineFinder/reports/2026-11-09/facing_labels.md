# Step 3: facing labels

## Label set in each GT group (427 evaluated images)

| GT group | Images | Labels (count) |
|---|---:|---|
| Older format (airports, supermarkets, banks, online) | 109 | Facing away 49, Facing towards 29, Facing sideways 29, other 2 |
| Luna | 179 | away 62, sideways-left 49, sideways-right 49, towards 19 |
| Olivia | 139 | away 62, towards 30, sideways-right 24, sideways-left 23 |

**Luna and Olivia distinguish sideways-left from sideways-right** (145 of their 318 images). The older-format GT has only "sideways". `merge_two_format_gt.py` collapses everything to "sideways" in `GT_json_canonical.csv`.

## How each method was scored in the paper

- **LineFinder:** sideways merged in all images (canonical GT).
- **VLMs, Luna/Olivia** (`eval_<model>_luna_olivia.py`): exact label match, **no merge**, so a VLM answering sideways-right on a sideways-left line is wrong.
- **Gemini and Kimi, older images** (`eval_gemini_old.py`, `eval_kimi_old.py`): merged.
- **GPT, older images** (`eval_gpt.py`, GPT-4o run): exact match. GPT-4o's labels use the older-format vocabulary, so no merge is needed.

So the VLMs were **not** scored with the same sideways merge as LineFinder on Luna/Olivia.

## Facing accuracy on 427 images, with and without the merge

| Method | Paper's rule | Merged (all methods alike) | Unmerged (left/right must match) |
|---|---:|---:|---:|
| Gemini | 57.38% | 66.28% | 51.05% |
| GPT | 32.55% | 35.36% | 32.55% |
| Kimi | 35.13% | 36.07% | 32.55% |
| LineFinder | 84.78% | 84.78% | 77.52% |

Unmerged, the older-format GT can only match a plain "sideways" (or "Facing sideways") prediction, so models that answer left/right score lower there.
