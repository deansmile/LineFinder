# Tables 1–3: paper vs recomputed (corrected GT, 427 images, seeded LineFinder rerun)

Each cell is `paper → new`. **Bold** = best method in that row of the new values. GPT on the 109 older images is the GPT-4o run (kept as decided).

## Table 1: line understanding (proportion correct)

Facing uses the sideways merge for all four methods, and Overall is the mean of the seven rows. The paper's LineFinder Overall 0.6283 is a typo; the mean of its seven values is 0.6567.

| Row | Gemini | GPT | Kimi | LineFinder |
|---|---:|---:|---:|---:|
| Line facing direction | 0.5672 → 0.6628 | 0.3225 → 0.3536 | 0.3449 → 0.3607 | 0.8289 → **0.8478** |
| End of line visible | 0.3727 → 0.3794 | 0.3180 → 0.3255 | 0.4051 → 0.4075 | 0.6995 → **0.7002** |
| End location | 0.1690 → 0.1710 | 0.1381 → 0.1382 | 0.3171 → 0.3162 | 0.6549 → **0.6534** |
| Direction to turn (end) | 0.1088 → 0.1077 | 0.2947 → 0.3091 | 0.3473 → 0.3536 | 0.6502 → **0.6511** |
| Start of line visible | 0.5024 → 0.5059 | 0.4554 → 0.4684 | 0.5741 → 0.5761 | 0.6408 → **0.6323** |
| Start location | 0.1967 → 0.1991 | 0.2204 → 0.2225 | 0.4421 → 0.4450 | 0.5282 → **0.5293** |
| Direction to turn (start) | 0.2338 → 0.2365 | 0.4456 → 0.4543 | 0.4954 → 0.4965 | 0.5942 → **0.5878** |
| Overall | 0.3072 → 0.3232 | 0.3135 → 0.3245 | 0.4180 → 0.4222 | 0.6567 → **0.6574** |

## Table 2: person description (proportion correct, Gemini 3 Flash judge)

New values use the paper's scoring: VLMs with the strict judge prompt and the description-presence rule, LineFinder with its own lenient prompt and visibility-first rule. The last three rows score all four methods the same way (strict prompt, presence rule).

| Row | Gemini | GPT | Kimi | LineFinder |
|---|---:|---:|---:|---:|
| Start person | 0.1527 → 0.1475 | 0.1666 → 0.1733 | 0.2000 → 0.2834 | 0.4601 → **0.4496** |
| End person | 0.1155 → 0.1311 | 0.1343 → 0.1405 | 0.1913 → 0.2319 | 0.6197 → **0.6159** |
| Overall | 0.1341 → 0.1393 | 0.1505 → 0.1569 | 0.1957 → 0.2576 | 0.5399 → **0.5328** |
| Start person (same scoring for all) | 0.1475 | 0.1733 | 0.2834 | **0.3091** |
| End person (same scoring for all) | 0.1311 | 0.1405 | 0.2319 | **0.4426** |
| Overall (same scoring for all) | 0.1393 | 0.1569 | 0.2576 | **0.3759** |

## Table 3: counting

Best: lowest MAE, RMSE and mean relative error; mean signed error closest to 0; highest exact and within ±1/±2; lowest under- and overcount rates. Gemini has 426 images (no count for outdoor_050).

| Metric | Gemini | GPT | Kimi | LineFinder |
|---|---:|---:|---:|---:|
| MAE | 5.61 → 5.62 | 3.89 → 3.89 | 4.49 → 4.50 | 4.26 → **3.84** |
| RMSE | 13.72 → 13.55 | 7.14 → **7.13** | 11.16 → 11.10 | 10.52 → 7.43 |
| Mean signed error | 3.81 → 3.87 | 0.56 → **0.52** | 2.64 → 2.65 | 0.00 → -2.86 |
| Mean relative error | 0.426 → 0.426 | 0.320 → 0.320 | 0.345 → 0.345 | 0.285 → **0.252** |
| Exact match | 0.269 → **0.275** | 0.181 → 0.180 | 0.245 → 0.246 | 0.236 → 0.225 |
| Within ±1 | 0.464 → 0.472 | 0.426 → 0.424 | 0.505 → **0.504** | 0.516 → 0.468 |
| Within ±2 | 0.599 → 0.603 | 0.563 → 0.560 | 0.627 → **0.628** | 0.635 → 0.609 |
| Undercount rate | 0.292 → 0.284 | 0.354 → 0.356 | 0.273 → **0.272** | 0.420 → 0.553 |
| Overcount rate | 0.438 → 0.441 | 0.465 → 0.464 | 0.481 → 0.482 | 0.370 → **0.222** |
