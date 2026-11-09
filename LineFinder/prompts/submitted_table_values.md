Paper values for the paper-vs-new comparison table (from the submitted tables file). Recompute every cell on the corrected 427-image ground truth with the seeded rerun, and put the paper's value next to each one.

Table 1 (line understanding, proportion correct). Columns: Gemini / GPT / Kimi / LineFinder
- Line facing direction: 0.5672 / 0.3225 / 0.3449 / 0.8289
- End of line visible: 0.3727 / 0.3180 / 0.4051 / 0.6995
- End location: 0.1690 / 0.1381 / 0.3171 / 0.6549
- Direction to turn (end): 0.1088 / 0.2947 / 0.3473 / 0.6502
- Start of line visible: 0.5024 / 0.4554 / 0.5741 / 0.6408
- Start location: 0.1967 / 0.2204 / 0.4421 / 0.5282
- Direction to turn (start): 0.2338 / 0.4456 / 0.4954 / 0.5942
- Overall: 0.3072 / 0.3135 / 0.4180 / 0.6283
  (LineFinder's 0.6283 is a typo in the table. The mean of its seven values is 0.6567, which matches the text's 65.67%. Report the recomputed mean.)

Table 2 (person description, proportion correct). Columns: Gemini / GPT / Kimi / LineFinder
- Start person: 0.1527 / 0.1666 / 0.2000 / 0.4601
- End person: 0.1155 / 0.1343 / 0.1913 / 0.6197
- Overall: 0.1341 / 0.1505 / 0.1957 / 0.5399

Table 3 (counting). Columns: Gemini / GPT / Kimi / LineFinder
- MAE: 5.61 / 3.89 / 4.49 / 4.26
- RMSE: 13.72 / 7.14 / 11.16 / 10.52
- Mean signed error: 3.81 / 0.56 / 2.64 / 0.00
- Mean relative error: 0.426 / 0.320 / 0.345 / 0.285
- Exact match: 0.269 / 0.181 / 0.245 / 0.236
- Within ±1: 0.464 / 0.426 / 0.505 / 0.516
- Within ±2: 0.599 / 0.563 / 0.627 / 0.635
- Undercount rate: 0.292 / 0.354 / 0.273 / 0.420
- Overcount rate: 0.438 / 0.465 / 0.481 / 0.370

Also mark which method is best in each row of the new values, since the tables bold the best method per row.

Table 4: check whether the seeded rerun changes LineFinder's outputs for the three qualitative examples. The paper says: (1) "9 people, facing away, start and end not visible, turn left for start, turn right for end"; (2) "8 people, facing away, start and end not visible, turn left for start, turn right for end"; (3) "6 people, facing right, start visible center right, end visible center left". Report the image IDs and any differences.