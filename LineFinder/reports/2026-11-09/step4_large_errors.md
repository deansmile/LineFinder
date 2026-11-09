# LineFinder count errors > 10 (seeded rerun, corrected GT, 427 images)

40 images; together 83.4% of LineFinder's total squared error (sum = 23547).

Shortfall split: detection miss = GT − detected boxes; RANSAC rejection = boxes − inliers. Label = the larger part.

| Image | Scene | GT | Boxes | Inliers (count) | Error | Detection miss | RANSAC rejection | Mainly | Share of SE |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| IMG_6787 | bank | 70 | 15 | 14 | -56 | +55 | +1 | detection miss | 13.3% |
| IMG_6782 | bank | 57 | 16 | 11 | -46 | +41 | +5 | detection miss | 9.0% |
| IMG_5978 | online | 50 | 13 | 10 | -40 | +37 | +3 | detection miss | 6.8% |
| themepark_017 | theme park | 45 | 14 | 10 | -35 | +31 | +4 | detection miss | 5.2% |
| IMG_6704 | airport | 43 | 14 | 9 | -34 | +29 | +5 | detection miss | 4.9% |
| IMG_5972 | online | 50 | 21 | 19 | -31 | +29 | +2 | detection miss | 4.1% |
| IMG_6784 | bank | 47 | 26 | 19 | -28 | +21 | +7 | detection miss | 3.3% |
| themepark_042 | theme park | 35 | 15 | 12 | -23 | +20 | +3 | detection miss | 2.2% |
| themepark_054 | theme park | 33 | 14 | 10 | -23 | +19 | +4 | detection miss | 2.2% |
| IMG_5958 | online | 30 | 12 | 9 | -21 | +18 | +3 | detection miss | 1.9% |
| IMG_6777 | bank | 35 | 15 | 14 | -21 | +20 | +1 | detection miss | 1.9% |
| outdoor_086 | outdoor | 33 | 23 | 12 | -21 | +10 | +11 | RANSAC rejection | 1.9% |
| themepark_085 | theme park | 31 | 11 | 10 | -21 | +20 | +1 | detection miss | 1.9% |
| IMG_6767 | bank | 32 | 16 | 12 | -20 | +16 | +4 | detection miss | 1.7% |
| themepark_098 | theme park | 35 | 16 | 15 | -20 | +19 | +1 | detection miss | 1.7% |
| IMG_5980 | online | 30 | 14 | 12 | -18 | +16 | +2 | detection miss | 1.4% |
| IMG_6711 | airport | 39 | 23 | 21 | -18 | +16 | +2 | detection miss | 1.4% |
| IMG_6745 | supermarket (old) | 28 | 10 | 10 | -18 | +18 | +0 | detection miss | 1.4% |
| outdoor_070 | outdoor | 35 | 21 | 18 | -17 | +14 | +3 | detection miss | 1.2% |
| themepark_035 | theme park | 27 | 16 | 10 | -17 | +11 | +6 | detection miss | 1.2% |
| outdoor_122 | outdoor | 30 | 14 | 14 | -16 | +16 | +0 | detection miss | 1.1% |
| outdoor_130 | outdoor | 35 | 19 | 19 | -16 | +16 | +0 | detection miss | 1.1% |
| themepark_093 | theme park | 28 | 18 | 12 | -16 | +10 | +6 | detection miss | 1.1% |
| outdoor_043 | outdoor | 34 | 19 | 19 | -15 | +15 | +0 | detection miss | 1.0% |
| IMG_6692 | airport | 28 | 14 | 14 | -14 | +14 | +0 | detection miss | 0.8% |
| IMG_6773 | bank | 29 | 24 | 15 | -14 | +5 | +9 | RANSAC rejection | 0.8% |
| IMG_6786 | bank | 24 | 15 | 10 | -14 | +9 | +5 | detection miss | 0.8% |
| outdoor_014 | outdoor | 26 | 12 | 12 | -14 | +14 | +0 | detection miss | 0.8% |
| outdoor_185 | outdoor | 32 | 20 | 18 | -14 | +12 | +2 | detection miss | 0.8% |
| themepark_032 | theme park | 27 | 14 | 13 | -14 | +13 | +1 | detection miss | 0.8% |
| themepark_010 | theme park | 23 | 17 | 10 | -13 | +6 | +7 | RANSAC rejection | 0.7% |
| themepark_034 | theme park | 21 | 10 | 8 | -13 | +11 | +2 | detection miss | 0.7% |
| themepark_095 | theme park | 29 | 19 | 17 | -12 | +10 | +2 | detection miss | 0.6% |
| IMG_6685 | airport | 18 | 10 | 7 | -11 | +8 | +3 | detection miss | 0.5% |
| IMG_6766 | bank | 24 | 15 | 13 | -11 | +9 | +2 | detection miss | 0.5% |
| IMG_6778 | bank | 27 | 16 | 16 | -11 | +11 | +0 | detection miss | 0.5% |
| IMG_6785 | bank | 21 | 13 | 10 | -11 | +8 | +3 | detection miss | 0.5% |
| outdoor_006 | outdoor | 31 | 20 | 20 | -11 | +11 | +0 | detection miss | 0.5% |
| themepark_055 | theme park | 23 | 16 | 12 | -11 | +7 | +4 | detection miss | 0.5% |
| themepark_103 | theme park | 21 | 16 | 10 | -11 | +5 | +6 | RANSAC rejection | 0.5% |

Totals: {'detection miss': 36, 'RANSAC rejection': 4}.
By scene: {'bank': 10, 'online': 4, 'theme park': 13, 'airport': 4, 'outdoor': 8, 'supermarket (old)': 1}.
