| Text given to model | CNN F1 | CNN change | LSTM F1 | LSTM change |
|---|---|---|---|---|
| none (full text) | 0.705 | +0.000 | 0.699 | +0.000 |
| strip_formatting | 0.676 | -0.029 | 0.666 | -0.033 |
| first_50_words | 0.593 | -0.112 | 0.626 | -0.073 |
| lowercase_words_only | 0.645 | -0.060 | 0.612 | -0.086 |
| drop_first_10_words | 0.623 | -0.083 | 0.597 | -0.102 |
