| Test domain | Model | Setting | Train rows | Test rows | Acc | Macro-F1 | F1 GPT | F1 Claude | F1 Llama | F1 PaLM |
|---|---|---|---|---|---|---|---|---|---|---|
| writing | CNN | cross-domain | 6370 | 238 | 0.618 | 0.603 | 0.500 | 0.748 | 0.462 | 0.703 |
| writing | CNN | in-domain | 6370 | 238 | 0.723 | 0.714 | 0.634 | 0.917 | 0.476 | 0.828 |
| writing | LSTM | cross-domain | 6370 | 238 | 0.660 | 0.654 | 0.561 | 0.821 | 0.455 | 0.779 |
| writing | LSTM | in-domain | 6370 | 238 | 0.723 | 0.723 | 0.639 | 0.885 | 0.556 | 0.812 |
| coding | CNN | cross-domain | 6447 | 261 | 0.644 | 0.635 | 0.605 | 0.738 | 0.490 | 0.709 |
| coding | CNN | in-domain | 6447 | 261 | 0.716 | 0.710 | 0.662 | 0.816 | 0.606 | 0.758 |
| coding | LSTM | cross-domain | 6447 | 261 | 0.701 | 0.693 | 0.656 | 0.825 | 0.552 | 0.736 |
| coding | LSTM | in-domain | 6447 | 261 | 0.709 | 0.709 | 0.649 | 0.794 | 0.619 | 0.772 |
