"""Project settings: paths, family mapping, preprocessing and model hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
PROCESSED_CSV = DATA_DIR / "processed.csv"

SEED = 42

# LMSYS Chatbot Arena model name -> LLM family (our class label).
# Models not listed here are dropped.
# Vicuna, Koala, Alpaca, WizardLM, Guanaco and GPT4All are all fine-tuned from Meta's LLaMA,
# so they are grouped into the Llama family.
FAMILY_MAP = {
    "gpt-4": "GPT",
    "gpt-3.5-turbo": "GPT",
    "claude-v1": "Claude",
    "claude-instant-v1": "Claude",
    "palm-2": "PaLM",
    "llama-13b": "Llama",
    "vicuna-13b": "Llama",
    "vicuna-7b": "Llama",
    "koala-13b": "Llama",
    "alpaca-13b": "Llama",
    "wizardlm-13b": "Llama",
    "guanaco-33b": "Llama",
    "gpt4all-13b-snoozy": "Llama",
}
FAMILIES = ["GPT", "Claude", "Llama", "PaLM"]

# Data filtering
LANGUAGE = "English"
MIN_OUTPUT_CHARS = 20          # drop empty / one-word answers
MAX_PER_CLASS = None           # None = downsample every family to the smallest family's size

# Split sizes (grouped by prompt so a prompt never appears in two splits)
VAL_FRAC = 0.1
TEST_FRAC = 0.2

# Tokenizer / vocabulary
VOCAB_SIZE = 30000
MIN_FREQ = 2
MAX_LEN = {"input": 128, "output": 384, "input+output": 512}

# Model hyperparameters
EMB_DIM = 128
CNN_FILTERS = 128
CNN_KERNELS = (3, 4, 5)
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT = 0.5

# Training
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 15
PATIENCE = 3                   # early stopping: stop after 3 epochs without val macro-F1 improving
