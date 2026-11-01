# Product Price Deep Neural Network

A standalone residual deep neural network that predicts product prices from text summaries. Extracted from the [LLM Engineering](https://github.com/ed-donner/llm_engineering) Week 6 pricer work, packaged to show a full train → evaluate flow we ran ourselves.

The model hashes product text with scikit-learn's `HashingVectorizer`, then regresses log-normalized price with a deep PyTorch network of residual blocks (~289M parameters at default settings).

## What's included

```
product-price-dnn/
├── pricer/
│   ├── deep_neural_network.py   # Residual DNN + DeepNeuralNetworkRunner
│   ├── items.py                 # Product Item model + Hugging Face loaders
│   └── evaluator.py             # MAE / MSE / R² charts for predictors
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- A Hugging Face account and token (to download the product dataset)
- Optional but recommended: NVIDIA GPU (CUDA) or Apple Silicon (MPS)

## Setup

```bash
git clone <your-repo-url> product-price-dnn
cd product-price-dnn

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# or: pip install -e .
```

Copy the env template and add your Hugging Face token:

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env` to a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Data

| Mode | Dataset | Approx. train size |
|------|---------|--------------------|
| Full | `ed-donner/items_full` | 800,000 |
| Lite | `ed-donner/items_lite` | smaller / faster |

Each `Item` includes `title`, `category`, `price`, and a `summary` used as the model input.

## Train

```python
from dotenv import load_dotenv
import os
from huggingface_hub import login
from pricer import Item, DeepNeuralNetworkRunner

load_dotenv(override=True)
login(os.environ["HF_TOKEN"], add_to_git_credential=True)

train, val, test = Item.from_hub("ed-donner/items_full")  # or items_lite
runner = DeepNeuralNetworkRunner(train, val[:1000])
runner.setup()
runner.train(epochs=5)
runner.save("deep_neural_network.pth")
```

What the runner does:

1. Vectorize summaries with `HashingVectorizer(n_features=5000)`
2. Log-normalize prices and standardize with train mean/std
3. Train with AdamW, L1 loss, cosine LR schedule, and gradient clipping
4. Report validation MAE in dollars each epoch
5. Save weights to `deep_neural_network.pth`

Full training (5 epochs on the full dataset) can take several hours depending on hardware.

## Evaluate

```python
from pricer import evaluate

runner.load("deep_neural_network.pth", "cpu")  # or "cuda" / "mps"
evaluate(lambda item: runner.inference(item), test)
```

The evaluator prints per-example absolute error (color-coded) and shows Plotly scatter + cumulative-error charts.

## Model notes

| Piece | Detail |
|-------|--------|
| Input | Binary hashed bag-of-words, 5,000 features |
| Architecture | Linear stem → residual blocks (`LayerNorm` + ReLU + Dropout) → linear head |
| Defaults | 10 layers, hidden size 4096, dropout 0.2 |
| Target | Log-price, standardized on the training set |
| Loss | L1 (MAE) on normalized targets |
| Optimizer | AdamW (`lr=0.001`, `weight_decay=0.01`) |

Device selection is automatic: CUDA → MPS → CPU.

## Publishing this as its own GitHub repo

This folder is already self-contained. From inside `product-price-dnn`:

```bash
git init
git add .
git commit -m "Initial commit: product price deep neural network"
gh repo create product-price-dnn --public --source=. --remote=origin --push
```

Do **not** commit `.env` or large `*.pth` weight files (both are gitignored). Host weights externally if you want to share them.

