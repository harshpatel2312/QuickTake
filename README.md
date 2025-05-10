# QuickTake

A high-performance article summarizer based on a version of `distilbart-cnn-12-6`, designed for fast and memory-efficient use. This project focuses on optimizing pre-trained Hugging Face models using __dynamic quantization__, making them more accessible for deployment and personal use.

---

## 🚀 Overview
- ✅ Uses sshleifer/distilbart-cnn-12-6 (a distilled version of Facebook's BART model)
- 🧠 Capable of generating abstractive summaries for long-form text
- ⚡ Optimized using PyTorch dynamic quantization to reduce model size and speed up CPU inference
- 💡 Easy to integrate into Python scripts or Jupyter notebooks

---

## Setting Up Environment & Installing Dependencies

### 1. Clone the repository
```bash
git clone https://github.com/harshpatel2312/QuickTake.git
cd QuickTake
```

### 2. Virtual environment setup
```bash
# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧰 Model Details
- Base model: [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- Architecture: Transformer Encoder-Decoder (Seq2Seq)
- Vocabulary size: ~50K
- Max input length: 1024 tokens
- Quantization: `torch.quantization.quantize_dynamic`

---

## ⚙️ Optimization Method: Dynamic Quantization
Dynamic quantization reduces model size and speeds up inference by:
- Converting `torch.nn.Linear` weights to __INT8__
- Keeping non-linear ops (like attention) in FP32
- Ideal for __CPU-based inference__

```python
import torch
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.eval()
```

---

## 🧪 How to Use
### 1. Option 1: Using the `QuickTake.pt` from GitHub

```python
# Load model structure
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Load saved weights
quantized_model.load_state_dict(torch.load("path/to/QuickTake.pt"))
quantized_model.eval()
```

### 2. Option 2: Using the code below

#### Load Quantized Mdoel

```python
from transformers import AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Load quantized model (as shown above)
# Assuming quantized_model is already loaded
text = "The Canadian government has introduced a new climate policy to reduce emissions."

# Tokenize
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = quantized_model.generate(inputs["input_ids"], max_length=60, min_length=20, do_sample=False)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("📝 Summary:", summary)
```

---

## 📊 Benefits

| Metric         | Original Model | Quantized Model |
| -------------- | -------------- | --------------- |
| Size (MB)      | \~440MB        | \~220MB         |
| Inference Time | \~1.5s         | \~0.6s          |
| Accuracy Drop  | \~1–2% max     | Acceptable      |

---

## 🧪 Example
See [`quantized_summarizer.ipynb`](https://github.com/harshpatel2312/QuickTake/blob/train_model/model/quantized_summarizer.ipynb) for a notebook demo with timing benchmarks and model usage examples.

---

## 📃 License & Attributions
🧠 Model: [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) — MIT License
📦 Project: © Harsh Patel — MIT License
