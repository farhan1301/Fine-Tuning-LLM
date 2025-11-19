# Quick Start Guide

## Choose Your Path

### 🚀 Option 1: Google Colab (Recommended)

**Use FREE T4 GPU - Train in 1-2 hours!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhan1301/llm-brd-finetuning/blob/main/colab/COLAB_QUICKSTART.ipynb)

**Why Colab?**
- ⚡ 10-20x faster than CPU (1-2 hrs vs 12-24 hrs)
- 🆓 Free T4 GPU access
- 📦 No local setup needed
- 💾 Saves to Google Drive

**Just click the badge above and follow the notebook!**

See `colab/README.md` for detailed instructions.

---

### 💻 Option 2: Local Setup (CPU/GPU)

**Prerequisites:**
- Conda installed
- 8GB+ RAM
- 10GB free disk space

## Local Setup (5 minutes)

### 1. Environment is Ready ✅

Your conda environment `llm-finetuning` is already created!

### 2. Configure API Keys

Edit `.env` file:

```bash
nano .env
```

Add your keys:

```env
HUGGINGFACE_TOKEN=hf_yourTokenHere
ANTHROPIC_API_KEY=sk-ant-yourKeyHere
```

**Get Keys:**
- Hugging Face: https://huggingface.co/settings/tokens
  - Accept Llama 3.2 license: https://huggingface.co/meta-llama/Llama-3.2-1B
- Anthropic: https://console.anthropic.com/

### 3. Activate & Start

```bash
# Activate environment
conda activate llm-finetuning

# Set environment variables
export HUGGINGFACE_TOKEN="your_token"
export ANTHROPIC_API_KEY="your_key"

# Login to Hugging Face
huggingface-cli login

# Launch Jupyter
jupyter notebook
```

### 4. Run Notebooks in Order

Open `notebooks/01_setup.ipynb` and select kernel: **"Python (LLM Fine-tuning)"**

| # | Notebook | Time | Purpose |
|---|----------|------|---------|
| 1 | `01_setup.ipynb` | 10 min | Verify installation |
| 2 | `02_data_generation.ipynb` | 2-4 hrs | Generate training data |
| 3 | `03_data_preparation.ipynb` | 30 min | Format data |
| 4 | `04_training.ipynb` | 12-24 hrs | Fine-tune model (overnight) |
| 5 | `05_evaluation.ipynb` | 1-2 hrs | Evaluate performance |
| 6 | `06_inference.ipynb` | 30 min | Production inference |
| 7 | `07_demo.ipynb` | 30 min | Interactive demo |

## Common Commands

```bash
# Activate environment
conda activate llm-finetuning

# Deactivate
conda deactivate

# Launch Jupyter
jupyter notebook

# Run demo app
python app.py
```

## Troubleshooting

**Kernel not found:**
```bash
conda activate llm-finetuning
python -m ipykernel install --user --name=llm-finetuning --force
```

**Package not found:**
```bash
conda activate llm-finetuning
pip install <package-name>
```

**API key errors:**
- Check `.env` has correct keys
- Verify Llama 3.2 license accepted
- Try: `huggingface-cli login`

## Timeline

- Setup: ✅ Done!
- Data generation: 2-4 hours
- Training: 12-24 hours (overnight)
- Evaluation & demo: 2-3 hours
- **Total: 1.5-2 days**

## See Also

- Full documentation: `README.md`
- Project structure: `README.md#project-structure`
