# Google Colab Fine-tuning (Recommended)

Train on **FREE T4 GPU** instead of CPU - 10-20x faster!

## ⚡ Quick Start (5 minutes)

**Click to open in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhan1301/Fine-Tuning-LLM/blob/main/colab/COLAB_QUICKSTART.ipynb)

This all-in-one notebook does everything:
1. Setup + verify GPU
2. Generate training data
3. Train model
4. Test results

## ⏱️ Time Comparison

| Method | Hardware | Time | Cost |
|--------|----------|------|------|
| **Colab (Recommended)** | T4 GPU (free) | **1-2 hours** | Free (+ API costs) |
| Local CPU | Intel MacBook | 12-24 hours | Free (+ API costs) |

**You save 10-20 hours using Colab!**

## 📋 Before Starting

### 1. Get API Keys

- **Hugging Face**: https://huggingface.co/settings/tokens
  - Create token (read access)
  - Accept Llama 3.2 license: https://huggingface.co/meta-llama/Llama-3.2-1B

- **Anthropic**: https://console.anthropic.com/
  - Get API key for data generation
  - Estimated cost: $3-5 for 1000 samples

### 2. Enable GPU in Colab

- Click: **Runtime** → **Change runtime type**
- Select: **T4 GPU**
- Click: **Save**

### 3. Run the Notebook

Open `COLAB_QUICKSTART.ipynb` and run all cells!

## 📁 Files Saved to Google Drive

All outputs save to: `Google Drive/LLM_Finetuning/`

- **models/llama-3.2-1b-brd/final/** - Your fine-tuned model
- **data/** - Generated training data

## 🚀 Alternative: Step-by-Step Notebooks

If you prefer separate notebooks for each step:

1. `00_colab_setup.ipynb` - Setup and verify (10 min)
2. Generate data using `COLAB_QUICKSTART.ipynb` (skip training)
3. Train using local notebooks with generated data

## 💡 Tips

### Save Your Work
- Models auto-save to Google Drive
- Colab disconnects after ~12 hours
- You can resume from checkpoints

### Free GPU Limits
- ~15 hours/day of T4 GPU (free tier)
- Enough for multiple training runs
- Upgrade to Colab Pro for more

### API Cost Optimization
- Start with 100 samples (~$0.50) for testing
- Scale to 1000 samples (~$5) for full training
- Results are similar with 500+ samples

## 🔧 Troubleshooting

**"No GPU available"**
- Check: Runtime → Change runtime type → T4 GPU

**"API key invalid"**
- Verify key in Colab Secrets (🔑 icon)
- Or enter manually when prompted

**"Llama access denied"**
- Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B
- Wait 5-10 minutes for approval

**"Out of memory"**
- Reduce batch size in training config
- T4 has 16GB - should be enough for 1B model

## 📊 What You'll Get

After training:
- Fine-tuned Llama 3.2 1B model
- LoRA adapters (~10-50 MB)
- Training metrics and logs
- Test results

## 🎯 Next Steps

After training:
1. Download model from Google Drive
2. Test locally or in another Colab
3. Deploy with FastAPI/Gradio
4. Share on Hugging Face for portfolio

## 🆚 Colab vs Local

**Use Colab if:**
- You want fast results (1-2 hours)
- Have limited local compute
- Want to try quickly
- Training multiple models

**Use Local if:**
- You have good GPU
- Want full control
- Need privacy (no cloud)
- Doing development work

## ⚠️ Important Notes

1. **Colab sessions expire** - Save frequently!
2. **Free tier limits** - ~15 hrs/day GPU
3. **Data stays in Drive** - Models persist
4. **Can resume** - Training checkpoints saved

---

**Ready?** Click the badge above to start! ⬆️
