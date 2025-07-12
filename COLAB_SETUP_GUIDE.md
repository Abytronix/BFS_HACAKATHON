# 🚀 Google Colab Setup Guide - AI Stock Analysis Assistant

## 📋 Quick Start for Google Colab

### 🎯 Overview
This guide will help you set up and run the AI-powered BFS stock analysis assistant in Google Colab with Google Gemma-2B integration.

---

## 📦 Step 1: Setup Google Colab

1. **Open Google Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. **Create New Notebook**: Click "New Notebook"
3. **Enable GPU**: Go to `Runtime > Change runtime type > Hardware accelerator > GPU (T4)`
4. **Connect**: Click the "Connect" button in the top-right corner

---

## 💻 Step 2: Install Dependencies

Copy and paste this code into the first cell and run it:

```python
# Install required packages
!pip install transformers>=4.36.0
!pip install torch>=2.0.0
!pip install accelerate>=0.20.0
!pip install bitsandbytes>=0.41.0
!pip install gradio>=4.0.0
!pip install plotly>=5.0.0
!pip install yfinance>=0.2.0
!pip install pandas>=1.5.0
!pip install numpy>=1.24.0
!pip install matplotlib>=3.7.0
!pip install seaborn>=0.12.0
!pip install huggingface-hub>=0.16.0

print("✅ All packages installed successfully!")
```

---

## 📁 Step 3: Upload Your Data

### Option A: Upload BFS_Share_Price.csv

```python
from google.colab import files
import os

# Upload your BFS data file
if not os.path.exists('BFS_Share_Price.csv'):
    print("📁 Please upload your BFS_Share_Price.csv file:")
    uploaded = files.upload()
    
    if 'BFS_Share_Price.csv' in uploaded:
        print("✅ BFS data uploaded successfully!")
    else:
        print("⚠️ BFS_Share_Price.csv not found.")
else:
    print("✅ BFS_Share_Price.csv already exists!")
```

### Option B: Create Sample Data (if you don't have the CSV)

```python
import pandas as pd
import numpy as np

# Create sample BFS data
print("📊 Creating sample BFS stock data...")
dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
dates = dates[dates.weekday < 5]  # Remove weekends

np.random.seed(42)
prices = []
initial_price = 1500
price = initial_price

for i in range(len(dates)):
    daily_return = np.random.normal(0.0005, 0.02)
    price = price * (1 + daily_return)
    prices.append(round(price, 2))

df = pd.DataFrame({
    'Date': dates,
    'Close Price': prices
})

df.to_csv('BFS_Share_Price.csv', index=False)
print("✅ Sample BFS data created successfully!")
```

---

## 🤖 Step 4: Load the AI Assistant

Copy the complete assistant code into a new cell:

```python
# Complete AI Stock Analysis Assistant
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# [COPY THE COMPLETE CODE FROM colab_stock_assistant.py HERE]
```

**Important**: Copy the entire content of `colab_stock_assistant.py` into this cell.

---

## 🚀 Step 5: Run the Assistant

### Quick Launch (Recommended)

```python
# Run the complete assistant
if __name__ == "__main__":
    main()
```

### Manual Step-by-Step

```python
# 1. Initialize Assistant
print("🤖 Initializing AI Stock Analysis Assistant...")
assistant = AIStockAnalysisAssistant()

# 2. Load Data
success = assistant.load_bfs_data("BFS_Share_Price.csv")
if success:
    print("✅ Data loaded successfully!")
else:
    print("❌ Failed to load data")

# 3. Generate Analysis
analysis_data = assistant.analyze_stock_performance()
summary = assistant.generate_ai_summary(analysis_data)
print(summary)

# 4. Create Interactive Chart
fig = assistant.create_comprehensive_chart()
if fig:
    fig.show()

# 5. Launch Web Interface
demo = create_gradio_interface(assistant)
demo.launch(share=True, debug=True)
```

---

## 💬 Step 6: Test the Q&A System

```python
# Test questions
test_questions = [
    "What was the average price in Q1 2024?",
    "What is the current trend?",
    "How volatile is this stock?",
    "Should I buy or sell?",
    "What are the key support and resistance levels?",
    "Give me a risk assessment",
    "What's the performance this year?"
]

for question in test_questions:
    print(f"\n❓ Question: {question}")
    answer = assistant.answer_question(question)
    print(f"🤖 Answer: {answer}")
    print("-" * 50)
```

---

## 🌐 Step 7: Access the Web Interface

After running the assistant, you'll see:

1. **Local URL**: `http://127.0.0.1:7860` (for Colab internal access)
2. **Public URL**: `https://xxxxx.gradio.live` (shareable link)

The web interface includes:
- **Setup Tab**: Load BFS data
- **Analysis Tab**: Comprehensive stock analysis
- **Q&A Tab**: Natural language questions
- **Documentation Tab**: Features and usage guide

---

## 🎯 Features You Can Use

### 📊 Technical Analysis
- Simple Moving Averages (SMA 10, 20, 50, 200)
- Exponential Moving Averages (EMA 10, 20, 50)
- Relative Strength Index (RSI)
- MACD with Signal Line and Histogram
- Bollinger Bands
- Support and Resistance Levels

### 🤖 AI-Powered Features
- Natural language understanding with Google Gemma-2B
- Intelligent question answering
- Trading recommendations
- Risk assessment
- Performance analysis

### 📈 Interactive Visualizations
- Multi-panel Plotly charts
- Technical indicator overlays
- Zoom and pan capabilities
- Professional financial styling

### 💬 Example Questions You Can Ask
- "What was the average price in Q1 2024?"
- "Show me the current trend and momentum"
- "How volatile is this stock compared to market standards?"
- "Should I buy, sell, or hold BFS stock?"
- "What is the performance this year?"
- "What are the key support and resistance levels?"
- "Give me a risk assessment for this stock"
- "What's the trend for 2023 vs 2024?"

---

## 🛠️ Troubleshooting

### Common Issues:

1. **GPU Memory Error**:
   ```python
   # Use smaller model or disable quantization
   assistant = AIStockAnalysisAssistant(use_quantization=False)
   ```

2. **Model Loading Error**:
   ```python
   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

3. **Data Loading Error**:
   ```python
   # Check file format
   import pandas as pd
   df = pd.read_csv('BFS_Share_Price.csv')
   print(df.head())
   print(df.columns)
   ```

4. **Gradio Interface Not Loading**:
   ```python
   # Try without sharing
   demo.launch(share=False, debug=True)
   ```

---

## 🚀 Deployment to Hugging Face Spaces

1. Create account on [Hugging Face](https://huggingface.co/)
2. Create new Space with Gradio
3. Upload these files:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `BFS_Share_Price.csv` (data file)
4. Set hardware to GPU for better performance

---

## 📝 Tips for Best Performance

1. **Use GPU**: Enable GPU in Colab for faster LLM inference
2. **Restart Runtime**: If you encounter memory issues, restart and run again
3. **Monitor Resources**: Check GPU memory usage periodically
4. **Save Progress**: Download important results before session ends
5. **Share Links**: Use the public Gradio link to share with others

---

## ⚠️ Important Disclaimers

- **Educational Purpose**: This tool is for educational and research purposes only
- **Not Financial Advice**: All analysis should not be considered as financial advice
- **Risk Warning**: Please consult qualified financial advisors before making investment decisions
- **Data Accuracy**: Ensure your data is accurate and up-to-date
- **Model Limitations**: AI responses may not always be perfect or complete

---

## 🔗 Useful Links

- [Google Colab](https://colab.research.google.com/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your data file format matches the expected structure
4. Try restarting the Colab runtime
5. Check GPU availability and memory

---

**🎉 You're now ready to use your AI-Powered Stock Analysis Assistant!**

Happy analyzing! 📈🤖