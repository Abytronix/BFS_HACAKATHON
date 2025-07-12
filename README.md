# 🤖 AI-Powered Stock Analysis Assistant

An intelligent stock analysis assistant that leverages Large Language Models (LLMs) to provide dynamic insights on financial data with natural language interaction capabilities.

## 🚀 Features

- **🧠 AI-Powered Analysis**: Uses Google Gemma-2B LLM for natural language understanding and generation
- **📊 Interactive Charts**: Plotly-based visualizations with technical indicators
- **💬 Natural Language Queries**: Ask questions in plain English about stock data
- **📈 Technical Analysis**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **🎯 Risk Assessment**: Volatility analysis and Sharpe ratio calculations
- **📱 Web Interface**: Beautiful Gradio-based UI for easy interaction
- **🔗 Multiple Data Sources**: Support for Yahoo Finance API and CSV files

## 🛠️ Installation

### For Google Colab (Recommended)

```python
# Install required packages
!pip install pandas numpy matplotlib seaborn plotly yfinance
!pip install torch transformers accelerate bitsandbytes gradio
!pip install scipy scikit-learn datasets tokenizers huggingface-hub

# Clone or download the script
!wget https://raw.githubusercontent.com/your-repo/stock_analysis_assistant.py

# Run the assistant
%run stock_analysis_assistant.py
```

### For Local Development

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-stock-analysis-assistant.git
cd ai-stock-analysis-assistant

# Install dependencies
pip install -r requirements.txt

# Run the assistant
python stock_analysis_assistant.py
```

## 📖 Usage

### 1. Loading Data

**Via Stock Symbol:**
```python
# Load Apple stock data
symbol = "AAPL"
start_date = "2022-01-01"
end_date = "2024-12-31"
```

**Via CSV File:**
```python
# Load your BFS data or any CSV file
csv_path = "path/to/your/bfs_data.csv"
```

### 2. Interactive Charts

The assistant creates interactive charts with:
- **Price Chart with Moving Averages**: SMA 20, SMA 50, EMA 20
- **Bollinger Bands**: Upper and lower bands
- **Volume Analysis**: Trading volume bars
- **RSI Indicator**: Relative Strength Index with overbought/oversold levels
- **Candlestick Charts**: OHLC visualization

### 3. AI-Powered Analysis

The LLM provides comprehensive analysis including:
- **Performance Assessment**: Current price, daily/quarterly/yearly returns
- **Technical Indicators**: RSI, moving averages, trend signals
- **Risk Analysis**: Volatility, Sharpe ratio, risk level
- **Trading Recommendations**: Based on technical analysis

### 4. Natural Language Queries

Ask questions like:
- "What was the average price in Q1 2024?"
- "Show me the trend for 2023 with moving averages"
- "How volatile is this stock?"
- "What's the current performance?"
- "Is this stock overbought or oversold?"

## 🎯 Technical Indicators

### Moving Averages
- **SMA (Simple Moving Average)**: 10, 20, 50, 200 days
- **EMA (Exponential Moving Average)**: 10, 20, 50 days

### Technical Oscillators
- **RSI (Relative Strength Index)**: 14-day period
- **MACD (Moving Average Convergence Divergence)**: 12/26/9 settings
- **Bollinger Bands**: 20-day period with 2 standard deviations

### Risk Metrics
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return measure
- **Daily Returns**: Percentage change calculations

## 🤖 AI Model Information

- **Model**: Google Gemma-2B Instruction Tuned
- **Quantization**: 4-bit quantization for efficiency
- **Context**: Financial analysis and stock market expertise
- **Capabilities**: Natural language understanding, technical analysis interpretation

## 📊 Data Sources

### Yahoo Finance API
- Real-time stock data
- Historical OHLCV data
- Automatic data retrieval

### CSV Files
- Custom datasets (like BFS data)
- Historical data files
- Quarterly reports

**Required CSV Format:**
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.0,155.0,149.0,154.0,1000000
2024-01-02,154.0,156.0,153.0,155.5,1200000
...
```

## 🚀 Deployment

### Hugging Face Spaces
1. Fork this repository
2. Create a new Hugging Face Space
3. Upload the files
4. Set the SDK to "gradio"
5. The app will automatically deploy

### Google Colab
1. Open Google Colab
2. Install dependencies
3. Run the script
4. Use the public URL provided by Gradio

## 📋 Example Use Cases

### 1. BFS Stock Analysis
```python
# Load BFS data from your GitHub repository
assistant = StockAnalysisAssistant()
assistant.load_stock_data("path/to/bfs_data.csv")

# Generate AI analysis
analysis = assistant.analyze_stock_data()
summary = assistant.generate_natural_language_summary(analysis)
print(summary)
```

### 2. Quarterly Performance Analysis
```python
# Ask about specific quarters
question = "What was the average price in Q1 2024?"
answer = assistant.answer_question(question)
print(answer)
```

### 3. Technical Analysis
```python
# Create interactive charts
chart = assistant.create_interactive_chart("price_with_indicators")
chart.show()
```

## 🔧 Configuration

### Model Selection
```python
# Use different LLM models
assistant = StockAnalysisAssistant(model_name="microsoft/DialoGPT-medium")
# or
assistant = StockAnalysisAssistant(model_name="google/gemma-2b-it")
```

### Chart Customization
```python
# Different chart types
chart_types = ["price_with_indicators", "candlestick"]
```

## 📝 Sample Questions

- **Performance**: "How has the stock performed this year?"
- **Technical**: "What do the moving averages tell us about the trend?"
- **Risk**: "How risky is this investment?"
- **Comparison**: "Compare Q1 and Q2 performance"
- **Prediction**: "What signals indicate future movement?"

## 🚨 Limitations & Disclaimers

- **Educational Purpose**: This tool is for educational and research purposes only
- **Not Financial Advice**: Do not use for actual trading decisions
- **Data Accuracy**: Relies on external data sources
- **Model Limitations**: LLM responses may not always be accurate
- **Market Risk**: Past performance doesn't guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Hugging Face Spaces Demo](https://huggingface.co/spaces/your-username/ai-stock-analysis)
- [Google Colab Notebook](https://colab.research.google.com/drive/your-notebook-id)
- [Documentation](https://your-docs-link.com)

## 📧 Contact

For questions, suggestions, or support:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Email: your-email@example.com

---

**⚠️ Important**: This is a demonstration tool for educational purposes. Always consult with financial professionals before making investment decisions.