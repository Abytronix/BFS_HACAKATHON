#!/usr/bin/env python3
"""
🚀 AI-Powered Stock Analysis Assistant for Google Colab
Complete implementation with BFS stock data analysis and LLM integration
"""

# Install required packages (run this in Google Colab)
"""
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
!pip install PyPDF2>=3.0.0
!pip install huggingface-hub>=0.16.0
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# AI/ML Libraries
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Web interface
import gradio as gr

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 not available. PDF processing will be disabled.")

import io
import os

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Configure display settings
pd.set_option('display.max_columns', None)
plt.style.use('default')  # Use default style for better compatibility
plt.rcParams['figure.figsize'] = (12, 8)

class AIStockAnalysisAssistant:
    def __init__(self, model_name="google/gemma-2b-it", use_quantization=True):
        """
        Initialize the AI-powered stock analysis assistant
        
        Args:
            model_name: Name of the LLM model to use
            use_quantization: Whether to use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.data = None
        self.stock_symbol = None
        self.earnings_data = {}
        
        # Initialize the LLM
        self.setup_llm()
        
    def setup_llm(self):
        """Setup the Large Language Model for natural language processing"""
        try:
            print("🤖 Loading LLM model...")
            
            if self.use_quantization and torch.cuda.is_available():
                # Configure quantization for memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                # Load without quantization for CPU or if quantization is disabled
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
            )
            
            print("✅ LLM model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            print("🔄 Falling back to basic responses...")
            self.generator = None
    
    def load_bfs_data(self, file_path="BFS_Share_Price.csv"):
        """Load BFS stock data from CSV file"""
        try:
            print("📊 Loading BFS stock data...")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"❌ File {file_path} not found. Please upload the file to Colab.")
                return False
            
            # Load the CSV file
            self.data = pd.read_csv(file_path)
            
            # Ensure Date column is datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
            
            # Rename columns for consistency
            if 'Close Price' in self.data.columns:
                self.data.rename(columns={'Close Price': 'Close'}, inplace=True)
            
            self.stock_symbol = "BFS"
            
            # Calculate technical indicators
            self.calculate_technical_indicators()
            
            print(f"✅ Data loaded successfully for {self.stock_symbol}")
            print(f"📈 Data shape: {self.data.shape}")
            print(f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate SMA, EMA, and other technical indicators"""
        try:
            print("🔧 Calculating technical indicators...")
            
            # Calculate Simple Moving Averages
            self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
            
            # Calculate Exponential Moving Averages
            self.data['EMA_10'] = self.data['Close'].ewm(span=10).mean()
            self.data['EMA_20'] = self.data['Close'].ewm(span=20).mean()
            self.data['EMA_50'] = self.data['Close'].ewm(span=50).mean()
            
            # Calculate RSI
            self.data['RSI'] = self.calculate_rsi(self.data['Close'])
            
            # Calculate MACD
            self.data['MACD'], self.data['MACD_signal'] = self.calculate_macd(self.data['Close'])
            
            # Calculate Bollinger Bands
            self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = self.calculate_bollinger_bands(self.data['Close'])
            
            # Calculate daily returns
            self.data['Returns'] = self.data['Close'].pct_change()
            
            # Calculate volatility
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
            
            print("✅ Technical indicators calculated successfully!")
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators: {e}")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    
    def create_comprehensive_chart(self):
        """Create comprehensive interactive charts using Plotly"""
        if self.data is None:
            print("❌ No data loaded. Please load stock data first.")
            return None
        
        try:
            # Create subplot figure with 4 rows
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f'{self.stock_symbol} Stock Price with Moving Averages & Bollinger Bands',
                    'RSI (Relative Strength Index)',
                    'MACD',
                    'Daily Returns & Volatility'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # 1. Price data with moving averages and Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Close'],
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='green', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['EMA_20'],
                    name='EMA 20',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    fill=None
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BB_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )
            
            # 2. RSI
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            # 3. MACD
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD_signal'],
                    name='MACD Signal',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
            
            # MACD histogram
            macd_histogram = self.data['MACD'] - self.data['MACD_signal']
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=macd_histogram,
                    name='MACD Histogram',
                    marker_color=['green' if x >= 0 else 'red' for x in macd_histogram],
                    opacity=0.7
                ),
                row=3, col=1
            )
            
            # 4. Returns and Volatility
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Returns'],
                    name='Daily Returns',
                    line=dict(color='blue', width=1),
                    mode='lines'
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Volatility'],
                    name='20-day Volatility',
                    line=dict(color='orange', width=1),
                    yaxis='y5'
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'📈 {self.stock_symbol} - Comprehensive Stock Analysis Dashboard',
                height=1200,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Update x-axis
            fig.update_xaxes(title_text="Date", row=4, col=1)
            
            # Update y-axes
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Returns", row=4, col=1)
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
            return None
    
    def analyze_stock_performance(self):
        """Analyze stock performance and provide comprehensive insights"""
        if self.data is None:
            return None
        
        try:
            # Basic statistics
            current_price = self.data['Close'].iloc[-1]
            previous_price = self.data['Close'].iloc[-2]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            # Performance metrics
            quarterly_return = self.calculate_period_return(months=3)
            yearly_return = self.calculate_period_return(years=1)
            ytd_return = self.calculate_ytd_return()
            
            # Risk metrics
            volatility = self.data['Returns'].std() * np.sqrt(252) * 100  # Annualized volatility in %
            max_drawdown = self.calculate_max_drawdown()
            sharpe_ratio = self.calculate_sharpe_ratio()
            
            # Technical indicators
            current_rsi = self.data['RSI'].iloc[-1]
            current_macd = self.data['MACD'].iloc[-1]
            current_macd_signal = self.data['MACD_signal'].iloc[-1]
            sma_20 = self.data['SMA_20'].iloc[-1]
            sma_50 = self.data['SMA_50'].iloc[-1]
            sma_200 = self.data['SMA_200'].iloc[-1]
            
            # Support and resistance levels
            support_level = self.data['Close'].rolling(window=50).min().iloc[-1]
            resistance_level = self.data['Close'].rolling(window=50).max().iloc[-1]
            
            analysis = {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'quarterly_return': quarterly_return,
                'yearly_return': yearly_return,
                'ytd_return': ytd_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'current_rsi': current_rsi,
                'current_macd': current_macd,
                'current_macd_signal': current_macd_signal,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'trend_signal': self.get_trend_signal(),
                'momentum_signal': self.get_momentum_signal()
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing data: {e}")
            return None
    
    def calculate_period_return(self, days=None, months=None, years=None):
        """Calculate return for a specific period"""
        try:
            if years:
                period_ago = self.data.index[-1] - pd.DateOffset(years=years)
            elif months:
                period_ago = self.data.index[-1] - pd.DateOffset(months=months)
            elif days:
                period_ago = self.data.index[-1] - pd.DateOffset(days=days)
            else:
                return 0
            
            period_data = self.data[self.data.index >= period_ago]
            
            if len(period_data) < 2:
                return 0
                
            start_price = period_data['Close'].iloc[0]
            end_price = period_data['Close'].iloc[-1]
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception as e:
            return 0
    
    def calculate_ytd_return(self):
        """Calculate year-to-date return"""
        try:
            current_year = self.data.index[-1].year
            ytd_data = self.data[self.data.index.year == current_year]
            
            if len(ytd_data) < 2:
                return 0
                
            start_price = ytd_data['Close'].iloc[0]
            end_price = ytd_data['Close'].iloc[-1]
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception as e:
            return 0
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + self.data['Returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min() * 100
        except Exception as e:
            return 0
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        try:
            excess_returns = self.data['Returns'].mean() * 252 - risk_free_rate
            return excess_returns / (self.data['Returns'].std() * np.sqrt(252))
        except Exception as e:
            return 0
    
    def get_trend_signal(self):
        """Get trend signal based on moving averages"""
        try:
            current_price = self.data['Close'].iloc[-1]
            sma_20 = self.data['SMA_20'].iloc[-1]
            sma_50 = self.data['SMA_50'].iloc[-1]
            sma_200 = self.data['SMA_200'].iloc[-1]
            
            if current_price > sma_20 > sma_50 > sma_200:
                return "🟢 Strong Bullish"
            elif current_price > sma_20 > sma_50:
                return "🟢 Bullish"
            elif current_price < sma_20 < sma_50 < sma_200:
                return "🔴 Strong Bearish"
            elif current_price < sma_20 < sma_50:
                return "🔴 Bearish"
            else:
                return "🟡 Neutral"
                
        except Exception as e:
            return "❓ Unknown"
    
    def get_momentum_signal(self):
        """Get momentum signal based on RSI and MACD"""
        try:
            rsi = self.data['RSI'].iloc[-1]
            macd = self.data['MACD'].iloc[-1]
            macd_signal = self.data['MACD_signal'].iloc[-1]
            
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            macd_signal_trend = "Bullish" if macd > macd_signal else "Bearish"
            
            return f"RSI: {rsi_signal}, MACD: {macd_signal_trend}"
            
        except Exception as e:
            return "Unknown"
    
    def generate_ai_summary(self, analysis_data):
        """Generate AI-powered natural language summary"""
        if analysis_data is None:
            return "❌ No analysis data available"
        
        try:
            if self.generator is not None:
                return self.generate_llm_summary(analysis_data)
            else:
                return self.generate_basic_summary(analysis_data)
                
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return self.generate_basic_summary(analysis_data)
    
    def generate_llm_summary(self, analysis_data):
        """Generate summary using LLM"""
        prompt = f"""You are a professional financial analyst. Provide a comprehensive analysis of BFS stock based on the following data:
        
        Stock: {self.stock_symbol}
        Current Price: ₹{analysis_data['current_price']:.2f}
        Daily Change: ₹{analysis_data['price_change']:.2f} ({analysis_data['price_change_pct']:.2f}%)
        YTD Return: {analysis_data['ytd_return']:.2f}%
        Quarterly Return: {analysis_data['quarterly_return']:.2f}%
        Yearly Return: {analysis_data['yearly_return']:.2f}%
        Volatility: {analysis_data['volatility']:.2f}%
        Max Drawdown: {analysis_data['max_drawdown']:.2f}%
        Sharpe Ratio: {analysis_data['sharpe_ratio']:.2f}
        RSI: {analysis_data['current_rsi']:.2f}
        Trend: {analysis_data['trend_signal']}
        Momentum: {analysis_data['momentum_signal']}
        
        Provide insights on:
        1. Current performance and trend
        2. Risk assessment
        3. Technical analysis
        4. Investment outlook
        
        Keep it professional and concise (max 300 words)."""
        
        try:
            response = self.generator(
                prompt, 
                max_new_tokens=300, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = response[0]['generated_text']
            # Extract only the generated part
            ai_response = generated_text.replace(prompt, '').strip()
            return ai_response if ai_response else self.generate_basic_summary(analysis_data)
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            return self.generate_basic_summary(analysis_data)
    
    def generate_basic_summary(self, analysis_data):
        """Generate basic summary without LLM"""
        trend_emoji = "📈" if "Bullish" in analysis_data['trend_signal'] else "📉" if "Bearish" in analysis_data['trend_signal'] else "➡️"
        risk_level = "🔴 High" if analysis_data['volatility'] > 30 else "🟡 Medium" if analysis_data['volatility'] > 15 else "🟢 Low"
        
        summary = f"""
# 📊 {self.stock_symbol} Stock Analysis Summary

## {trend_emoji} Current Performance
- **Current Price**: ₹{analysis_data['current_price']:.2f}
- **Daily Change**: ₹{analysis_data['price_change']:.2f} ({analysis_data['price_change_pct']:.2f}%)
- **YTD Return**: {analysis_data['ytd_return']:.2f}%
- **Quarterly Return**: {analysis_data['quarterly_return']:.2f}%
- **Yearly Return**: {analysis_data['yearly_return']:.2f}%

## 🔍 Technical Analysis
- **Trend Signal**: {analysis_data['trend_signal']}
- **Momentum**: {analysis_data['momentum_signal']}
- **RSI**: {analysis_data['current_rsi']:.2f} {'🔴 (Overbought)' if analysis_data['current_rsi'] > 70 else '🟢 (Oversold)' if analysis_data['current_rsi'] < 30 else '🟡 (Neutral)'}
- **Price vs SMA 20**: {'🟢 Above' if analysis_data['current_price'] > analysis_data['sma_20'] else '🔴 Below'} (₹{analysis_data['sma_20']:.2f})
- **Price vs SMA 50**: {'🟢 Above' if analysis_data['current_price'] > analysis_data['sma_50'] else '🔴 Below'} (₹{analysis_data['sma_50']:.2f})

## ⚠️ Risk Assessment
- **Volatility**: {analysis_data['volatility']:.2f}% (Annualized)
- **Max Drawdown**: {analysis_data['max_drawdown']:.2f}%
- **Sharpe Ratio**: {analysis_data['sharpe_ratio']:.2f}
- **Risk Level**: {risk_level}

## 🎯 Key Levels
- **Support Level**: ₹{analysis_data['support_level']:.2f}
- **Resistance Level**: ₹{analysis_data['resistance_level']:.2f}

## 💡 Investment Insights
- **Trend Analysis**: {'Strong uptrend with price above all major moving averages' if 'Strong Bullish' in analysis_data['trend_signal'] else 'Bullish trend with positive momentum' if 'Bullish' in analysis_data['trend_signal'] else 'Bearish trend with negative momentum' if 'Bearish' in analysis_data['trend_signal'] else 'Sideways movement with mixed signals'}
- **Risk Profile**: {'High volatility suggests aggressive risk profile' if analysis_data['volatility'] > 30 else 'Moderate volatility indicates balanced risk-reward' if analysis_data['volatility'] > 15 else 'Low volatility suggests conservative investment'}
- **Technical Outlook**: {'RSI indicates potential reversal zone' if analysis_data['current_rsi'] > 70 or analysis_data['current_rsi'] < 30 else 'Technical indicators show balanced momentum'}
        """
        
        return summary
    
    def answer_question(self, question):
        """Answer questions about the stock using AI"""
        if self.data is None:
            return "❌ No data loaded. Please load BFS stock data first."
        
        try:
            question_lower = question.lower()
            analysis_data = self.analyze_stock_performance()
            
            if analysis_data is None:
                return "❌ Unable to analyze data."
            
            # Handle different types of questions
            if "average price" in question_lower:
                return self.handle_price_questions(question_lower)
            elif "trend" in question_lower or "direction" in question_lower:
                return f"📈 Current trend for {self.stock_symbol}: {analysis_data['trend_signal']}\n\n💡 The stock is showing {analysis_data['momentum_signal'].lower()} momentum based on technical indicators."
            elif "volatility" in question_lower or "risk" in question_lower:
                risk_level = '🔴 High' if analysis_data['volatility'] > 30 else '🟡 Medium' if analysis_data['volatility'] > 15 else '🟢 Low'
                return f"⚠️ Risk Analysis for {self.stock_symbol}:\n\n• **Volatility**: {analysis_data['volatility']:.2f}% (annualized)\n• **Risk Level**: {risk_level}\n• **Max Drawdown**: {analysis_data['max_drawdown']:.2f}%\n• **Sharpe Ratio**: {analysis_data['sharpe_ratio']:.2f}\n\n💡 {self._get_risk_interpretation(analysis_data['volatility'], analysis_data['sharpe_ratio'])}"
            elif "performance" in question_lower or "return" in question_lower:
                return f"📈 Performance Summary for {self.stock_symbol}:\n\n• **YTD Return**: {analysis_data['ytd_return']:.2f}%\n• **Quarterly Return**: {analysis_data['quarterly_return']:.2f}%\n• **Yearly Return**: {analysis_data['yearly_return']:.2f}%\n• **Current Price**: ₹{analysis_data['current_price']:.2f}\n\n💡 {self._get_performance_interpretation(analysis_data)}"
            elif "recommendation" in question_lower or "buy" in question_lower or "sell" in question_lower:
                return self.generate_trading_recommendation(analysis_data)
            elif "support" in question_lower or "resistance" in question_lower:
                return f"🎯 Key Levels for {self.stock_symbol}:\n\n• **Support Level**: ₹{analysis_data['support_level']:.2f}\n• **Resistance Level**: ₹{analysis_data['resistance_level']:.2f}\n• **Current Price**: ₹{analysis_data['current_price']:.2f}\n\n💡 Price is {'near resistance' if abs(analysis_data['current_price'] - analysis_data['resistance_level']) < abs(analysis_data['current_price'] - analysis_data['support_level']) else 'near support'} level."
            else:
                # Use LLM for complex questions
                if self.generator is not None:
                    return self.generate_llm_response(question, analysis_data)
                else:
                    return self.generate_ai_summary(analysis_data)
                    
        except Exception as e:
            return f"❌ Error answering question: {e}"
    
    def _get_risk_interpretation(self, volatility, sharpe_ratio):
        """Get risk interpretation"""
        if volatility > 30:
            return "High volatility indicates significant price swings. Suitable for aggressive investors."
        elif volatility > 15:
            return "Moderate volatility suggests balanced risk-reward profile."
        else:
            return "Low volatility indicates stable price movements. Suitable for conservative investors."
    
    def _get_performance_interpretation(self, analysis_data):
        """Get performance interpretation"""
        if analysis_data['yearly_return'] > 15:
            return "Strong yearly performance indicates good growth potential."
        elif analysis_data['yearly_return'] > 0:
            return "Positive yearly returns show steady growth."
        else:
            return "Negative yearly returns suggest challenging market conditions."
    
    def handle_price_questions(self, question):
        """Handle questions about average prices"""
        try:
            if "q1" in question or "first quarter" in question:
                return self.get_quarterly_average("Q1")
            elif "q2" in question or "second quarter" in question:
                return self.get_quarterly_average("Q2")
            elif "q3" in question or "third quarter" in question:
                return self.get_quarterly_average("Q3")
            elif "q4" in question or "fourth quarter" in question:
                return self.get_quarterly_average("Q4")
            elif "2023" in question:
                return self.get_yearly_average(2023)
            elif "2024" in question:
                return self.get_yearly_average(2024)
            else:
                avg_price = self.data['Close'].mean()
                min_price = self.data['Close'].min()
                max_price = self.data['Close'].max()
                return f"📊 Price Statistics for {self.stock_symbol}:\n\n• **Average Price**: ₹{avg_price:.2f}\n• **Minimum Price**: ₹{min_price:.2f}\n• **Maximum Price**: ₹{max_price:.2f}"
        except Exception as e:
            return f"❌ Error calculating average price: {e}"
    
    def get_quarterly_average(self, quarter):
        """Get average price for a specific quarter"""
        try:
            year = 2024  # You can modify this to handle different years
            
            quarter_ranges = {
                "Q1": ("01-01", "03-31"),
                "Q2": ("04-01", "06-30"),
                "Q3": ("07-01", "09-30"),
                "Q4": ("10-01", "12-31")
            }
            
            start_month_day, end_month_day = quarter_ranges[quarter]
            start_date = f"{year}-{start_month_day}"
            end_date = f"{year}-{end_month_day}"
            
            # Filter data for the quarter
            quarterly_data = self.data[
                (self.data.index >= start_date) & 
                (self.data.index <= end_date)
            ]
            
            if len(quarterly_data) == 0:
                return f"❌ No data available for {quarter} {year}"
            
            avg_price = quarterly_data['Close'].mean()
            min_price = quarterly_data['Close'].min()
            max_price = quarterly_data['Close'].max()
            
            return f"📊 {quarter} {year} Analysis for {self.stock_symbol}:\n\n• **Average Price**: ₹{avg_price:.2f}\n• **Minimum Price**: ₹{min_price:.2f}\n• **Maximum Price**: ₹{max_price:.2f}\n• **Trading Days**: {len(quarterly_data)}"
            
        except Exception as e:
            return f"❌ Error calculating quarterly average: {e}"
    
    def get_yearly_average(self, year):
        """Get average price for a specific year"""
        try:
            yearly_data = self.data[self.data.index.year == year]
            
            if len(yearly_data) == 0:
                return f"❌ No data available for {year}"
            
            avg_price = yearly_data['Close'].mean()
            min_price = yearly_data['Close'].min()
            max_price = yearly_data['Close'].max()
            start_price = yearly_data['Close'].iloc[0]
            end_price = yearly_data['Close'].iloc[-1]
            yearly_return = ((end_price - start_price) / start_price) * 100
            
            return f"📊 {year} Analysis for {self.stock_symbol}:\n\n• **Average Price**: ₹{avg_price:.2f}\n• **Start Price**: ₹{start_price:.2f}\n• **End Price**: ₹{end_price:.2f}\n• **Yearly Return**: {yearly_return:.2f}%\n• **Price Range**: ₹{min_price:.2f} - ₹{max_price:.2f}"
            
        except Exception as e:
            return f"❌ Error calculating yearly average: {e}"
    
    def generate_trading_recommendation(self, analysis_data):
        """Generate comprehensive trading recommendation"""
        try:
            rsi = analysis_data['current_rsi']
            trend = analysis_data['trend_signal']
            volatility = analysis_data['volatility']
            macd = analysis_data['current_macd']
            macd_signal = analysis_data['current_macd_signal']
            
            # Technical signals
            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            macd_signal_trend = "Bullish" if macd > macd_signal else "Bearish"
            
            # Overall recommendation
            bullish_signals = 0
            bearish_signals = 0
            
            if "Bullish" in trend:
                bullish_signals += 2
            elif "Bearish" in trend:
                bearish_signals += 2
            
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
            
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                overall_signal = "🟢 BUY"
                confidence = "High" if bullish_signals >= 3 else "Medium"
            elif bearish_signals > bullish_signals:
                overall_signal = "🔴 SELL"
                confidence = "High" if bearish_signals >= 3 else "Medium"
            else:
                overall_signal = "🟡 HOLD"
                confidence = "Low"
            
            risk_category = "🔴 High Risk" if volatility > 30 else "🟡 Medium Risk" if volatility > 15 else "🟢 Low Risk"
            
            recommendation = f"""
💡 **Trading Recommendation for {self.stock_symbol}**

## 🎯 Overall Signal
**{overall_signal}** (Confidence: {confidence})

## 📊 Technical Analysis
• **Trend**: {trend}
• **RSI Signal**: {rsi_signal} ({rsi:.1f})
• **MACD Signal**: {macd_signal_trend}
• **Risk Category**: {risk_category}

## 💰 Price Targets
• **Current Price**: ₹{analysis_data['current_price']:.2f}
• **Support Level**: ₹{analysis_data['support_level']:.2f}
• **Resistance Level**: ₹{analysis_data['resistance_level']:.2f}

## 🎲 Investment Strategy
• **Risk Profile**: {risk_category.split()[1]} Risk
• **Time Horizon**: {'Short-term' if volatility > 25 else 'Medium-term' if volatility > 15 else 'Long-term'}
• **Position Size**: {'Small (2-5%)' if volatility > 30 else 'Medium (5-10%)' if volatility > 15 else 'Standard (10-15%)'}

⚠️ **Disclaimer**: This analysis is for educational purposes only and should not be considered as financial advice. Please consult with a qualified financial advisor before making investment decisions.
            """
            
            return recommendation
            
        except Exception as e:
            return f"❌ Error generating recommendation: {e}"
    
    def generate_llm_response(self, question, analysis_data):
        """Generate response using LLM"""
        context = f"""Stock: {self.stock_symbol}
Current Price: ₹{analysis_data['current_price']:.2f}
Daily Change: {analysis_data['price_change_pct']:.2f}%
Trend: {analysis_data['trend_signal']}
RSI: {analysis_data['current_rsi']:.2f}
Volatility: {analysis_data['volatility']:.2f}%"""
        
        prompt = f"""You are a financial analyst. Answer this question about BFS stock: {question}

Current market data:
{context}

Provide a helpful, concise answer (max 200 words)."""
        
        try:
            response = self.generator(
                prompt, 
                max_new_tokens=200, 
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = response[0]['generated_text']
            ai_response = generated_text.replace(prompt, '').strip()
            return ai_response if ai_response else f"I can help you analyze {self.stock_symbol}. {self.generate_ai_summary(analysis_data)}"
        except Exception as e:
            return f"❌ Error generating LLM response: {e}"

# Utility functions for Google Colab
def download_sample_data():
    """Download sample BFS data if not available"""
    try:
        # Create sample BFS data if CSV is not available
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        # Remove weekends
        dates = dates[dates.weekday < 5]
        
        # Generate realistic stock price data
        np.random.seed(42)
        prices = []
        initial_price = 1500
        price = initial_price
        
        for i in range(len(dates)):
            # Add some trend and volatility
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return with 2% volatility
            price = price * (1 + daily_return)
            prices.append(round(price, 2))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close Price': prices
        })
        
        # Save to CSV
        df.to_csv('BFS_Share_Price.csv', index=False)
        print("✅ Sample BFS data created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def create_gradio_interface(assistant):
    """Create Gradio interface for the stock analysis assistant"""
    
    def analyze_stock():
        """Analyze stock and return summary"""
        if assistant.data is None:
            return "❌ No data loaded. Please load BFS stock data first.", None
        
        analysis_data = assistant.analyze_stock_performance()
        summary = assistant.generate_ai_summary(analysis_data)
        chart = assistant.create_comprehensive_chart()
        
        return summary, chart
    
    def answer_question(question):
        """Answer user question"""
        if not question:
            return "❓ Please ask a question about BFS stock."
        
        return assistant.answer_question(question)
    
    def load_data():
        """Load BFS data"""
        success = assistant.load_bfs_data("BFS_Share_Price.csv")
        if success:
            return "✅ BFS stock data loaded successfully!"
        else:
            return "❌ Failed to load data. Please check if BFS_Share_Price.csv exists."
    
    # Create interface
    with gr.Blocks(title="🚀 AI Stock Analysis Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 AI-Powered Stock Analysis Assistant
            
            ## 📊 BFS Stock Analysis with Large Language Models
            
            This assistant combines financial data analysis with AI to provide:
            - 📈 Advanced technical analysis with multiple indicators
            - 🤖 AI-powered insights using Google Gemma-2B
            - 💬 Natural language Q&A about stock performance
            - 📊 Interactive visualizations and charts
            - 🎯 Trading recommendations and risk assessment
            """
        )
        
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### 📊 Load BFS Stock Data")
            load_btn = gr.Button("📥 Load BFS Data", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)
            
            load_btn.click(load_data, outputs=[load_status])
        
        with gr.Tab("📊 Stock Analysis"):
            gr.Markdown("### 🚀 Comprehensive Stock Analysis")
            analyze_btn = gr.Button("📈 Analyze BFS Stock", variant="primary", size="lg")
            
            with gr.Row():
                analysis_output = gr.Markdown(label="Analysis Summary")
            
            with gr.Row():
                chart_output = gr.Plot(label="Interactive Chart")
            
            analyze_btn.click(
                analyze_stock,
                outputs=[analysis_output, chart_output]
            )
        
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown(
                """
                ### 💬 Ask Questions About BFS Stock
                
                **Example Questions:**
                - What was the average price in Q1 2024?
                - Show me the current trend and momentum
                - How volatile is this stock compared to market standards?
                - Should I buy, sell, or hold BFS stock?
                - What is the performance this year?
                - What are the key support and resistance levels?
                - Give me a risk assessment for this stock
                - What's the trend for 2023 vs 2024?
                """
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about BFS stock analysis...",
                    lines=3,
                    scale=3
                )
                question_btn = gr.Button("🤖 Get AI Answer", variant="primary", scale=1)
            
            answer_output = gr.Markdown(label="AI Response")
            
            # Sample questions as buttons
            gr.Markdown("### 🎯 Quick Questions")
            with gr.Row():
                q1_btn = gr.Button("📊 Current Performance", size="sm")
                q2_btn = gr.Button("📈 Trend Analysis", size="sm")
                q3_btn = gr.Button("⚠️ Risk Assessment", size="sm")
                q4_btn = gr.Button("💡 Trading Recommendation", size="sm")
            
            # Button actions
            question_btn.click(
                answer_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
            
            q1_btn.click(
                lambda: assistant.answer_question("What is the current performance and key metrics?"),
                outputs=[answer_output]
            )
            
            q2_btn.click(
                lambda: assistant.answer_question("What is the current trend and momentum?"),
                outputs=[answer_output]
            )
            
            q3_btn.click(
                lambda: assistant.answer_question("What is the risk assessment and volatility?"),
                outputs=[answer_output]
            )
            
            q4_btn.click(
                lambda: assistant.answer_question("Should I buy, sell, or hold?"),
                outputs=[answer_output]
            )
        
        with gr.Tab("📄 Documentation"):
            gr.Markdown(
                """
                ## 📚 Features & Documentation
                
                ### 🎯 Key Features
                
                1. **📊 Advanced Technical Analysis**
                   - Simple Moving Averages (SMA 10, 20, 50, 200)
                   - Exponential Moving Averages (EMA 10, 20, 50)
                   - Relative Strength Index (RSI)
                   - MACD with Signal Line and Histogram
                   - Bollinger Bands
                   - Support and Resistance Levels
                
                2. **🤖 AI-Powered Insights**
                   - Google Gemma-2B for natural language understanding
                   - Intelligent question answering
                   - Automated analysis summaries
                   - Trading recommendations
                
                3. **📈 Interactive Visualizations**
                   - Multi-panel Plotly charts
                   - Technical indicator overlays
                   - Zoom and pan capabilities
                   - Professional financial chart styling
                
                4. **💡 Risk Management**
                   - Volatility analysis
                   - Maximum drawdown calculation
                   - Sharpe ratio assessment
                   - Risk categorization
                
                5. **🚀 Deployment Ready**
                   - Optimized for Google Colab
                   - Ready for Hugging Face Spaces
                   - Gradio web interface
                   - Professional UI/UX
                
                ### 🛠️ Technical Implementation
                
                - **AI Model**: Google Gemma-2B with 4-bit quantization
                - **Data Processing**: Pandas, NumPy for financial calculations
                - **Visualization**: Plotly for interactive charts
                - **Web Interface**: Gradio for user interaction
                - **Deployment**: Compatible with Hugging Face Spaces
                
                ### 🎨 Usage Tips
                
                1. **Data Loading**: Always load data first using the Setup tab
                2. **Analysis**: Use the Stock Analysis tab for comprehensive insights
                3. **Questions**: Ask natural language questions in the Q&A tab
                4. **Charts**: Interactive charts support zoom, pan, and hover
                5. **Recommendations**: Get AI-powered trading recommendations
                
                ### ⚠️ Disclaimer
                
                This tool is for educational and research purposes only. All analysis and recommendations should not be considered as financial advice. Please consult with qualified financial advisors before making investment decisions.
                """
            )
    
    return demo

# Main execution function for Google Colab
def main():
    """Main function to run in Google Colab"""
    print("🚀 Starting AI-Powered Stock Analysis Assistant")
    print("=" * 60)
    
    # Check if data file exists, if not create sample data
    if not os.path.exists("BFS_Share_Price.csv"):
        print("📊 BFS data file not found. Creating sample data...")
        download_sample_data()
    
    # Initialize the assistant
    print("\n🤖 Initializing AI Stock Analysis Assistant...")
    assistant = AIStockAnalysisAssistant()
    
    # Load data
    print("\n📊 Loading BFS stock data...")
    success = assistant.load_bfs_data("BFS_Share_Price.csv")
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\n📈 Quick Analysis:")
        analysis_data = assistant.analyze_stock_performance()
        if analysis_data:
            print(f"Current Price: ₹{analysis_data['current_price']:.2f}")
            print(f"Daily Change: {analysis_data['price_change_pct']:.2f}%")
            print(f"Trend: {analysis_data['trend_signal']}")
        
        # Create and launch Gradio interface
        print("\n🌐 Creating web interface...")
        demo = create_gradio_interface(assistant)
        
        print("\n🚀 Launching AI Stock Analysis Assistant...")
        print("💡 The web interface will open in a new tab")
        print("🔗 You can also share the public link for demo purposes")
        
        # Launch with public link for sharing
        demo.launch(
            share=True,
            debug=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
    else:
        print("❌ Failed to load data. Please check your setup.")

if __name__ == "__main__":
    main()