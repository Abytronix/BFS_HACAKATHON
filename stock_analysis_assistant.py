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

# AI/LLM imports
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import gradio as gr
from datetime import datetime
import re
import io
import base64

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StockAnalysisAssistant:
    def __init__(self, model_name="google/gemma-2b-it"):
        """
        Initialize the AI-powered stock analysis assistant
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.data = None
        self.stock_symbol = None
        
        # Initialize the LLM
        self.setup_llm()
        
    def setup_llm(self):
        """Setup the Large Language Model for natural language processing"""
        try:
            print("Loading LLM model...")
            
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
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("LLM model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            # Fallback to a lighter model or basic responses
            self.generator = None
    
    def load_stock_data(self, symbol_or_file, start_date="2022-01-01", end_date=None):
        """
        Load stock data from Yahoo Finance API or CSV file
        
        Args:
            symbol_or_file: Stock symbol (e.g., 'AAPL') or path to CSV file
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        try:
            if symbol_or_file.endswith('.csv'):
                # Load from CSV file
                self.data = pd.read_csv(symbol_or_file)
                self.stock_symbol = symbol_or_file.split('/')[-1].replace('.csv', '')
                
                # Ensure Date column is datetime
                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    self.data.set_index('Date', inplace=True)
                elif 'date' in self.data.columns:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                    self.data.set_index('date', inplace=True)
                
            else:
                # Load from Yahoo Finance
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                self.data = yf.download(symbol_or_file, start=start_date, end=end_date)
                self.stock_symbol = symbol_or_file
            
            # Calculate technical indicators
            self.calculate_technical_indicators()
            
            print(f"Data loaded successfully for {self.stock_symbol}")
            print(f"Data shape: {self.data.shape}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate SMA, EMA, and other technical indicators"""
        try:
            # Ensure we have the right column names
            if 'Close' not in self.data.columns and 'close' in self.data.columns:
                self.data['Close'] = self.data['close']
            if 'Volume' not in self.data.columns and 'volume' in self.data.columns:
                self.data['Volume'] = self.data['volume']
            
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
            
            print("Technical indicators calculated successfully!")
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
    
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
    
    def create_interactive_chart(self, chart_type="price_with_indicators"):
        """Create interactive charts using Plotly"""
        if self.data is None:
            return "No data loaded. Please load stock data first."
        
        try:
            if chart_type == "price_with_indicators":
                # Create subplot figure
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Price with Moving Averages', 'Volume', 'RSI'),
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Add price data
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
                
                # Add volume
                if 'Volume' in self.data.columns:
                    fig.add_trace(
                        go.Bar(
                            x=self.data.index,
                            y=self.data['Volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                
                # Add RSI
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ),
                    row=3, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                # Update layout
                fig.update_layout(
                    title=f'{self.stock_symbol} - Stock Analysis Dashboard',
                    xaxis_title='Date',
                    height=800,
                    showlegend=True,
                    template='plotly_white'
                )
                
                return fig
                
            elif chart_type == "candlestick":
                # Create candlestick chart
                fig = go.Figure()
                
                if all(col in self.data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    fig.add_trace(
                        go.Candlestick(
                            x=self.data.index,
                            open=self.data['Open'],
                            high=self.data['High'],
                            low=self.data['Low'],
                            close=self.data['Close'],
                            name='Candlestick'
                        )
                    )
                    
                    # Add moving averages
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data['SMA_20'],
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data['SMA_50'],
                            name='SMA 50',
                            line=dict(color='green', width=1)
                        )
                    )
                    
                    fig.update_layout(
                        title=f'{self.stock_symbol} - Candlestick Chart with Moving Averages',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=600,
                        template='plotly_white'
                    )
                    
                    return fig
                else:
                    return "OHLC data not available for candlestick chart"
                    
            else:
                return "Invalid chart type specified"
                
        except Exception as e:
            return f"Error creating chart: {e}"
    
    def analyze_stock_data(self, query_type="summary"):
        """Analyze stock data and provide insights"""
        if self.data is None:
            return "No data loaded. Please load stock data first."
        
        try:
            analysis = {}
            
            # Basic statistics
            current_price = self.data['Close'].iloc[-1]
            price_change = self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]
            price_change_pct = (price_change / self.data['Close'].iloc[-2]) * 100
            
            # Calculate quarterly and yearly statistics
            quarterly_return = self.calculate_quarterly_return()
            yearly_return = self.calculate_yearly_return()
            
            # Risk metrics
            volatility = self.data['Returns'].std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (self.data['Returns'].mean() * 252) / (self.data['Returns'].std() * np.sqrt(252))
            
            # Technical analysis
            current_rsi = self.data['RSI'].iloc[-1]
            sma_20 = self.data['SMA_20'].iloc[-1]
            sma_50 = self.data['SMA_50'].iloc[-1]
            
            analysis = {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'quarterly_return': quarterly_return,
                'yearly_return': yearly_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'current_rsi': current_rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend_signal': self.get_trend_signal()
            }
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing data: {e}"
    
    def calculate_quarterly_return(self):
        """Calculate quarterly returns"""
        try:
            # Get last 3 months of data
            three_months_ago = self.data.index[-1] - pd.DateOffset(months=3)
            quarterly_data = self.data[self.data.index >= three_months_ago]
            
            if len(quarterly_data) < 2:
                return 0
                
            start_price = quarterly_data['Close'].iloc[0]
            end_price = quarterly_data['Close'].iloc[-1]
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception as e:
            return 0
    
    def calculate_yearly_return(self):
        """Calculate yearly returns"""
        try:
            # Get last 12 months of data
            one_year_ago = self.data.index[-1] - pd.DateOffset(years=1)
            yearly_data = self.data[self.data.index >= one_year_ago]
            
            if len(yearly_data) < 2:
                return 0
                
            start_price = yearly_data['Close'].iloc[0]
            end_price = yearly_data['Close'].iloc[-1]
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception as e:
            return 0
    
    def get_trend_signal(self):
        """Get trend signal based on moving averages"""
        try:
            current_price = self.data['Close'].iloc[-1]
            sma_20 = self.data['SMA_20'].iloc[-1]
            sma_50 = self.data['SMA_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                return "Bullish"
            elif current_price < sma_20 < sma_50:
                return "Bearish"
            else:
                return "Neutral"
                
        except Exception as e:
            return "Unknown"
    
    def generate_natural_language_summary(self, analysis_data):
        """Generate natural language summary using LLM"""
        if self.generator is None:
            return self.generate_basic_summary(analysis_data)
        
        try:
            # Create prompt for LLM
            prompt = f"""
            As a financial analyst, provide a comprehensive analysis of the stock {self.stock_symbol} based on the following data:
            
            Current Price: ${analysis_data['current_price']:.2f}
            Price Change: ${analysis_data['price_change']:.2f} ({analysis_data['price_change_pct']:.2f}%)
            Quarterly Return: {analysis_data['quarterly_return']:.2f}%
            Yearly Return: {analysis_data['yearly_return']:.2f}%
            Volatility: {analysis_data['volatility']:.2f}%
            Sharpe Ratio: {analysis_data['sharpe_ratio']:.2f}
            Current RSI: {analysis_data['current_rsi']:.2f}
            SMA 20: ${analysis_data['sma_20']:.2f}
            SMA 50: ${analysis_data['sma_50']:.2f}
            Trend Signal: {analysis_data['trend_signal']}
            
            Please provide:
            1. Overall assessment of the stock's performance
            2. Technical analysis insights
            3. Risk assessment
            4. Potential trading recommendations
            
            Keep the analysis professional and concise (max 300 words).
            """
            
            # Generate response
            response = self.generator(prompt, max_new_tokens=300, temperature=0.7)
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the prompt from the response
            summary = generated_text.replace(prompt, "").strip()
            
            return summary
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return self.generate_basic_summary(analysis_data)
    
    def generate_basic_summary(self, analysis_data):
        """Generate basic summary without LLM"""
        summary = f"""
        📊 **Stock Analysis Summary for {self.stock_symbol}**
        
        **Current Performance:**
        - Current Price: ${analysis_data['current_price']:.2f}
        - Daily Change: ${analysis_data['price_change']:.2f} ({analysis_data['price_change_pct']:.2f}%)
        - Quarterly Return: {analysis_data['quarterly_return']:.2f}%
        - Yearly Return: {analysis_data['yearly_return']:.2f}%
        
        **Technical Indicators:**
        - RSI: {analysis_data['current_rsi']:.2f} {'(Overbought)' if analysis_data['current_rsi'] > 70 else '(Oversold)' if analysis_data['current_rsi'] < 30 else '(Neutral)'}
        - Trend Signal: {analysis_data['trend_signal']}
        - Price vs SMA 20: {'Above' if analysis_data['current_price'] > analysis_data['sma_20'] else 'Below'}
        - Price vs SMA 50: {'Above' if analysis_data['current_price'] > analysis_data['sma_50'] else 'Below'}
        
        **Risk Assessment:**
        - Volatility: {analysis_data['volatility']:.2f}% (Annualized)
        - Sharpe Ratio: {analysis_data['sharpe_ratio']:.2f}
        - Risk Level: {'High' if analysis_data['volatility'] > 30 else 'Medium' if analysis_data['volatility'] > 15 else 'Low'}
        """
        
        return summary
    
    def answer_question(self, question):
        """Answer natural language questions about the stock data"""
        if self.data is None:
            return "No data loaded. Please load stock data first."
        
        try:
            # Analyze the question and extract key information
            question_lower = question.lower()
            
            # Get analysis data
            analysis_data = self.analyze_stock_data()
            
            # Handle different types of questions
            if "average price" in question_lower:
                if "q1" in question_lower or "first quarter" in question_lower:
                    return self.get_quarterly_average("Q1")
                elif "q2" in question_lower or "second quarter" in question_lower:
                    return self.get_quarterly_average("Q2")
                elif "q3" in question_lower or "third quarter" in question_lower:
                    return self.get_quarterly_average("Q3")
                elif "q4" in question_lower or "fourth quarter" in question_lower:
                    return self.get_quarterly_average("Q4")
                else:
                    avg_price = self.data['Close'].mean()
                    return f"The average price of {self.stock_symbol} over the entire period is ${avg_price:.2f}"
            
            elif "trend" in question_lower:
                if "2023" in question_lower:
                    return self.get_yearly_trend("2023")
                elif "2024" in question_lower:
                    return self.get_yearly_trend("2024")
                else:
                    return f"Current trend for {self.stock_symbol}: {analysis_data['trend_signal']}"
            
            elif "moving average" in question_lower or "sma" in question_lower or "ema" in question_lower:
                return self.get_moving_average_analysis()
            
            elif "volatility" in question_lower or "risk" in question_lower:
                return f"The current volatility of {self.stock_symbol} is {analysis_data['volatility']:.2f}% (annualized). This represents a {'high' if analysis_data['volatility'] > 30 else 'medium' if analysis_data['volatility'] > 15 else 'low'} risk level."
            
            elif "performance" in question_lower or "return" in question_lower:
                return f"Performance summary for {self.stock_symbol}: Quarterly return: {analysis_data['quarterly_return']:.2f}%, Yearly return: {analysis_data['yearly_return']:.2f}%"
            
            else:
                # Use LLM for general questions
                if self.generator is not None:
                    return self.generate_llm_response(question, analysis_data)
                else:
                    return self.generate_natural_language_summary(analysis_data)
                    
        except Exception as e:
            return f"Error answering question: {e}"
    
    def get_quarterly_average(self, quarter):
        """Get average price for a specific quarter"""
        try:
            year = 2024  # Default to 2024, can be made dynamic
            
            if quarter == "Q1":
                start_date = f"{year}-01-01"
                end_date = f"{year}-03-31"
            elif quarter == "Q2":
                start_date = f"{year}-04-01"
                end_date = f"{year}-06-30"
            elif quarter == "Q3":
                start_date = f"{year}-07-01"
                end_date = f"{year}-09-30"
            elif quarter == "Q4":
                start_date = f"{year}-10-01"
                end_date = f"{year}-12-31"
            else:
                return "Invalid quarter specified"
            
            # Filter data for the quarter
            quarterly_data = self.data[
                (self.data.index >= start_date) & 
                (self.data.index <= end_date)
            ]
            
            if len(quarterly_data) == 0:
                return f"No data available for {quarter} {year}"
            
            avg_price = quarterly_data['Close'].mean()
            return f"The average price of {self.stock_symbol} in {quarter} {year} was ${avg_price:.2f}"
            
        except Exception as e:
            return f"Error calculating quarterly average: {e}"
    
    def get_yearly_trend(self, year):
        """Get trend analysis for a specific year"""
        try:
            year_data = self.data[self.data.index.year == int(year)]
            
            if len(year_data) == 0:
                return f"No data available for {year}"
            
            start_price = year_data['Close'].iloc[0]
            end_price = year_data['Close'].iloc[-1]
            yearly_return = ((end_price - start_price) / start_price) * 100
            
            # Calculate moving averages trend
            year_data_with_ma = year_data.copy()
            year_data_with_ma['SMA_20'] = year_data_with_ma['Close'].rolling(window=20).mean()
            year_data_with_ma['SMA_50'] = year_data_with_ma['Close'].rolling(window=50).mean()
            
            trend_analysis = f"""
            📈 **{year} Trend Analysis for {self.stock_symbol}:**
            
            - Starting Price: ${start_price:.2f}
            - Ending Price: ${end_price:.2f}
            - Yearly Return: {yearly_return:.2f}%
            - Trend: {'Bullish' if yearly_return > 0 else 'Bearish'}
            
            **Moving Averages Analysis:**
            - Average SMA 20: ${year_data_with_ma['SMA_20'].mean():.2f}
            - Average SMA 50: ${year_data_with_ma['SMA_50'].mean():.2f}
            """
            
            return trend_analysis
            
        except Exception as e:
            return f"Error calculating yearly trend: {e}"
    
    def get_moving_average_analysis(self):
        """Get detailed moving average analysis"""
        try:
            current_price = self.data['Close'].iloc[-1]
            sma_10 = self.data['SMA_10'].iloc[-1]
            sma_20 = self.data['SMA_20'].iloc[-1]
            sma_50 = self.data['SMA_50'].iloc[-1]
            sma_200 = self.data['SMA_200'].iloc[-1]
            
            ema_10 = self.data['EMA_10'].iloc[-1]
            ema_20 = self.data['EMA_20'].iloc[-1]
            ema_50 = self.data['EMA_50'].iloc[-1]
            
            analysis = f"""
            📊 **Moving Average Analysis for {self.stock_symbol}:**
            
            **Current Price:** ${current_price:.2f}
            
            **Simple Moving Averages:**
            - SMA 10: ${sma_10:.2f} {'✅' if current_price > sma_10 else '❌'}
            - SMA 20: ${sma_20:.2f} {'✅' if current_price > sma_20 else '❌'}
            - SMA 50: ${sma_50:.2f} {'✅' if current_price > sma_50 else '❌'}
            - SMA 200: ${sma_200:.2f} {'✅' if current_price > sma_200 else '❌'}
            
            **Exponential Moving Averages:**
            - EMA 10: ${ema_10:.2f} {'✅' if current_price > ema_10 else '❌'}
            - EMA 20: ${ema_20:.2f} {'✅' if current_price > ema_20 else '❌'}
            - EMA 50: ${ema_50:.2f} {'✅' if current_price > ema_50 else '❌'}
            
            **Trend Signal:** {self.get_trend_signal()}
            
            ✅ = Price Above MA | ❌ = Price Below MA
            """
            
            return analysis
            
        except Exception as e:
            return f"Error generating moving average analysis: {e}"
    
    def generate_llm_response(self, question, analysis_data):
        """Generate LLM response for general questions"""
        try:
            # Create context-aware prompt
            prompt = f"""
            You are a financial analyst AI assistant. Answer the following question about {self.stock_symbol} stock based on the provided data:
            
            Question: {question}
            
            Stock Data:
            - Current Price: ${analysis_data['current_price']:.2f}
            - Price Change: ${analysis_data['price_change']:.2f} ({analysis_data['price_change_pct']:.2f}%)
            - Quarterly Return: {analysis_data['quarterly_return']:.2f}%
            - Yearly Return: {analysis_data['yearly_return']:.2f}%
            - Volatility: {analysis_data['volatility']:.2f}%
            - RSI: {analysis_data['current_rsi']:.2f}
            - Trend: {analysis_data['trend_signal']}
            
            Provide a helpful and accurate response in 2-3 sentences.
            """
            
            # Generate response
            response = self.generator(prompt, max_new_tokens=200, temperature=0.7)
            generated_text = response[0]['generated_text']
            
            # Extract answer
            answer = generated_text.replace(prompt, "").strip()
            
            return answer
            
        except Exception as e:
            return f"Error generating LLM response: {e}"

# Create Gradio Interface
def create_gradio_interface():
    """Create Gradio interface for the stock analysis assistant"""
    
    # Initialize the assistant
    assistant = StockAnalysisAssistant()
    
    def load_data_interface(symbol_or_file, start_date, end_date):
        """Interface function to load data"""
        try:
            if end_date == "":
                end_date = None
            
            success = assistant.load_stock_data(symbol_or_file, start_date, end_date)
            
            if success:
                return f"✅ Data loaded successfully for {symbol_or_file}!"
            else:
                return "❌ Failed to load data. Please check the symbol or file path."
                
        except Exception as e:
            return f"❌ Error: {e}"
    
    def create_chart_interface(chart_type):
        """Interface function to create charts"""
        try:
            if assistant.data is None:
                return None, "Please load data first."
            
            fig = assistant.create_interactive_chart(chart_type)
            
            if isinstance(fig, str):
                return None, fig
            else:
                return fig, "Chart created successfully!"
                
        except Exception as e:
            return None, f"Error creating chart: {e}"
    
    def analyze_data_interface():
        """Interface function to analyze data"""
        try:
            if assistant.data is None:
                return "Please load data first."
            
            analysis_data = assistant.analyze_stock_data()
            summary = assistant.generate_natural_language_summary(analysis_data)
            
            return summary
            
        except Exception as e:
            return f"Error analyzing data: {e}"
    
    def answer_question_interface(question):
        """Interface function to answer questions"""
        try:
            if assistant.data is None:
                return "Please load data first."
            
            answer = assistant.answer_question(question)
            return answer
            
        except Exception as e:
            return f"Error answering question: {e}"
    
    # Create Gradio interface
    with gr.Blocks(title="🤖 AI-Powered Stock Analysis Assistant") as demo:
        gr.Markdown("""
        # 🤖 AI-Powered Stock Analysis Assistant
        
        This assistant uses Large Language Models (LLMs) to provide natural language insights on financial data.
        
        ## Features:
        - 📊 Interactive charts with SMA/EMA overlays
        - 🤖 AI-powered natural language summaries
        - 💬 Ask questions in plain English
        - 📈 Technical analysis and risk assessment
        """)
        
        with gr.Tab("📈 Data Loading"):
            gr.Markdown("### Load Stock Data")
            with gr.Row():
                symbol_input = gr.Textbox(
                    label="Stock Symbol or CSV File Path",
                    placeholder="e.g., AAPL, GOOGL, or path/to/bfs_data.csv",
                    value="AAPL"
                )
                start_date_input = gr.Textbox(
                    label="Start Date",
                    placeholder="YYYY-MM-DD",
                    value="2022-01-01"
                )
                end_date_input = gr.Textbox(
                    label="End Date (optional)",
                    placeholder="YYYY-MM-DD"
                )
            
            load_button = gr.Button("📊 Load Data", variant="primary")
            load_output = gr.Textbox(label="Status", interactive=False)
            
            load_button.click(
                load_data_interface,
                inputs=[symbol_input, start_date_input, end_date_input],
                outputs=load_output
            )
        
        with gr.Tab("📊 Interactive Charts"):
            gr.Markdown("### Create Interactive Charts")
            chart_type_dropdown = gr.Dropdown(
                choices=["price_with_indicators", "candlestick"],
                value="price_with_indicators",
                label="Chart Type"
            )
            
            chart_button = gr.Button("📈 Create Chart", variant="primary")
            chart_output = gr.Plot(label="Interactive Chart")
            chart_status = gr.Textbox(label="Status", interactive=False)
            
            chart_button.click(
                create_chart_interface,
                inputs=chart_type_dropdown,
                outputs=[chart_output, chart_status]
            )
        
        with gr.Tab("🤖 AI Analysis"):
            gr.Markdown("### AI-Powered Stock Analysis")
            analyze_button = gr.Button("🤖 Generate AI Analysis", variant="primary")
            analysis_output = gr.Textbox(
                label="AI Analysis",
                lines=15,
                interactive=False
            )
            
            analyze_button.click(
                analyze_data_interface,
                outputs=analysis_output
            )
        
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Ask Questions in Natural Language")
            gr.Markdown("""
            **Example Questions:**
            - What was the average price in Q1 2024?
            - Show me the trend for 2023 with moving averages
            - How volatile is this stock?
            - What's the current performance?
            """)
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What was the average price in Q1 2024?",
                lines=2
            )
            
            ask_button = gr.Button("❓ Ask Question", variant="primary")
            answer_output = gr.Textbox(
                label="AI Answer",
                lines=10,
                interactive=False
            )
            
            ask_button.click(
                answer_question_interface,
                inputs=question_input,
                outputs=answer_output
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Assistant
            
            This AI-powered stock analysis assistant leverages:
            
            - **Large Language Models (LLMs)**: Google Gemma-2B for natural language understanding
            - **Technical Analysis**: SMA, EMA, RSI, MACD, Bollinger Bands
            - **Interactive Visualizations**: Plotly charts with overlays
            - **Natural Language Queries**: Ask questions in plain English
            
            ### How to Use:
            1. **Load Data**: Enter a stock symbol (e.g., AAPL) or CSV file path
            2. **View Charts**: Create interactive charts with technical indicators
            3. **Get AI Analysis**: Generate comprehensive analysis using AI
            4. **Ask Questions**: Query the data in natural language
            
            ### Supported Data Sources:
            - Yahoo Finance API (for real-time data)
            - CSV files (for custom datasets like BFS data)
            
            ### Technical Indicators:
            - Simple Moving Averages (SMA): 10, 20, 50, 200 days
            - Exponential Moving Averages (EMA): 10, 20, 50 days
            - Relative Strength Index (RSI)
            - Moving Average Convergence Divergence (MACD)
            - Bollinger Bands
            
            **Note**: This is for educational purposes only. Not financial advice.
            """)
    
    return demo

# Main execution
if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    
    # Launch with public sharing for Hugging Face Spaces
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )