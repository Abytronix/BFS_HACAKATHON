#!/usr/bin/env python3
"""
Fixed version of Bajaj Finserv Chatbot with better error handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from datetime import datetime, timedelta
import warnings
import re
import os
import PyPDF2
import io
from typing import Dict, List, Any
import json

warnings.filterwarnings('ignore')

class BajajFinservChatbotFixed:
    def __init__(self):
        """Initialize the Bajaj Finserv chatbot with better error handling"""
        self.stock_data = None
        self.transcripts = {}
        self.knowledge_base = {}
        self.data_loaded = False
        self.load_all_data()
        
    def load_all_data(self):
        """Load all available data sources with error handling"""
        try:
            # Load stock price data
            self.load_stock_data()
            
            # Load earnings call transcripts
            self.load_transcripts()
            
            # Create knowledge base
            self.create_knowledge_base()
            
            print("✅ Chatbot initialized successfully!")
            self.data_loaded = True
            
        except Exception as e:
            print(f"⚠️ Error during initialization: {e}")
            print("Chatbot will work with limited functionality")
    
    def load_stock_data(self):
        """Load BFS stock price data with better error handling"""
        try:
            # Check if file exists
            if not os.path.exists('BFS_Share_Price.csv'):
                print("❌ BFS_Share_Price.csv not found in current directory")
                print(f"📁 Current directory: {os.getcwd()}")
                print(f"📂 Files available: {[f for f in os.listdir('.') if f.endswith('.csv')]}")
                return
            
            # Load the CSV file
            self.stock_data = pd.read_csv('BFS_Share_Price.csv')
            print(f"✅ CSV file loaded - {len(self.stock_data)} rows")
            print(f"📊 Columns: {list(self.stock_data.columns)}")
            
            # Check for required columns and fix if needed
            if 'Date' not in self.stock_data.columns:
                # Try to find date column
                date_columns = [col for col in self.stock_data.columns if 'date' in col.lower()]
                if date_columns:
                    self.stock_data.rename(columns={date_columns[0]: 'Date'}, inplace=True)
                    print(f"✅ Renamed '{date_columns[0]}' to 'Date'")
                else:
                    print("❌ No date column found")
                    return
            
            # Check for price column
            if 'Close Price' not in self.stock_data.columns:
                # Try to find price column
                price_columns = [col for col in self.stock_data.columns 
                               if 'close' in col.lower() or 'price' in col.lower()]
                if price_columns:
                    self.stock_data.rename(columns={price_columns[0]: 'Close Price'}, inplace=True)
                    print(f"✅ Renamed '{price_columns[0]}' to 'Close Price'")
                else:
                    print("❌ No price column found")
                    return
            
            # Try different date formats
            date_formats = ['%d-%b-%y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
            date_parsed = False
            
            for fmt in date_formats:
                try:
                    self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], format=fmt)
                    print(f"✅ Date parsing successful with format: {fmt}")
                    date_parsed = True
                    break
                except:
                    continue
            
            if not date_parsed:
                # Try automatic parsing
                try:
                    self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
                    print("✅ Date parsing successful with automatic detection")
                    date_parsed = True
                except Exception as e:
                    print(f"❌ Date parsing failed: {e}")
                    return
            
            # Set date as index
            self.stock_data.set_index('Date', inplace=True)
            
            # Create Close column for compatibility
            if 'Close Price' in self.stock_data.columns and 'Close' not in self.stock_data.columns:
                self.stock_data['Close'] = self.stock_data['Close Price']
            
            # Calculate technical indicators
            self.calculate_technical_indicators()
            
            print("✅ Stock data loaded and processed successfully!")
            
        except Exception as e:
            print(f"❌ Error loading stock data: {e}")
            self.stock_data = None
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators with error handling"""
        try:
            if self.stock_data is None or len(self.stock_data) == 0:
                return
            
            # Calculate Simple Moving Averages
            self.stock_data['SMA_20'] = self.stock_data['Close'].rolling(window=20).mean()
            self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()
            self.stock_data['SMA_200'] = self.stock_data['Close'].rolling(window=200).mean()
            
            # Calculate returns
            self.stock_data['Returns'] = self.stock_data['Close'].pct_change()
            
            # Calculate volatility
            self.stock_data['Volatility'] = self.stock_data['Returns'].rolling(window=20).std()
            
            print("✅ Technical indicators calculated")
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators: {e}")
    
    def load_transcripts(self):
        """Load earnings call transcripts from PDF files"""
        try:
            transcript_files = [
                'Earnings Call Transcript Q1 - FY25  .pdf',
                'Earnings Call Transcript Q2 - FY25.pdf',
                'Earnings Call Transcript Q3 - FY25.pdf',
                'Earnings Call Transcript Q4 - FY25.pdf'
            ]
            
            loaded_count = 0
            for file in transcript_files:
                if os.path.exists(file):
                    quarter = self.extract_quarter_from_filename(file)
                    text = self.extract_text_from_pdf(file)
                    if text:
                        self.transcripts[quarter] = text
                        loaded_count += 1
                        print(f"✅ Loaded {quarter} transcript")
            
            if loaded_count == 0:
                print("⚠️ No PDF transcripts found - transcript analysis will be limited")
            else:
                print(f"✅ Loaded {loaded_count} transcript files")
                
        except Exception as e:
            print(f"❌ Error loading transcripts: {e}")
    
    def extract_quarter_from_filename(self, filename):
        """Extract quarter from filename"""
        if 'Q1' in filename:
            return 'Q1 FY25'
        elif 'Q2' in filename:
            return 'Q2 FY25'
        elif 'Q3' in filename:
            return 'Q3 FY25'
        elif 'Q4' in filename:
            return 'Q4 FY25'
        return 'Unknown'
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"❌ Error reading PDF {pdf_path}: {e}")
            return ""
    
    def create_knowledge_base(self):
        """Create a knowledge base with key information"""
        self.knowledge_base = {
            'companies': {
                'BAGIC': 'Bajaj Allianz General Insurance Company',
                'BALIC': 'Bajaj Allianz Life Insurance Company',
                'Bajaj Markets': 'Digital marketplace for financial products',
                'Bajaj Finance': 'NBFC arm of Bajaj Finserv',
                'Bajaj Housing Finance': 'Housing finance subsidiary'
            },
            'key_topics': {
                'hero_partnership': ['Hero', 'partnership', 'collaboration', 'strategic alliance'],
                'allianz_stake': ['Allianz', 'stake', 'divestment', 'sale', 'equity'],
                'motor_insurance': ['motor insurance', 'automobile insurance', 'vehicle insurance'],
                'organic_traffic': ['organic traffic', 'digital marketing', 'online acquisition'],
                'bagic_headwinds': ['BAGIC', 'headwinds', 'challenges', 'motor insurance']
            }
        }
    
    def answer_question(self, question):
        """Main function to answer questions with better error handling"""
        try:
            # Check if data is loaded
            if not self.data_loaded:
                return "⚠️ System is starting up. Please wait and try again in a moment."
            
            if self.stock_data is None:
                return "❌ Stock data is not available. Please check if BFS_Share_Price.csv is in the correct location with proper format."
            
            question_lower = question.lower()
            
            # Route to appropriate handler based on question type
            if any(keyword in question_lower for keyword in ['stock price', 'highest', 'lowest', 'average price']):
                return self.handle_stock_price_questions(question)
            elif any(keyword in question_lower for keyword in ['compare', 'comparison']):
                return self.handle_comparison_questions(question)
            elif any(keyword in question_lower for keyword in ['organic traffic', 'bajaj markets']):
                return self.handle_organic_traffic_questions(question)
            elif any(keyword in question_lower for keyword in ['bagic', 'motor insurance', 'headwinds']):
                return self.handle_bagic_questions(question)
            elif any(keyword in question_lower for keyword in ['hero', 'partnership']):
                return self.handle_hero_partnership_questions(question)
            elif any(keyword in question_lower for keyword in ['allianz', 'stake', 'sale']):
                return self.handle_allianz_stake_questions(question)
            elif any(keyword in question_lower for keyword in ['cfo', 'investor call', 'commentary']):
                return self.handle_cfo_commentary_questions(question)
            else:
                return self.handle_general_questions(question)
                
        except Exception as e:
            return f"❌ Error processing question: {e}\n\nPlease try rephrasing your question or check the system status."
    
    def handle_stock_price_questions(self, question):
        """Handle stock price related questions with error handling"""
        try:
            if self.stock_data is None or len(self.stock_data) == 0:
                return "❌ No stock data available. Please ensure BFS_Share_Price.csv is loaded correctly."
            
            question_lower = question.lower()
            
            # Extract date range if mentioned
            month = None
            if 'jan' in question_lower or 'january' in question_lower:
                month = 1
            elif 'feb' in question_lower or 'february' in question_lower:
                month = 2
            elif 'mar' in question_lower or 'march' in question_lower:
                month = 3
            elif 'apr' in question_lower or 'april' in question_lower:
                month = 4
            elif 'may' in question_lower:
                month = 5
            elif 'jun' in question_lower or 'june' in question_lower:
                month = 6
            elif 'jul' in question_lower or 'july' in question_lower:
                month = 7
            elif 'aug' in question_lower or 'august' in question_lower:
                month = 8
            elif 'sep' in question_lower or 'september' in question_lower:
                month = 9
            elif 'oct' in question_lower or 'october' in question_lower:
                month = 10
            elif 'nov' in question_lower or 'november' in question_lower:
                month = 11
            elif 'dec' in question_lower or 'december' in question_lower:
                month = 12
            
            # Extract year
            year = None
            if '2024' in question_lower or '24' in question_lower:
                year = 2024
            elif '2023' in question_lower or '23' in question_lower:
                year = 2023
            elif '2022' in question_lower or '22' in question_lower:
                year = 2022
            
            # Filter data based on date
            if month and year:
                filtered_data = self.stock_data[
                    (self.stock_data.index.month == month) & 
                    (self.stock_data.index.year == year)
                ]
                period = f"{month:02d}-{year}"
            elif year:
                filtered_data = self.stock_data[self.stock_data.index.year == year]
                period = str(year)
            else:
                filtered_data = self.stock_data
                period = "entire available period"
            
            if len(filtered_data) == 0:
                return f"❌ No data available for the specified period: {period}\n\nAvailable data range: {self.stock_data.index.min().strftime('%d-%b-%Y')} to {self.stock_data.index.max().strftime('%d-%b-%Y')}"
            
            # Calculate statistics
            highest_price = filtered_data['Close'].max()
            lowest_price = filtered_data['Close'].min()
            average_price = filtered_data['Close'].mean()
            
            # Get dates for highest and lowest
            highest_date = filtered_data[filtered_data['Close'] == highest_price].index[0]
            lowest_date = filtered_data[filtered_data['Close'] == lowest_price].index[0]
            
            if 'highest' in question_lower:
                return f"📈 **Highest Stock Price for {period}:**\n\n" + \
                       f"**Price:** ₹{highest_price:.2f}\n" + \
                       f"**Date:** {highest_date.strftime('%d-%b-%Y')}\n\n" + \
                       f"📊 **Additional Context:**\n" + \
                       f"• Average price during this period: ₹{average_price:.2f}\n" + \
                       f"• Lowest price during this period: ₹{lowest_price:.2f} ({lowest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Data points analyzed: {len(filtered_data)} trading days"
            
            elif 'lowest' in question_lower:
                return f"📉 **Lowest Stock Price for {period}:**\n\n" + \
                       f"**Price:** ₹{lowest_price:.2f}\n" + \
                       f"**Date:** {lowest_date.strftime('%d-%b-%Y')}\n\n" + \
                       f"📊 **Additional Context:**\n" + \
                       f"• Average price during this period: ₹{average_price:.2f}\n" + \
                       f"• Highest price during this period: ₹{highest_price:.2f} ({highest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Data points analyzed: {len(filtered_data)} trading days"
            
            else:  # average
                return f"📊 **Average Stock Price for {period}:**\n\n" + \
                       f"**Average Price:** ₹{average_price:.2f}\n\n" + \
                       f"📈 **Price Range:**\n" + \
                       f"• Highest: ₹{highest_price:.2f} ({highest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Lowest: ₹{lowest_price:.2f} ({lowest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Volatility: {((highest_price - lowest_price) / average_price * 100):.2f}%\n" + \
                       f"• Data points analyzed: {len(filtered_data)} trading days"
            
        except Exception as e:
            return f"❌ Error analyzing stock price: {e}\n\nPlease check your data format or try a different question."
    
    def handle_general_questions(self, question):
        """Handle general questions"""
        return "I can help you with:\n\n" + \
               "📊 **Stock Analysis**: Ask about highest, lowest, or average prices\n" + \
               "📈 **Comparisons**: Compare performance between time periods\n" + \
               "🏢 **Business Insights**: Questions about BAGIC, partnerships, etc.\n" + \
               "💼 **CFO Commentary**: Help draft investor call commentary\n\n" + \
               "**Sample questions:**\n" + \
               "• 'What was the highest stock price in 2024?'\n" + \
               "• 'Compare BFS from Jan-24 to Mar-24'\n" + \
               "• 'Tell me about BAGIC motor insurance headwinds'\n\n" + \
               f"**Data Status:**\n" + \
               f"• Stock Data: {'✅ Loaded' if self.stock_data is not None else '❌ Not Available'}\n" + \
               f"• Transcripts: {'✅ Available' if self.transcripts else '❌ Not Available'}"

# Initialize the fixed chatbot
chatbot = BajajFinservChatbotFixed()

# Create Gradio interface
def chat_interface(message, history):
    """Chat interface function"""
    try:
        response = chatbot.answer_question(message)
        return response
    except Exception as e:
        return f"❌ Error: {e}\n\nPlease try again or check the system status."

# Create Gradio app
with gr.Blocks(title="Bajaj Finserv AI Chatbot (Fixed)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 Bajaj Finserv AI Chatbot (Fixed Version)
    
    **🔧 This is an improved version with better error handling and debugging.**
    
    ## ✅ What I Can Do:
    - **Stock Price Analysis**: Highest/lowest/average prices across different periods
    - **Performance Comparison**: Compare BFS performance between different time periods  
    - **Business Intelligence**: Information from earnings call transcripts
    - **CFO Commentary**: Draft investor call commentary
    
    ## 📊 System Status:
    The chatbot will show you exactly what data is available and any issues found.
    """)
    
    chatbot_interface = gr.ChatInterface(
        fn=chat_interface,
        title="💬 Ask me anything about Bajaj Finserv!",
        description="I'll provide detailed analysis and let you know if there are any data issues.",
        examples=[
            "What was the highest stock price in 2024?",
            "Show me the average price in 2023",
            "What's the system status?",
            "Tell me about available data"
        ],
        theme="soft"
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)