#!/usr/bin/env python3
"""
Bajaj Finserv AI Chatbot
A comprehensive chatbot that can analyze stock price data and earnings call transcripts
to answer questions about Bajaj Finserv's financial performance, strategic decisions, and market insights.
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

class BajajFinservChatbot:
    def __init__(self):
        """Initialize the Bajaj Finserv chatbot"""
        self.stock_data = None
        self.transcripts = {}
        self.knowledge_base = {}
        self.load_all_data()
        
    def load_all_data(self):
        """Load all available data sources"""
        try:
            # Load stock price data
            self.load_stock_data()
            
            # Load earnings call transcripts
            self.load_transcripts()
            
            # Create knowledge base
            self.create_knowledge_base()
            
            print("All data loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def load_stock_data(self):
        """Load BFS stock price data"""
        try:
            if os.path.exists('BFS_Share_Price.csv'):
                self.stock_data = pd.read_csv('BFS_Share_Price.csv')
                
                # Convert Date column to datetime
                self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], format='%d-%b-%y')
                self.stock_data.set_index('Date', inplace=True)
                
                # Rename column to match standard format
                if 'Close Price' in self.stock_data.columns:
                    self.stock_data['Close'] = self.stock_data['Close Price']
                
                # Calculate technical indicators
                self.calculate_technical_indicators()
                
                print("Stock data loaded successfully!")
                
        except Exception as e:
            print(f"Error loading stock data: {e}")
    
    def load_transcripts(self):
        """Load earnings call transcripts from PDF files"""
        try:
            transcript_files = [
                'Earnings Call Transcript Q1 - FY25  .pdf',
                'Earnings Call Transcript Q2 - FY25.pdf',
                'Earnings Call Transcript Q3 - FY25.pdf',
                'Earnings Call Transcript Q4 - FY25.pdf'
            ]
            
            for file in transcript_files:
                if os.path.exists(file):
                    quarter = self.extract_quarter_from_filename(file)
                    text = self.extract_text_from_pdf(file)
                    self.transcripts[quarter] = text
                    print(f"Loaded {quarter} transcript")
                    
        except Exception as e:
            print(f"Error loading transcripts: {e}")
    
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
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for stock data"""
        try:
            # Calculate Simple Moving Averages
            self.stock_data['SMA_20'] = self.stock_data['Close'].rolling(window=20).mean()
            self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()
            self.stock_data['SMA_200'] = self.stock_data['Close'].rolling(window=200).mean()
            
            # Calculate returns
            self.stock_data['Returns'] = self.stock_data['Close'].pct_change()
            
            # Calculate volatility
            self.stock_data['Volatility'] = self.stock_data['Returns'].rolling(window=20).std()
            
            # Calculate RSI
            self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'])
            
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
        """Main function to answer questions"""
        try:
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
            return f"Error processing question: {e}"
    
    def handle_stock_price_questions(self, question):
        """Handle stock price related questions"""
        try:
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
                period = "entire period"
            
            if len(filtered_data) == 0:
                return f"No data available for the specified period: {period}"
            
            # Calculate statistics
            highest_price = filtered_data['Close'].max()
            lowest_price = filtered_data['Close'].min()
            average_price = filtered_data['Close'].mean()
            
            # Get dates for highest and lowest
            highest_date = filtered_data[filtered_data['Close'] == highest_price].index[0]
            lowest_date = filtered_data[filtered_data['Close'] == lowest_price].index[0]
            
            if 'highest' in question_lower:
                return f"Highest Stock Price for {period}:\n\n" + \
                       f"Price: Rs.{highest_price:.2f}\n" + \
                       f"Date: {highest_date.strftime('%d-%b-%Y')}\n\n" + \
                       f"Additional Context:\n" + \
                       f"• Average price during this period: Rs.{average_price:.2f}\n" + \
                       f"• Lowest price during this period: Rs.{lowest_price:.2f} ({lowest_date.strftime('%d-%b-%Y')})"
            
            elif 'lowest' in question_lower:
                return f"Lowest Stock Price for {period}:\n\n" + \
                       f"Price: Rs.{lowest_price:.2f}\n" + \
                       f"Date: {lowest_date.strftime('%d-%b-%Y')}\n\n" + \
                       f"Additional Context:\n" + \
                       f"• Average price during this period: Rs.{average_price:.2f}\n" + \
                       f"• Highest price during this period: Rs.{highest_price:.2f} ({highest_date.strftime('%d-%b-%Y')})"
            
            else:  # average
                return f"Average Stock Price for {period}:\n\n" + \
                       f"Average Price: Rs.{average_price:.2f}\n\n" + \
                       f"Price Range:\n" + \
                       f"• Highest: Rs.{highest_price:.2f} ({highest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Lowest: Rs.{lowest_price:.2f} ({lowest_date.strftime('%d-%b-%Y')})\n" + \
                       f"• Volatility: {((highest_price - lowest_price) / average_price * 100):.2f}%"
            
        except Exception as e:
            return f"Error analyzing stock price: {e}"
    
    def handle_comparison_questions(self, question):
        """Handle comparison questions"""
        try:
            # Extract time periods from question
            periods = re.findall(r'(\w{3})-(\d{2}|\d{4})', question)
            
            if len(periods) >= 2:
                period1 = periods[0]
                period2 = periods[1]
                
                # Convert to datetime ranges
                date1 = self.parse_period_to_date(period1)
                date2 = self.parse_period_to_date(period2)
                
                # Get data for both periods
                data1 = self.get_data_for_period(date1)
                data2 = self.get_data_for_period(date2)
                
                if data1 is None or data2 is None:
                    return "Unable to find data for the specified periods"
                
                # Calculate metrics for comparison
                metrics1 = self.calculate_period_metrics(data1)
                metrics2 = self.calculate_period_metrics(data2)
                
                return f"Bajaj Finserv Comparison: {period1[0]}-{period1[1]} vs {period2[0]}-{period2[1]}\n\n" + \
                       f"Period 1 ({period1[0]}-{period1[1]}):\n" + \
                       f"• Average Price: Rs.{metrics1['avg_price']:.2f}\n" + \
                       f"• Highest Price: Rs.{metrics1['high_price']:.2f}\n" + \
                       f"• Lowest Price: Rs.{metrics1['low_price']:.2f}\n" + \
                       f"• Volatility: {metrics1['volatility']:.2f}%\n\n" + \
                       f"Period 2 ({period2[0]}-{period2[1]}):\n" + \
                       f"• Average Price: Rs.{metrics2['avg_price']:.2f}\n" + \
                       f"• Highest Price: Rs.{metrics2['high_price']:.2f}\n" + \
                       f"• Lowest Price: Rs.{metrics2['low_price']:.2f}\n" + \
                       f"• Volatility: {metrics2['volatility']:.2f}%\n\n" + \
                       f"Key Insights:\n" + \
                       f"• Price Change: {((metrics2['avg_price'] - metrics1['avg_price']) / metrics1['avg_price'] * 100):+.2f}%\n" + \
                       f"• Performance: {'Better' if metrics2['avg_price'] > metrics1['avg_price'] else 'Lower'} in Period 2"
            
            return "Please specify two time periods to compare (e.g., 'Compare BFS from Jan-24 to Mar-24')"
            
        except Exception as e:
            return f"Error in comparison: {e}"
    
    def handle_organic_traffic_questions(self, question):
        """Handle questions about organic traffic and Bajaj Markets"""
        try:
            relevant_info = self.search_transcripts(['organic traffic', 'bajaj markets', 'digital', 'online', 'customer acquisition'])
            
            if not relevant_info:
                return "No specific information about organic traffic found in recent transcripts"
            
            response = "Organic Traffic & Bajaj Markets Insights:\n\n"
            
            for quarter, info in relevant_info.items():
                response += f"{quarter}:\n"
                response += f"{info[:500]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching for organic traffic information: {e}"
    
    def handle_bagic_questions(self, question):
        """Handle questions about BAGIC and motor insurance headwinds"""
        try:
            relevant_info = self.search_transcripts(['BAGIC', 'motor insurance', 'headwinds', 'challenges', 'general insurance'])
            
            if not relevant_info:
                return "No specific information about BAGIC headwinds found in recent transcripts"
            
            response = "BAGIC Motor Insurance Headwinds:\n\n"
            
            for quarter, info in relevant_info.items():
                response += f"{quarter}:\n"
                response += f"{info[:500]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching for BAGIC information: {e}"
    
    def handle_hero_partnership_questions(self, question):
        """Handle questions about Hero partnership"""
        try:
            relevant_info = self.search_transcripts(['Hero', 'partnership', 'collaboration', 'strategic alliance'])
            
            if not relevant_info:
                return "No specific information about Hero partnership found in recent transcripts"
            
            response = "Hero Partnership Rationale:\n\n"
            
            for quarter, info in relevant_info.items():
                response += f"{quarter}:\n"
                response += f"{info[:500]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching for Hero partnership information: {e}"
    
    def handle_allianz_stake_questions(self, question):
        """Handle questions about Allianz stake sale"""
        try:
            relevant_info = self.search_transcripts(['Allianz', 'stake', 'divestment', 'sale', 'equity'])
            
            if not relevant_info:
                return "No specific information about Allianz stake sale found in recent transcripts"
            
            response = "Allianz Stake Sale Discussions:\n\n"
            
            # Create a table format for dates and discussions
            response += "Quarter | Key Discussion Points\n"
            response += "--------|----------------------\n"
            
            for quarter, info in relevant_info.items():
                # Extract key dates and points
                key_points = info[:200] + "..." if len(info) > 200 else info
                response += f"{quarter} | {key_points}\n"
            
            return response
            
        except Exception as e:
            return f"Error searching for Allianz stake information: {e}"
    
    def handle_cfo_commentary_questions(self, question):
        """Handle CFO commentary requests"""
        try:
            # Get latest financial data
            latest_data = self.get_latest_financial_summary()
            
            response = "CFO Commentary Draft for Upcoming Investor Call:\n\n"
            response += "Opening Remarks:\n"
            response += "Good morning/afternoon, everyone. Thank you for joining us today for our quarterly investor call.\n\n"
            
            response += "Key Financial Highlights:\n"
            response += f"• Our stock has shown resilience with current levels at Rs.{latest_data['current_price']:.2f}\n"
            response += f"• Year-to-date performance: {latest_data['ytd_performance']:+.2f}%\n"
            response += f"• Quarterly volatility remains within manageable ranges at {latest_data['volatility']:.2f}%\n\n"
            
            response += "Strategic Initiatives:\n"
            response += "• Continue to focus on digital transformation across all business verticals\n"
            response += "• Strengthen our market position in the insurance and lending segments\n"
            response += "• Optimize operational efficiency while maintaining growth trajectory\n\n"
            
            response += "Outlook:\n"
            response += "We remain optimistic about our long-term growth prospects and will continue to deliver value to our stakeholders.\n\n"
            
            response += "Q&A Session:\n"
            response += "We will now open the floor for questions from analysts and investors.\n\n"
            
            response += "Note: This is a template commentary. Please customize with actual financial metrics and company-specific updates."
            
            return response
            
        except Exception as e:
            return f"Error generating CFO commentary: {e}"
    
    def handle_general_questions(self, question):
        """Handle general questions"""
        try:
            # Search across all transcripts for relevant information
            relevant_info = self.search_transcripts(question.split())
            
            if not relevant_info:
                return "No specific information found for your question. Please try rephrasing or ask about stock prices, earnings insights, or specific business segments."
            
            response = "Information Found:\n\n"
            
            for quarter, info in relevant_info.items():
                response += f"{quarter}:\n"
                response += f"{info[:400]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"Error processing general question: {e}"
    
    def search_transcripts(self, keywords):
        """Search for keywords in transcripts"""
        try:
            results = {}
            
            for quarter, transcript in self.transcripts.items():
                transcript_lower = transcript.lower()
                
                for keyword in keywords:
                    if keyword.lower() in transcript_lower:
                        # Extract relevant context around the keyword
                        context = self.extract_context(transcript, keyword)
                        if context:
                            results[quarter] = context
                            break
            
            return results
            
        except Exception as e:
            print(f"Error searching transcripts: {e}")
            return {}
    
    def extract_context(self, text, keyword, context_size=500):
        """Extract context around a keyword"""
        try:
            text_lower = text.lower()
            keyword_lower = keyword.lower()
            
            index = text_lower.find(keyword_lower)
            if index == -1:
                return None
            
            start = max(0, index - context_size)
            end = min(len(text), index + context_size)
            
            return text[start:end]
            
        except Exception as e:
            return None
    
    def parse_period_to_date(self, period):
        """Parse period tuple to date"""
        try:
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            month = month_map.get(period[0].lower(), 1)
            year = int(period[1]) if len(period[1]) == 4 else 2000 + int(period[1])
            
            return datetime(year, month, 1)
            
        except Exception as e:
            return None
    
    def get_data_for_period(self, date):
        """Get data for a specific period"""
        try:
            if date is None:
                return None
            
            # Get data for the month
            filtered_data = self.stock_data[
                (self.stock_data.index.month == date.month) & 
                (self.stock_data.index.year == date.year)
            ]
            
            return filtered_data if len(filtered_data) > 0 else None
            
        except Exception as e:
            return None
    
    def calculate_period_metrics(self, data):
        """Calculate metrics for a period"""
        try:
            return {
                'avg_price': data['Close'].mean(),
                'high_price': data['Close'].max(),
                'low_price': data['Close'].min(),
                'volatility': data['Close'].std() / data['Close'].mean() * 100
            }
        except Exception as e:
            return {}
    
    def get_latest_financial_summary(self):
        """Get latest financial summary"""
        try:
            if self.stock_data is None or len(self.stock_data) == 0:
                return {'current_price': 0, 'ytd_performance': 0, 'volatility': 0}
            
            current_price = self.stock_data['Close'].iloc[-1]
            ytd_start = self.stock_data[self.stock_data.index.year == 2024]['Close'].iloc[0]
            ytd_performance = (current_price - ytd_start) / ytd_start * 100
            volatility = self.stock_data['Close'].tail(90).std() / self.stock_data['Close'].tail(90).mean() * 100
            
            return {
                'current_price': current_price,
                'ytd_performance': ytd_performance,
                'volatility': volatility
            }
            
        except Exception as e:
            return {'current_price': 0, 'ytd_performance': 0, 'volatility': 0}
    
    def create_stock_chart(self):
        """Create an interactive stock chart"""
        try:
            if self.stock_data is None:
                return None
            
            fig = go.Figure()
            
            # Add stock price
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Close'],
                mode='lines',
                name='BFS Stock Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='green', width=1)
            ))
            
            fig.update_layout(
                title='Bajaj Finserv Stock Price Analysis',
                xaxis_title='Date',
                yaxis_title='Price (Rs.)',
                template='plotly_white',
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None

# Initialize the chatbot
chatbot = BajajFinservChatbot()

# Gradio interface
def chat_interface(message, history):
    """Chat interface function"""
    try:
        response = chatbot.answer_question(message)
        return response
    except Exception as e:
        return f"Error: {e}"

def create_chart_interface():
    """Create chart interface"""
    try:
        return chatbot.create_stock_chart()
    except Exception as e:
        return None

# Create Gradio app
with gr.Blocks(title="Bajaj Finserv AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Bajaj Finserv AI Chatbot
    
    Welcome to the comprehensive Bajaj Finserv chatbot! I can help you with:
    
    ## What I Can Do:
    - Stock Price Analysis: Highest/lowest/average prices across different periods
    - Performance Comparison: Compare BFS performance between different time periods
    - Organic Traffic Insights: Information about Bajaj Markets digital strategy
    - BAGIC Analysis: Motor insurance business headwinds and challenges
    - Hero Partnership: Strategic partnership rationale and details
    - Allianz Stake Sale: Discussions and timelines about stake divestment
    - CFO Commentary: Draft investor call commentary as a CFO
    
    ## Example Questions:
    - "What was the highest stock price in Jan-24?"
    - "Compare Bajaj Finserv from Mar-23 to Jun-23"
    - "Tell me about organic traffic of Bajaj Markets"
    - "Why is BAGIC facing headwinds in Motor insurance?"
    - "What's the rationale of Hero partnership?"
    - "Give me table with dates explaining discussions regarding Allianz stake sale"
    - "Act as a CFO of BAGIC and help me draft commentary for upcoming investor call"
    """)
    
    with gr.Tab("Chat with AI"):
        chatbot_interface = gr.ChatInterface(
            fn=chat_interface,
            title="Ask me anything about Bajaj Finserv!",
            description="I have access to stock price data and earnings call transcripts from the last 4 quarters.",
            examples=[
                "What was the highest stock price in 2024?",
                "Compare BFS performance from Jan-24 to Mar-24",
                "Tell me about organic traffic of Bajaj Markets",
                "Why is BAGIC facing headwinds in Motor insurance?",
                "What's the Hero partnership rationale?",
                "Show me Allianz stake sale discussions",
                "Act as CFO and draft investor call commentary"
            ],
            theme="soft"
        )
    
    with gr.Tab("Stock Chart"):
        gr.Markdown("### Interactive Stock Price Chart")
        chart_button = gr.Button("Generate Chart", variant="primary")
        chart_output = gr.Plot()
        
        chart_button.click(
            create_chart_interface,
            outputs=chart_output
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Chatbot
        
        This AI-powered chatbot is specifically designed for Bajaj Finserv analysis and can:
        
        ### Data Sources:
        - Stock Price Data: Historical BFS stock prices with technical indicators
        - Earnings Transcripts: Q1-Q4 FY25 quarterly earnings call transcripts
        - Knowledge Base: Key information about BFS subsidiaries and strategic initiatives
        
        ### AI Capabilities:
        - Natural language understanding for complex financial queries
        - Context-aware responses based on earnings call transcripts
        - Technical analysis with moving averages and volatility metrics
        - Comparative analysis across different time periods
        
        ### Use Cases:
        - Investment Analysis: Stock price trends and performance metrics
        - Strategic Insights: Business segment analysis and market positioning
        - Regulatory Updates: Information from quarterly earnings calls
        - Executive Communication: CFO commentary and investor relations
        
        ### Important Notes:
        - This tool is for informational purposes only
        - Not intended as financial advice
        - Based on historical data and public earnings calls
        - Always consult with financial professionals for investment decisions
        """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)