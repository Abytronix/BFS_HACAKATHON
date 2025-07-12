#!/usr/bin/env python3
"""
Test script for Bajaj Finserv Chatbot
Demonstrates various capabilities with sample queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bajaj_finserv_chatbot import BajajFinservChatbot

def test_chatbot():
    """Test the chatbot with various sample queries"""
    print("🏦 Bajaj Finserv AI Chatbot - Test Demo")
    print("=" * 50)
    
    # Initialize chatbot
    print("\n🔧 Initializing chatbot...")
    chatbot = BajajFinservChatbot()
    
    # Test queries
    test_queries = [
        "What was the highest stock price in Jan-24?",
        "Show me the lowest price in 2023",
        "What's the average price for Mar-22?",
        "Compare Bajaj Finserv from Jan-24 to Mar-23",
        "Tell me about organic traffic of Bajaj Markets",
        "Why is BAGIC facing headwinds in Motor insurance business?",
        "What's the rationale of Hero partnership?",
        "Give me table with dates explaining discussions regarding Allianz stake sale",
        "Act as a CFO of BAGIC and help me draft commentary for upcoming investor call"
    ]
    
    print(f"\n📋 Running {len(test_queries)} test queries...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 60)
        
        try:
            response = chatbot.answer_question(query)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)
        
        # Add a separator between tests
        if i < len(test_queries):
            print("\n" + "=" * 50)

if __name__ == "__main__":
    test_chatbot()