#!/usr/bin/env python3
"""
Debug script to check data loading issues for Bajaj Finserv Chatbot
"""

import pandas as pd
import os

def check_csv_file():
    """Check if CSV file exists and can be loaded"""
    print("🔍 Checking BFS_Share_Price.csv...")
    
    # Check if file exists
    if not os.path.exists('BFS_Share_Price.csv'):
        print("❌ BFS_Share_Price.csv not found!")
        print("   Make sure the file is in the same folder as your Python script")
        return False
    
    print("✅ File exists")
    
    # Try to load the file
    try:
        df = pd.read_csv('BFS_Share_Price.csv')
        print(f"✅ File loaded successfully - {len(df)} rows")
        
        # Check columns
        print(f"\n📊 Columns found: {list(df.columns)}")
        
        # Check for required columns
        if 'Date' in df.columns:
            print("✅ 'Date' column found")
        else:
            print("❌ 'Date' column missing")
        
        if 'Close Price' in df.columns:
            print("✅ 'Close Price' column found")
        else:
            print("❌ 'Close Price' column missing")
            # Check for alternative column names
            possible_columns = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
            if possible_columns:
                print(f"   Found similar columns: {possible_columns}")
        
        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        # Try date parsing
        print(f"\n📅 Testing date parsing...")
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
            print("✅ Date parsing successful")
        except Exception as e:
            print(f"❌ Date parsing failed: {e}")
            print("   Trying alternative date formats...")
            
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    pd.to_datetime(df['Date'].iloc[0], format=fmt)
                    print(f"   ✅ Found working format: {fmt}")
                    break
                except:
                    continue
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

def check_pdf_files():
    """Check if PDF files exist"""
    print("\n🔍 Checking PDF files...")
    
    pdf_files = [
        'Earnings Call Transcript Q1 - FY25  .pdf',
        'Earnings Call Transcript Q2 - FY25.pdf',
        'Earnings Call Transcript Q3 - FY25.pdf',
        'Earnings Call Transcript Q4 - FY25.pdf'
    ]
    
    found_files = 0
    for file in pdf_files:
        if os.path.exists(file):
            print(f"✅ {file}")
            found_files += 1
        else:
            print(f"❌ {file}")
    
    if found_files == 0:
        print("⚠️  No PDF files found - transcript analysis won't work")
    else:
        print(f"✅ Found {found_files}/{len(pdf_files)} PDF files")

def test_chatbot_functionality():
    """Test basic chatbot functionality"""
    print("\n🧪 Testing chatbot functionality...")
    
    try:
        from bajaj_finserv_chatbot import BajajFinservChatbot
        print("✅ Chatbot import successful")
        
        chatbot = BajajFinservChatbot()
        print("✅ Chatbot initialization successful")
        
        # Test a simple question
        response = chatbot.answer_question("What is the average stock price in 2023?")
        print(f"✅ Question processing successful")
        print(f"Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        return False

def main():
    """Main debug function"""
    print("🔧 Bajaj Finserv Chatbot - Data Debug")
    print("=" * 50)
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📂 Files in directory: {os.listdir('.')}")
    
    # Check CSV file
    csv_ok = check_csv_file()
    
    # Check PDF files
    check_pdf_files()
    
    # Test chatbot if CSV is OK
    if csv_ok:
        chatbot_ok = test_chatbot_functionality()
    else:
        chatbot_ok = False
        print("\n❌ Skipping chatbot test - CSV file issues")
    
    print("\n" + "=" * 50)
    print("📊 DEBUG SUMMARY")
    print("=" * 50)
    
    if csv_ok and chatbot_ok:
        print("🎉 Everything looks good!")
        print("The chatbot should work properly.")
    else:
        print("⚠️  Issues found:")
        if not csv_ok:
            print("❌ CSV file has problems")
        if not chatbot_ok:
            print("❌ Chatbot functionality failed")
        
        print("\n🔧 SUGGESTED FIXES:")
        print("1. Make sure BFS_Share_Price.csv is in the same folder")
        print("2. Check that the CSV has 'Date' and 'Close Price' columns")
        print("3. Verify the date format in the CSV file")

if __name__ == "__main__":
    main()