#!/usr/bin/env python3
"""
Simple test script for Bajaj Finserv Chatbot
This script tests basic functionality and helps identify missing dependencies
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas - OK")
    except ImportError as e:
        print("❌ pandas - MISSING")
        print(f"   Install with: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✅ numpy - OK")
    except ImportError as e:
        print("❌ numpy - MISSING")
        print(f"   Install with: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib - OK")
    except ImportError as e:
        print("❌ matplotlib - MISSING")
        print(f"   Install with: pip install matplotlib")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly - OK")
    except ImportError as e:
        print("❌ plotly - MISSING")
        print(f"   Install with: pip install plotly")
        return False
    
    try:
        import gradio as gr
        print("✅ gradio - OK")
    except ImportError as e:
        print("❌ gradio - MISSING")
        print(f"   Install with: pip install gradio")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 - OK")
    except ImportError as e:
        print("❌ PyPDF2 - MISSING")
        print(f"   Install with: pip install PyPDF2")
        return False
    
    return True

def test_data_files():
    """Test if required data files exist"""
    print("\n📂 Testing data files...")
    
    import os
    
    files = [
        'BFS_Share_Price.csv',
        'Earnings Call Transcript Q1 - FY25  .pdf',
        'Earnings Call Transcript Q2 - FY25.pdf',
        'Earnings Call Transcript Q3 - FY25.pdf',
        'Earnings Call Transcript Q4 - FY25.pdf'
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """Test basic chatbot functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test pandas data loading
        import pandas as pd
        
        if not os.path.exists('BFS_Share_Price.csv'):
            print("❌ Cannot test - BFS_Share_Price.csv missing")
            return False
        
        df = pd.read_csv('BFS_Share_Price.csv')
        print(f"✅ Stock data loaded - {len(df)} records")
        
        # Test date parsing
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
        print("✅ Date parsing - OK")
        
        # Test basic analysis
        if 'Close Price' in df.columns:
            max_price = df['Close Price'].max()
            min_price = df['Close Price'].min()
            print(f"✅ Price analysis - Max: ₹{max_price:.2f}, Min: ₹{min_price:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🏦 Bajaj Finserv Chatbot - System Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test basic functionality if imports are OK
    if imports_ok:
        func_ok = test_basic_functionality()
    else:
        func_ok = False
        print("\n❌ Skipping functionality test - missing dependencies")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if imports_ok and data_ok and func_ok:
        print("🎉 ALL TESTS PASSED!")
        print("Your system is ready to run the Bajaj Finserv Chatbot!")
        print("\nRun the chatbot with:")
        print("python bajaj_finserv_chatbot.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        if not imports_ok:
            print("❌ Missing dependencies - install required packages")
        if not data_ok:
            print("❌ Missing data files - check your data files")
        if not func_ok:
            print("❌ Basic functionality failed - check data format")
        
        print("\n🔧 QUICK FIX:")
        print("1. Install all dependencies:")
        print("   pip install pandas numpy matplotlib plotly gradio PyPDF2")
        print("2. Make sure all data files are in the same folder")
        print("3. Run this test again: python simple_test.py")

if __name__ == "__main__":
    import os
    main()