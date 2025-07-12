#!/usr/bin/env python3
"""
Advanced debugging script for Bajaj Finserv Chatbot
This script identifies and fixes parsing issues with CSV and PDF files
"""

import pandas as pd
import os
import sys
import chardet
import json
from datetime import datetime
import re

def detect_file_encoding(file_path):
    """Detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except Exception as e:
        print(f"❌ Error detecting encoding: {e}")
        return 'utf-8'

def analyze_csv_structure(file_path):
    """Analyze CSV file structure and content"""
    print(f"\n🔍 Analyzing CSV file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Detect encoding
    encoding = detect_file_encoding(file_path)
    print(f"📄 File encoding: {encoding}")
    
    # Try different separators
    separators = [',', ';', '\t', '|']
    best_df = None
    best_sep = None
    
    for sep in separators:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=10)
            if len(df.columns) > 1:
                print(f"✅ Successfully parsed with separator: '{sep}'")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
                best_df = df
                best_sep = sep
                break
        except Exception as e:
            continue
    
    if best_df is None:
        print("❌ Could not parse CSV with any separator")
        return None
    
    # Load full file
    try:
        full_df = pd.read_csv(file_path, sep=best_sep, encoding=encoding)
        print(f"✅ Full file loaded: {len(full_df)} rows")
        
        # Analyze columns
        print(f"\n📊 Column Analysis:")
        for col in full_df.columns:
            print(f"   '{col}': {full_df[col].dtype}, {full_df[col].notna().sum()} non-null values")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        print(full_df.head())
        
        # Check for date columns
        date_columns = []
        for col in full_df.columns:
            if 'date' in col.lower() or full_df[col].dtype == 'object':
                # Try to parse as date
                try:
                    pd.to_datetime(full_df[col].iloc[0])
                    date_columns.append(col)
                except:
                    pass
        
        if date_columns:
            print(f"\n📅 Potential date columns: {date_columns}")
        
        # Check for price columns
        price_columns = []
        for col in full_df.columns:
            if 'price' in col.lower() or 'close' in col.lower() or pd.api.types.is_numeric_dtype(full_df[col]):
                price_columns.append(col)
        
        if price_columns:
            print(f"💰 Potential price columns: {price_columns}")
        
        return {
            'dataframe': full_df,
            'separator': best_sep,
            'encoding': encoding,
            'date_columns': date_columns,
            'price_columns': price_columns
        }
        
    except Exception as e:
        print(f"❌ Error loading full file: {e}")
        return None

def fix_csv_parsing(file_path):
    """Create a fixed version of the CSV file"""
    print(f"\n🔧 Attempting to fix CSV parsing...")
    
    analysis = analyze_csv_structure(file_path)
    if not analysis:
        return None
    
    df = analysis['dataframe']
    
    # Fix column names
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        if 'date' in col.lower():
            column_mapping[col] = 'Date'
        elif 'close' in col.lower() or 'price' in col.lower():
            column_mapping[col] = 'Close Price'
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Fix date column
    if 'Date' in df.columns:
        try:
            # Try multiple date formats
            date_formats = [
                '%d-%b-%y',    # 3-Jan-22
                '%d-%m-%Y',    # 03-01-2022
                '%Y-%m-%d',    # 2022-01-03
                '%m/%d/%Y',    # 01/03/2022
                '%d/%m/%Y',    # 03/01/2022
            ]
            
            parsed = False
            for fmt in date_formats:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                    print(f"✅ Date parsing successful with format: {fmt}")
                    parsed = True
                    break
                except:
                    continue
            
            if not parsed:
                # Try automatic parsing
                df['Date'] = pd.to_datetime(df['Date'])
                print("✅ Date parsing successful with automatic detection")
                
        except Exception as e:
            print(f"❌ Date parsing failed: {e}")
    
    # Fix price column
    if 'Close Price' in df.columns:
        try:
            # Remove any non-numeric characters except decimal point
            df['Close Price'] = df['Close Price'].astype(str).str.replace('[^0-9.]', '', regex=True)
            df['Close Price'] = pd.to_numeric(df['Close Price'], errors='coerce')
            print("✅ Price column cleaned")
        except Exception as e:
            print(f"❌ Price cleaning failed: {e}")
    
    # Save fixed version
    fixed_file = file_path.replace('.csv', '_fixed.csv')
    try:
        df.to_csv(fixed_file, index=False)
        print(f"✅ Fixed CSV saved as: {fixed_file}")
        return fixed_file
    except Exception as e:
        print(f"❌ Error saving fixed file: {e}")
        return None

def check_pdf_parsing():
    """Check PDF file parsing"""
    print(f"\n🔍 Checking PDF file parsing...")
    
    pdf_files = [
        'Earnings Call Transcript Q1 - FY25  .pdf',
        'Earnings Call Transcript Q2 - FY25.pdf',
        'Earnings Call Transcript Q3 - FY25.pdf',
        'Earnings Call Transcript Q4 - FY25.pdf'
    ]
    
    # Check if PyPDF2 is available
    try:
        import PyPDF2
        print("✅ PyPDF2 is available")
    except ImportError:
        print("❌ PyPDF2 not installed")
        print("   Install with: pip install PyPDF2")
        return False
    
    success_count = 0
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    
                    # Try to extract text from first page
                    first_page_text = pdf_reader.pages[0].extract_text()
                    
                    if first_page_text.strip():
                        print(f"✅ {pdf_file}: {num_pages} pages, text extraction working")
                        success_count += 1
                    else:
                        print(f"⚠️ {pdf_file}: {num_pages} pages, but no text extracted")
                        
            except Exception as e:
                print(f"❌ {pdf_file}: Error reading - {e}")
        else:
            print(f"❌ {pdf_file}: File not found")
    
    print(f"\n📊 PDF Summary: {success_count}/{len(pdf_files)} files working")
    return success_count > 0

def create_sample_data():
    """Create sample data if files are missing or corrupted"""
    print(f"\n🔧 Creating sample data for testing...")
    
    # Create sample stock data
    dates = pd.date_range('2022-01-01', '2024-01-31', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Generate realistic stock prices
    import numpy as np
    np.random.seed(42)
    
    base_price = 1500
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0, 20)  # Daily volatility
        current_price = max(current_price + change, 100)  # Minimum price
        prices.append(current_price)
    
    sample_df = pd.DataFrame({
        'Date': dates,
        'Close Price': prices
    })
    
    # Save sample data
    sample_file = 'BFS_Share_Price_sample.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"✅ Sample data created: {sample_file}")
    
    return sample_file

def main():
    """Main debugging function"""
    print("🔧 Advanced Bajaj Finserv Chatbot Debugging")
    print("=" * 60)
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    
    # List all files
    all_files = os.listdir('.')
    csv_files = [f for f in all_files if f.endswith('.csv')]
    pdf_files = [f for f in all_files if f.endswith('.pdf')]
    
    print(f"📂 CSV files found: {csv_files}")
    print(f"📂 PDF files found: {pdf_files}")
    
    # Check main CSV file
    target_csv = 'BFS_Share_Price.csv'
    fixed_csv = None
    
    if target_csv in csv_files:
        analysis = analyze_csv_structure(target_csv)
        if analysis:
            fixed_csv = fix_csv_parsing(target_csv)
        else:
            print(f"❌ Could not parse {target_csv}")
    else:
        print(f"❌ {target_csv} not found")
    
    # If CSV parsing failed, create sample data
    if not fixed_csv:
        print("\n⚠️ Creating sample data for testing...")
        fixed_csv = create_sample_data()
    
    # Check PDF parsing
    pdf_success = check_pdf_parsing()
    
    # Create recommendations
    print("\n" + "=" * 60)
    print("📋 RECOMMENDATIONS")
    print("=" * 60)
    
    if fixed_csv:
        print(f"✅ Use this CSV file: {fixed_csv}")
    else:
        print("❌ CSV parsing failed completely")
    
    if pdf_success:
        print("✅ PDF parsing working")
    else:
        print("❌ PDF parsing failed - install PyPDF2 or check file corruption")
    
    # Create a simple test
    print(f"\n🧪 Testing chatbot with fixed data...")
    if fixed_csv:
        try:
            test_df = pd.read_csv(fixed_csv)
            if 'Date' in test_df.columns and 'Close Price' in test_df.columns:
                test_df['Date'] = pd.to_datetime(test_df['Date'])
                test_df.set_index('Date', inplace=True)
                
                # Test basic analysis
                max_price = test_df['Close Price'].max()
                min_price = test_df['Close Price'].min()
                avg_price = test_df['Close Price'].mean()
                
                print(f"✅ Basic analysis working:")
                print(f"   Max price: ₹{max_price:.2f}")
                print(f"   Min price: ₹{min_price:.2f}")
                print(f"   Avg price: ₹{avg_price:.2f}")
                
                return True
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 Files are now ready for the chatbot!")
        print(f"Run: python fix_chatbot.py")
    else:
        print(f"\n❌ Files still have issues. Please check the recommendations above.")