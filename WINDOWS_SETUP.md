# 🪟 Windows Setup Instructions for Bajaj Finserv Chatbot

## 🚨 **Quick Fix for Your Error**

You're getting the `ModuleNotFoundError: No module named 'plotly'` because the required packages aren't installed on your local Windows machine.

## 📋 **Step-by-Step Installation**

### **Step 1: Open Command Prompt as Administrator**
- Press `Win + R`, type `cmd`, and press `Ctrl + Shift + Enter`
- Or search for "Command Prompt" → Right-click → "Run as administrator"

### **Step 2: Install Required Packages**

Copy and paste these commands one by one in your Command Prompt:

```bash
pip install pandas
```

```bash
pip install numpy
```

```bash
pip install matplotlib
```

```bash
pip install plotly
```

```bash
pip install gradio
```

```bash
pip install PyPDF2
```

### **Step 3: Install All at Once (Alternative)**

Or run this single command to install everything:

```bash
pip install pandas numpy matplotlib plotly gradio PyPDF2
```

### **Step 4: If pip doesn't work, try pip3**

If you get a "'pip' is not recognized" error, try:

```bash
pip3 install pandas numpy matplotlib plotly gradio PyPDF2
```

### **Step 5: If still having issues, use python -m pip**

```bash
python -m pip install pandas numpy matplotlib plotly gradio PyPDF2
```

## ✅ **Test the Installation**

After installing, test if everything works:

```bash
python -c "import plotly; print('Plotly installed successfully!')"
```

## 🚀 **Run the Chatbot**

Once all packages are installed, navigate to your project folder and run:

```bash
cd C:\Users\Abhi\Downloads\hackthon\HCK
python bajaj_finserv_chatbot.py
```

## 🔧 **Alternative: Using Virtual Environment (Recommended)**

For a cleaner setup, create a virtual environment:

```bash
# Navigate to your project folder
cd C:\Users\Abhi\Downloads\hackthon\HCK

# Create virtual environment
python -m venv bajaj_env

# Activate it
bajaj_env\Scripts\activate

# Install packages
pip install pandas numpy matplotlib plotly gradio PyPDF2

# Run the chatbot
python bajaj_finserv_chatbot.py
```

## 🆘 **If You Still Get Errors**

1. **Check Python Version**: Make sure you have Python 3.7+ installed
   ```bash
   python --version
   ```

2. **Update pip**: 
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install with --user flag**:
   ```bash
   pip install --user pandas numpy matplotlib plotly gradio PyPDF2
   ```

## 📞 **Quick Success Check**

After installation, you should see output like:
```
All data loaded successfully!
Stock data loaded successfully!
Running on local URL:  http://127.0.0.1:7860
```

Then open your browser and go to `http://127.0.0.1:7860` to use the chatbot! 🎉