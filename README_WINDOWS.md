# 🏦 Bajaj Finserv AI Chatbot - Windows Installation Guide

## 🚨 **FIXING YOUR ERROR**

You're getting this error:
```
ModuleNotFoundError: No module named 'plotly'
```

This means the required Python packages aren't installed on your Windows machine.

## 🚀 **QUICK SOLUTION (Choose One Method)**

### **Method 1: One-Command Installation**
Open Command Prompt as Administrator and run:
```bash
pip install pandas numpy matplotlib plotly gradio PyPDF2
```

### **Method 2: Using Requirements File**
```bash
# Navigate to your project folder
cd C:\Users\Abhi\Downloads\hackthon\HCK

# Install from requirements file
pip install -r requirements_local.txt
```

### **Method 3: Test Your System First**
```bash
# Test what's missing
python simple_test.py

# This will tell you exactly what to install
```

## 📋 **Step-by-Step Installation**

### **1. Open Command Prompt**
- Press `Win + X` and select "Command Prompt (Admin)"
- Or press `Win + R`, type `cmd`, press `Ctrl + Shift + Enter`

### **2. Check Python Version**
```bash
python --version
```
You need Python 3.7 or higher.

### **3. Update pip**
```bash
python -m pip install --upgrade pip
```

### **4. Install Required Packages**
```bash
pip install pandas numpy matplotlib plotly gradio PyPDF2
```

### **5. Test Installation**
```bash
python -c "import plotly; print('Success! All packages installed.')"
```

### **6. Run the Chatbot**
```bash
python bajaj_finserv_chatbot.py
```

## 🔧 **If You Still Get Errors**

### **Error: 'pip' is not recognized**
Try these alternatives:
```bash
python -m pip install pandas numpy matplotlib plotly gradio PyPDF2
```
or
```bash
py -m pip install pandas numpy matplotlib plotly gradio PyPDF2
```

### **Error: Permission denied**
Install with user flag:
```bash
pip install --user pandas numpy matplotlib plotly gradio PyPDF2
```

### **Error: Microsoft Visual C++ required**
Some packages need Visual Studio Build Tools:
1. Download Visual Studio Build Tools from Microsoft
2. Or install Anaconda/Miniconda instead of regular Python

## 🐍 **Alternative: Using Anaconda (Recommended)**

If you have issues with pip, use Anaconda:

### **Install Anaconda**
1. Download from https://www.anaconda.com/products/distribution
2. Install with default settings

### **Create Environment**
```bash
# Open Anaconda Prompt
conda create -n bajaj python=3.9
conda activate bajaj
conda install pandas numpy matplotlib plotly
pip install gradio PyPDF2
```

### **Run Chatbot**
```bash
cd C:\Users\Abhi\Downloads\hackthon\HCK
python bajaj_finserv_chatbot.py
```

## ✅ **Verify Everything Works**

### **Quick Test**
```bash
python simple_test.py
```

### **Expected Output**
```
🏦 Bajaj Finserv Chatbot - System Test
==================================================
🔍 Testing package imports...
✅ pandas - OK
✅ numpy - OK
✅ matplotlib - OK
✅ plotly - OK
✅ gradio - OK
✅ PyPDF2 - OK

📂 Testing data files...
✅ BFS_Share_Price.csv - Found
✅ Earnings Call Transcript Q1 - FY25  .pdf - Found
[...]

🎉 ALL TESTS PASSED!
```

## 🌐 **Running the Chatbot**

Once everything is installed:

```bash
python bajaj_finserv_chatbot.py
```

You should see:
```
All data loaded successfully!
Stock data loaded successfully!
Running on local URL: http://127.0.0.1:7860
```

Then open your browser and go to: **http://127.0.0.1:7860**

## 🆘 **Still Having Issues?**

### **Check Your Setup**
1. **Python Version**: Must be 3.7+
   ```bash
   python --version
   ```

2. **Pip Version**: Update to latest
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install Location**: Make sure you're in the right folder
   ```bash
   dir
   # Should show: bajaj_finserv_chatbot.py and BFS_Share_Price.csv
   ```

### **Common Solutions**
- **Use `python3` instead of `python`** if you have multiple Python versions
- **Run Command Prompt as Administrator** for permission issues
- **Restart your command prompt** after installing packages
- **Check firewall settings** if the web interface doesn't load

## 📞 **Success Indicators**

You'll know it's working when:
1. ✅ No import errors when running the script
2. ✅ See "All data loaded successfully!" message
3. ✅ Browser opens to http://127.0.0.1:7860
4. ✅ You can ask questions like "What was the highest stock price in Jan-24?"

## 🎯 **Sample Questions to Test**

Once running, try these questions:
- "What was the highest stock price in 2024?"
- "Compare BFS from Jan-24 to Mar-24"
- "Tell me about BAGIC motor insurance headwinds"
- "Show me CFO commentary for investor call"

Your chatbot should now be working perfectly! 🎉