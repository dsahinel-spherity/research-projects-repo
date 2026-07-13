# Steel DPP Demo Website

This Streamlit application uses synthetic Steel Digital Product Passport data as its backend.

It demonstrates a scenario simulation for predicting how changes in steel production conditions may affect the estimated carbon footprint.

## Features

* Scenario simulator
* Mock data explorer
* Model performance overview
* Synthetic Steel DPP dataset
* Carbon-footprint prediction using a Gradient Boosting Regressor

## Requirements

Before running the application, install:

* Python 3.10 or newer
* An internet connection for installing the required Python packages

You can check whether Python is installed by opening a terminal and running:

```bash
python --version
```

On Windows, use:

```powershell
py --version
```

---

# Run on Windows

## 1. Install Python

Download and install Python.

During installation, make sure to select:

```text
Add Python to PATH
```

## 2. Open the Project Folder

Open PowerShell or Command Prompt.

If the project folder is in Downloads, use one of the following commands.

### PowerShell

```powershell
cd "$HOME\Downloads\steel_dpp_demo_website"
```

### Command Prompt

```cmd
cd %USERPROFILE%\Downloads\steel_dpp_demo_website
```

Replace the path if the project is stored somewhere else.

## 3. Create a Virtual Environment

```powershell
py -m venv .venv
```

## 4. Activate the Virtual Environment

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
.venv\Scripts\activate
```

## 5. Install the Required Packages

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## 6. Start the Application

```powershell
py -m streamlit run app.py
```

The application should open automatically in your browser.

You can also open it manually at:

```text
http://localhost:8501
```

---

# Run on macOS

## 1. Install Python

Check whether Python is installed:

```bash
python3 --version
```

If Python is not installed, install it from the official Python website or by using Homebrew:

```bash
brew install python
```

## 2. Open the Project Folder

Open Terminal.

If the project folder is in Downloads, run:

```bash
cd ~/Downloads/steel_dpp_demo_website
```

Replace the path if the project is stored somewhere else.

## 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

## 4. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

## 5. Install the Required Packages

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 6. Start the Application

```bash
python3 -m streamlit run app.py
```

The application should open automatically in your browser.

You can also open it manually at:

```text
http://localhost:8501
```

---

# Main Pages

## Scenario Simulator

Change production-related values such as:

* Energy consumption
* Recycled content
* Production volume
* Technology route
* Steel composition
* Production conditions

The application uses a Gradient Boosting Regressor to estimate the resulting carbon footprint.

## Mock Data Explorer

View, filter, and inspect the synthetic Steel Digital Product Passport dataset used by the application.

## Model Performance

Review model evaluation results and compare predicted carbon-footprint values with the synthetic reference values.

---

# Stop the Application

Return to the terminal and press:

```text
Ctrl + C
```

---

# Common Windows Problems

## Python Is Not Recognized

Try:

```powershell
py --version
```

If this fails, reinstall Python and make sure to select:

```text
Add Python to PATH
```

## Streamlit Is Not Recognized

Instead of:

```powershell
streamlit run app.py
```

use:

```powershell
py -m streamlit run app.py
```

## The Requirements File Cannot Be Found

Make sure the terminal is inside the project folder:

```powershell
dir
```

You should see files such as:

```text
app.py
requirements.txt
README.md
```

## PowerShell Blocks Virtual Environment Activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Common macOS Problems

## Python Is Not Recognized

Try:

```bash
python3 --version
```

Use `python3` instead of `python` in the setup commands.

## Streamlit Is Not Recognized

Instead of:

```bash
streamlit run app.py
```

use:

```bash
python3 -m streamlit run app.py
```

## Permission Error During Package Installation

Make sure the virtual environment is activated:

```bash
source .venv/bin/activate
```

Then install the packages again:

```bash
python3 -m pip install -r requirements.txt
```

---

# Port Already in Use

If port `8501` is already in use, start the application on another port.

## Windows

```powershell
py -m streamlit run app.py --server.port 8502
```

## macOS

```bash
python3 -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# Important Notice

The dataset used by this application is completely synthetic.

The application uses only a Gradient Boosting Regressor for prediction.

Because the model is trained on synthetic data, it may contain strong bias and may produce unrealistic or inaccurate predictions.

The model and its results are intended only for scenario simulation and demonstration purposes.

They should not be used for real steel production, environmental reporting, regulatory compliance, engineering, or investment decisions.
