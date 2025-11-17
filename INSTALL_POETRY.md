# How to Install and Use Poetry on Windows

## Installation Methods

### Method 1: Using pip (Recommended if you have Python installed)

```powershell
# First, ensure Python is installed and in PATH
python --version

# Install Poetry using pip
pip install poetry

# Verify installation
poetry --version
```

### Method 2: Official Installer Script (Recommended)

```powershell
# Download and run the official installer
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Add Poetry to PATH (restart terminal after this)
# The installer will show you the path to add, typically:
# C:\Users\<YourUsername>\AppData\Roaming\Python\Scripts
```

### Method 3: Using pipx (if you have pipx installed)

```powershell
pipx install poetry
```

## Adding Poetry to PATH

After installation, you may need to add Poetry to your PATH:

1. Find where Poetry was installed (usually `%APPDATA%\Python\Scripts` or `%LOCALAPPDATA%\Programs\Python\Scripts`)
2. Add it to your system PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" variable
   - Add the Poetry Scripts directory
3. Restart your terminal/PowerShell

## Common Poetry Commands

### Project Setup

```powershell
# Install all dependencies (creates virtual environment if needed)
poetry install

# Install dependencies without dev dependencies
poetry install --no-dev

# Update dependencies
poetry update

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Remove a dependency
poetry remove package-name
```

### Running Commands

```powershell
# Run a command in the Poetry environment
poetry run python script.py

# Run the CLI application
poetry run marshab --help

# Run the web server
poetry run python -m marshab.web.server

# Activate the virtual environment (optional - you can use poetry run instead)
poetry shell
```

### Managing Virtual Environment

```powershell
# Show virtual environment path
poetry env info

# List all virtual environments
poetry env list

# Remove virtual environment
poetry env remove python

# Create new virtual environment
poetry env use python
```

### Lock File Management

```powershell
# Update lock file after changing pyproject.toml
poetry lock

# Update lock file and dependencies
poetry lock --no-update
```

## For This Project Specifically

After installing Poetry, run these commands:

```powershell
# Navigate to project directory
cd C:\Users\tanne\Gauntlet\MarsGIS

# Install all dependencies (including new scikit-image)
poetry install

# Run the backend server
poetry run python -m marshab.web.server

# Or use the shortcut script
.\start_web_server.ps1
```

## Troubleshooting

### Poetry not found after installation

1. Check if Poetry is installed:
   ```powershell
   Get-Command poetry -ErrorAction SilentlyContinue
   ```

2. If not found, manually add to PATH:
   ```powershell
   # Find Poetry installation
   $poetryPath = "$env:APPDATA\Python\Scripts"
   # Or try:
   $poetryPath = "$env:LOCALAPPDATA\Programs\Python\Scripts"
   
   # Add to PATH for current session
   $env:Path += ";$poetryPath"
   ```

3. Restart your terminal

### Python not found

If you see "Python was not found", you need to install Python first:

1. Download Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart your terminal
4. Verify: `python --version`

### Virtual environment issues

```powershell
# Remove and recreate virtual environment
poetry env remove python
poetry install
```

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `poetry install` |
| Run CLI | `poetry run marshab --help` |
| Run web server | `poetry run python -m marshab.web.server` |
| Add dependency | `poetry add package-name` |
| Update dependencies | `poetry update` |
| Show environment info | `poetry env info` |



