# Poetry Quick Start Guide for MarsHab

## Your Poetry Installation

Poetry is installed at: `C:\Users\tanne\AppData\Roaming\Python\Scripts\poetry.exe`

## Adding Poetry to PATH (Optional but Recommended)

To use `poetry` directly without the full path, add it to your system PATH:

1. Open System Properties → Environment Variables
2. Edit "Path" variable (User or System)
3. Add: `C:\Users\tanne\AppData\Roaming\Python\Scripts`
4. Restart your terminal

Or add to PATH for current PowerShell session:
```powershell
$env:Path += ";C:\Users\tanne\AppData\Roaming\Python\Scripts"
```

## Common Commands

### Project Setup (Already Done ✅)
```powershell
cd C:\Users\tanne\Gauntlet\MarsGIS
poetry install  # ✅ Already completed
```

### Running the Application

```powershell
# Run CLI commands
poetry run marshab --help
poetry run mars download mola --roi "40,41,180,181"
poetry run mars terrain --roi "40,41,180,181"
poetry run mars navigation 1 --start-lat 40.0 --start-lon 180.0

# Run the web server (backend)
poetry run python -m marshab.web.server

# Or use the shortcut script
.\start_web_server.ps1
```

### Managing Dependencies

```powershell
# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Remove a dependency
poetry remove package-name

# Update all dependencies
poetry update

# Update lock file after changing pyproject.toml
poetry lock
```

### Virtual Environment

```powershell
# Show virtual environment info
poetry env info

# Activate virtual environment (optional - poetry run works without this)
poetry shell

# Show virtual environment path
poetry env info --path
```

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `poetry install` |
| Run CLI | `poetry run marshab --help` |
| Run web server | `poetry run python -m marshab.web.server` |
| Run frontend | `cd webui; npm run dev` |
| Add dependency | `poetry add package-name` |
| Update dependencies | `poetry update` |
| Show env info | `poetry env info` |

## Troubleshooting

### If `poetry` command not found:

**Option 1: Use full path**
```powershell
C:\Users\tanne\AppData\Roaming\Python\Scripts\poetry.exe --version
```

**Option 2: Add to PATH for current session**
```powershell
$env:Path += ";C:\Users\tanne\AppData\Roaming\Python\Scripts"
```

**Option 3: Use Python module**
```powershell
py -m pip install poetry
py -m poetry --version
```

### If dependencies fail to install:

```powershell
# Clear cache and retry
poetry cache clear pypi --all
poetry install
```

## Next Steps

Now that dependencies are installed, you can:

1. **Start the backend server:**
   ```powershell
   poetry run python -m marshab.web.server
   ```

2. **Start the frontend (in another terminal):**
   ```powershell
   cd webui
   npm run dev
   ```

3. **Access the web UI:**
   - Frontend: http://localhost:4000
   - Backend API: http://localhost:5000
   - API Docs: http://localhost:5000/docs

4. **Test the new features:**
   - Polygonal site geometries in 2D map
   - 3D terrain viewer with interactive controls
   - Site and waypoint overlays



