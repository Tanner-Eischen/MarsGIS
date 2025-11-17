# Quick Start - MarsHab Web UI

## Starting the Application

### Step 1: Start the Backend API Server

Open a **new PowerShell terminal** and run:

```powershell
cd c:\Users\tanne\Gauntlet\MarsGIS
poetry run python -m marshab.web.server
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Keep this terminal open** - the server needs to keep running.

### Step 2: Start the Frontend

Open a **second PowerShell terminal** and run:

```powershell
cd c:\Users\tanne\Gauntlet\MarsGIS\webui
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:4000/
  ➜  Network: use --host to expose
```

### Step 3: Access the Application

1. Open your browser and go to: **http://localhost:4000**
2. The frontend will automatically connect to the backend API on port 5000

## Ports

- **Frontend**: http://localhost:4000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs

## Troubleshooting

### Backend won't start
- Make sure port 5000 is not in use: `netstat -ano | findstr ":5000"`
- Check that all dependencies are installed: `poetry install`
- Look for error messages in the terminal

### Frontend can't connect
- Ensure the backend is running (check terminal for "Uvicorn running" message)
- Check browser console for CORS errors
- Verify the backend is on port 5000

### CORS Errors
- The backend is configured to allow requests from ports 3000, 4000, and 4001
- If you see CORS errors, restart the backend server

## Stopping the Servers

- **Backend**: Press `Ctrl+C` in the backend terminal
- **Frontend**: Press `Ctrl+C` in the frontend terminal




