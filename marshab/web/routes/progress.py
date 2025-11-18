"""WebSocket progress tracking endpoints."""

import uuid
import asyncio
import time
from queue import Queue, Empty
from typing import Dict, Optional, Callable
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Store active WebSocket connections by task_id
_active_connections: Dict[str, WebSocket] = {}

# Store progress queues for each task (for thread-safe communication)
# Use a regular queue that can be accessed from any thread
_progress_queues: Dict[str, Queue] = {}


class ProgressEvent(BaseModel):
    """Progress event model."""
    task_id: str
    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    estimated_seconds_remaining: Optional[int] = None


class ProgressTracker:
    """Tracks progress for long-running operations."""
    
    def __init__(self, task_id: str):
        """Initialize progress tracker.
        
        Args:
            task_id: Unique task identifier
        """
        self.task_id = task_id
        self.current_stage = ""
        self.current_progress = 0.0
        self.start_time = time.time()
    
    def update(
        self,
        stage: str,
        progress: float,
        message: str,
        estimated_seconds_remaining: Optional[int] = None
    ):
        """Update progress (thread-safe, can be called from any thread).
        
        Args:
            stage: Stage name (e.g., "dem_loading", "terrain_metrics")
            progress: Progress value (0.0 to 1.0)
            message: Human-readable message
            estimated_seconds_remaining: Optional estimated time remaining
        """
        self.current_stage = stage
        self.current_progress = max(0.0, min(1.0, progress))
        
        # Calculate estimated time remaining if not provided
        if estimated_seconds_remaining is None and self.start_time:
            elapsed = time.time() - self.start_time
            if progress > 0:
                total_estimated = elapsed / progress
                estimated_seconds_remaining = int(total_estimated - elapsed)
        
        event = ProgressEvent(
            task_id=self.task_id,
            stage=stage,
            progress=self.current_progress,
            message=message,
            estimated_seconds_remaining=estimated_seconds_remaining
        )
        
        # Put event in queue (thread-safe)
        # If queue doesn't exist yet (WebSocket not connected), create it
        # This allows progress updates to be buffered before WebSocket connects
        if self.task_id not in _progress_queues:
            # Create thread-safe queue (can be created from any thread)
            _progress_queues[self.task_id] = Queue(maxsize=100)
        
        try:
            _progress_queues[self.task_id].put_nowait(event)
        except:
            # Queue full, drop event
            logger.warning(f"Progress queue full for task {self.task_id}, dropping event")


@router.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for progress updates.
    
    Clients connect to this endpoint to receive real-time progress updates
    for a specific task.
    """
    await websocket.accept()
    _active_connections[task_id] = websocket
    
    # Use existing queue if it exists (created by progress tracker), otherwise create new one
    if task_id not in _progress_queues:
        _progress_queues[task_id] = Queue(maxsize=100)
    
    # Get the thread-safe queue
    sync_queue = _progress_queues[task_id]
    
    logger.info(f"WebSocket connected for task {task_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "message": "Connected to progress stream"
        })
        
        # Start background task to send progress events
        async def send_progress():
            while True:
                try:
                    # Poll the thread-safe queue (non-blocking)
                    try:
                        event = sync_queue.get_nowait()
                        await websocket.send_json(event.model_dump())
                    except Empty:
                        # Queue empty, wait a bit then check again
                        await asyncio.sleep(0.1)
                        # Send ping to keep connection alive
                        try:
                            await websocket.send_json({"type": "ping"})
                        except:
                            break
                except Exception as e:
                    logger.warning(f"Error sending progress: {e}")
                    break
        
        # Start progress sender
        progress_task = asyncio.create_task(send_progress())
        
        # Keep connection alive and wait for messages
        while True:
            try:
                # Wait for client messages (ping/pong or disconnect)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        # Clean up connection and queue
        if task_id in _active_connections:
            del _active_connections[task_id]
        if task_id in _progress_queues:
            del _progress_queues[task_id]
        logger.info(f"WebSocket disconnected for task {task_id}")


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())

