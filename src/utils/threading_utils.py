"""
Threading Utilities
===================

This module provides thread-safe classes for passing data between
different pipeline stages (capture -> detection -> analysis -> visualizer).
"""

import threading
import queue
from typing import Any, Optional


class PipelineQueue:
    """
    A wrapper around queue.Queue to handle thread-safe item passing,
    allowing elements to be dropped if the queue is full (to prioritize fresh data).
    """
    
    def __init__(self, maxsize: int = 5):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
    
    def put(self, item: Any, drop_oldest: bool = True) -> bool:
        """
        Put an item into the queue.
        
        Args:
            item: The item to add.
            drop_oldest: If True and queue is full, drops the oldest item.
            
        Returns:
            True if inserted successfully, False otherwise.
        """
        with self._lock:
            if self._queue.full():
                if drop_oldest:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                else:
                    return False
            
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                return False
                
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Retrieve an item from the queue.
        """
        try:
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
            
    def qsize(self) -> int:
        return self._queue.qsize()
        
    def clear(self):
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
