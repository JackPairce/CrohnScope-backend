"""
Scheduler module for AI training tasks

This module provides scheduling functionality to periodically check for
training conditions and trigger model training when needed.
"""

import threading
from app.services.ai.train import start_training_if_needed, RETRAINING_THRESHOLD


class TrainingScheduler:
    """Scheduler to periodically check and trigger training when needed."""

    def __init__(self, check_interval=60):
        """
        Initialize the training scheduler.

        Args:
            check_interval: Time in seconds between checks for training conditions
        """
        self.check_interval = check_interval  # Check every minute by default
        self._stop_event = threading.Event()
        self._thread = None
        self._running = False

    def _run(self):
        """Main scheduler loop that checks for training conditions."""
        while not self._stop_event.is_set():
            start_training_if_needed()
            # Wait for the next check interval or until stop is called
            self._stop_event.wait(self.check_interval)

    def start(self):
        """Start the scheduler in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            # Already running
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        self._running = True
        print(f"Training scheduler started with check interval: {self.check_interval}s")
        return True

    def stop(self):
        """Stop the scheduler."""
        if self._thread is None or not self._thread.is_alive():
            # Not running
            return False

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._running = False
        print("Training scheduler stopped")
        return True

    @property
    def is_running(self):
        """Check if the scheduler is currently running."""
        return self._running and self._thread is not None and self._thread.is_alive()


# Create a singleton scheduler instance
scheduler = TrainingScheduler()
