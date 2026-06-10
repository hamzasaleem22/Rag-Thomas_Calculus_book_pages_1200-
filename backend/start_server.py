"""Start the uvicorn server in a background thread for testing."""
import threading
import time
import uvicorn
import requests

class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.config = uvicorn.Config(
            "app.api.endpoints:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True

if __name__ == "__main__":
    thread = ServerThread()
    thread.start()
    time.sleep(3)
    # Test health
    for attempt in range(10):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            print(f"Health: {r.status_code} {r.json()}")
            break
        except requests.exceptions.ConnectionError:
            if attempt == 9:
                print("Server failed to start")
            else:
                time.sleep(2)
    print("Server is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        thread.stop()
        print("Server stopped.")
