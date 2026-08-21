"""
GestureSense - Hand Gesture Recognition System
Run this file to start the FastAPI server.
"""
import uvicorn
import socket
from app import app

def find_free_port(starting_port=8000):
    for port in range(starting_port, starting_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return starting_port

if __name__ == "__main__":
    port = find_free_port(8000)
    print("")
    print("  ================================================")
    print("  |    GestureSense is Starting...               |")
    print("  |----------------------------------------------|")
    print("  |  Hand Gesture Recognition System             |")
    print("  |  MediaPipe + Random Forest Classifier        |")
    print("  |                                              |")
    print(f"  |  Open in browser:                            |")
    print(f"  |  -> http://127.0.0.1:{port:<5}                    |")
    print("  |                                              |")
    print("  |  Supported Gestures:                         |")
    print("  |  Open Palm, Fist, Thumbs Up, Peace, OK       |")
    print("  ================================================")
    print("")
    uvicorn.run(app, host="127.0.0.1", port=port)
