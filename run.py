#!/usr/bin/env python3
"""
AI Workout Coach - Run Script

Simple script to start the application.
"""
import os
import sys
import webbrowser
from threading import Timer

def open_browser():
    """Open the browser after a short delay"""
    webbrowser.open('http://localhost:8000/app')

def main():
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Check for dependencies
    try:
        import uvicorn
        import fastapi
        import mediapipe
        import cv2
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print("🏋️ AI Workout Coach")
    print("=" * 40)
    print("Starting server...")
    print("\n📍 Open http://localhost:8000/app in your browser")
    print("   (or it will open automatically in 2 seconds)")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Open browser after 2 seconds
    Timer(2, open_browser).start()
    
    # Start the server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()


