#!/usr/bin/env python3
"""
Simple HTTP server to serve the chess web application
"""
import http.server
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to the repository directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if not os.path.exists(DIRECTORY):
        print(f"Error: {DIRECTORY} directory not found")
        sys.exit(1)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving Heniya Chess AI at http://localhost:{PORT}")
        print(f"Directory: {os.path.abspath(DIRECTORY)}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main()