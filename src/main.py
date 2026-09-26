from http.server import BaseHTTPRequestHandler
import json
import os
import sys


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function HTTP handler."""

    def do_GET(self):
        api_key = os.environ.get("AI_API_KEY")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        body = {
            "status": "Harness ready",
            "api_key_configured": bool(api_key),
        }
        self.wfile.write(json.dumps(body).encode("utf-8"))


def main() -> None:
    api_key = os.environ.get("AI_API_KEY")

    if not api_key:
        print(
            "ERROR: AI_API_KEY environment variable is not set.\n"
            "Export it before running:  export AI_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Harness ready")
    sys.exit(0)


if __name__ == "__main__":
    main()
