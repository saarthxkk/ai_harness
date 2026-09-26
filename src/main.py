"""AI Coding-Agent Harness — Entry Point."""

import os
import sys


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
