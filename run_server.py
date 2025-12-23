import os
import sys
import uvicorn

def main():
    os.environ.setdefault("PORT", "8000")

    from server import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ["PORT"]),
        log_level="info",
    )

if __name__ == "__main__":
    main()
