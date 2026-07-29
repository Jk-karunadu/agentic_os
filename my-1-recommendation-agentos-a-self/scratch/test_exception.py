import requests
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import time

app = FastAPI()

@app.get("/test")
def test():
    raise HTTPException(status_code=503, detail="Ollama request failed: ReadTimeout(something)")

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="error")

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
time.sleep(2)

try:
    response = requests.get("http://127.0.0.1:8002/test")
    response.raise_for_status()
except Exception as e:
    print("EXCEPTION STRING:")
    print(str(e))
