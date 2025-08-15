from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError
from fastapi.responses import JSONResponse
import os
from ollama import chat

# Configure app
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)

# Optional: control GPU and model memory usage
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

# Input schema
class QueryRequest(BaseModel):
    requirements: str

# Graceful validation error handler
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Background task function
def run_chat_task(requirements: str, result_holder: dict, task_id: str):
    prompt = (
        "Given the following job requirements:\n"
        f"{requirements}\n"
        "Write a list of key technical skills. Only return the key technical skills separated by comma. Do not include any explanation."
    )
    try:
        result = chat(model="gemma3", messages=[{"role": "user", "content": prompt}])
        result_holder[task_id] = {"status": "completed", "skills": result["message"]["content"]}
    except Exception as e:
        result_holder[task_id] = {"status": "error", "error": str(e)}

# In-memory store for task results
task_results: dict[str, dict] = {}

# Endpoint to generate skills—returns immediately with a task ID
@app.post("/skill-generator")
async def generate_skills(request: QueryRequest, background_tasks: BackgroundTasks):
    import uuid
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "pending"}

    background_tasks.add_task(run_chat_task, request.requirements, task_results, task_id)
    return {"task_id": task_id, "status": "pending"}

# Endpoint to check the task status and get results
@app.get("/skill-generator/{task_id}")
async def get_skill_result(task_id: str):
    result = task_results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **result}

# Standard uvicorn target for development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, workers=4)
