from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat

app = FastAPI()


class QueryRequest(BaseModel):
    requirements: str
    resume_skills: list[str]


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat
import re
import ast

app = FastAPI()

class QueryRequest(BaseModel):
    requirements: str
    resume_skills: list[str]


@app.post("/skill-generator")
async def generate_skills(request: QueryRequest):
    # Enhanced prompt with more explicit instructions
    prompt = f"""
    CRITICAL INSTRUCTIONS: You are a resume skill matcher. Follow these rules EXACTLY:

    1. You have a list of RESUME SKILLS: {request.resume_skills}
    2. You have JOB REQUIREMENTS: "{request.requirements}"
    
    YOUR TASK:
    - Find skills that appear in BOTH the job requirements AND the resume skills list, do not hard match rather look for relevancy.
    - ONLY return skills that exist in the resume skills list or are relevant/similar: {request.resume_skills}
    - If a skill from job requirements is NOT in the resume skills list or is irrelevant, DO NOT include it
    - Return ONLY a valid Python list format: ["skill1", "skill2"]
    
    EXAMPLES:
    - If resume skills are ["Python", "SQL"] and job mentions "Python, JavaScript", return ["Python"]
    - If resume skills are ["Java", "React"] and job mentions "Python, Machine Learning", return []
    - If resume skills are ["Python", "Machine Learning", "SQL"] and job mentions "Python and ML experience", return ["Python", "Machine Learning"]

    Resume skills available: {request.resume_skills}
    Job requirements: {request.requirements}
    
    Return ONLY the Python list, no explanations:
    """

    try:
        response = chat(
            model='gemma3:12b',
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return {"response": response['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8000, reload=True)