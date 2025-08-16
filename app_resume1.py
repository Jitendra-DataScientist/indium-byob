from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat
import re
import ast

app = FastAPI()

class QueryRequest(BaseModel):
    requirements: str
    resume_skills: list[str]


def extract_matching_skills(requirements: str, resume_skills: list[str]) -> list[str]:
    """
    Fallback function to extract matching skills using keyword matching
    """
    matched_skills = []
    requirements_lower = requirements.lower()
    
    for skill in resume_skills:
        # Check if skill (case-insensitive) appears in requirements
        if skill.lower() in requirements_lower:
            matched_skills.append(skill)
    
    return matched_skills


@app.post("/skill-generator")
async def generate_skills(request: QueryRequest):
    # More explicit prompt with examples
    prompt = f"""
Return a python list of skills that the candidate has relevant to the job requirements. If none of the candidate's skills are relevant, return an empty python list.

Candidate SKILLS (these are the ONLY skills you can return in the response):
{request.resume_skills}

JOB REQUIREMENTS: 
{request.requirements}

Your response should have nothing else apart the python list.

"""

    try:
        response = chat(
            model='gemma3:latest',
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_response = response['message']['content']
        
        # Clean up the response
        cleaned = (
            raw_response.replace("```python", "")
                      .replace("```", "")
                      .strip()
        )
        '''
        # Try to parse the response
        try:
            result = ast.literal_eval(cleaned)
            
            # Validate that all returned skills are in the resume_skills list
            validated_result = []
            for skill in result:
                if skill in request.resume_skills:
                    validated_result.append(skill)
                else:
                    # Try case-insensitive matching
                    for resume_skill in request.resume_skills:
                        if skill.lower() == resume_skill.lower():
                            validated_result.append(resume_skill)
                            break
            
            return {
                "matched_skills": validated_result,
                "raw_response": raw_response
            }
        except (ValueError, SyntaxError):
            # If parsing fails, use fallback keyword matching
            fallback_result = extract_matching_skills(request.requirements, request.resume_skills)
            return {
                "matched_skills": fallback_result,
                "raw_response": raw_response,
                "note": "Used fallback keyword matching due to parsing error"
            }
            '''
        return {'response':raw_response}
    
    except Exception as e:
        # Final fallback
        fallback_result = extract_matching_skills(request.requirements, request.resume_skills)
        return {
            "matched_skills": fallback_result,
            "error": str(e),
            "note": "Used fallback keyword matching due to API error"
        }


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the API is working"""
    return {"status": "API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8000, reload=True)