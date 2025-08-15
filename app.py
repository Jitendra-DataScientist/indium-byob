from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat

app = FastAPI()


class QueryRequest(BaseModel):
    requirements: str

@app.post("/skill-generator")
async def generate_skills(request: QueryRequest):
    prompt = f"""
            Given the following job requirements:
            {request.requirements}
            Write a list of key technical skills. Only return the key technical skills as a python list. Do not include anythin else.
            Your response should not contain something in brackets like:
            API Testing (REST, SOAP)
            or
            Selenium/TestNG/JUnit (or equivalent automation tools)
            rather separate everything into separate entities like:
            REST, SOAP, Selenium, TestNG, JUnit
            Also, just include hardcore technical skills like python, java, sql, machine learning etc., not stuff like:
            Credit Card Domain Knowledge, Banking Domain Knowledge, etc.
            """
    prompt = f"""Given the following job requirements:
                {request.requirements}

                Extract only the core technical skills (e.g., Python, Java, SQL, Machine Learning) and return them **as a valid Python list of strings**.

                Rules:
                1. No descriptions or extra text — output only the list.
                2. Split grouped skills into separate items. For example:
                - "API Testing (REST, SOAP)" → ["REST", "SOAP"]
                - "Selenium/TestNG/JUnit (or equivalent automation tools)" → ["Selenium", "TestNG", "JUnit"]
                3. Exclude domain knowledge or soft skills (e.g., "Credit Card Domain Knowledge", "Banking Domain Knowledge").
                4. Include only hard technical skills.
                """
    try:
        response = chat(
            model='gemma3',
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