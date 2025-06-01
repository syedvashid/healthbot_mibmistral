from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
import json
import logging
from pathlib import Path

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Router
router = APIRouter()

# LLM Initialization
llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    max_tokens=300,
    timeout=30
)

# Pydantic Model for Request Validation
class OfflineReportRequest(BaseModel):
    name: str
    age: int
    gender: str
    department: str

# JSON Report Prompt
JSON_REPORT_PROMPT = """
Generate a JSON-formatted medical diagnosis report based on the following details:

**Patient Details**
- Name: {name}
- Age: {age}
- Gender: {gender}
- Department: {department}

Add diagnosis questions with multiple-choice answers. Each question should include:
- Question ID
- Question text
- Four options with associated EHR terms

Example format:
{
    "patientDetails": {
        "name": "John Doe",
        "age": 30,
        "gender": "Male",
        "department": "General Medicine"
    },
    "diagnosisQuestions": [
        {
            "id": 1,
            "text": "What is the duration of your symptoms?",
            "options": [
                { "option": "A", "text": "Less than 2 days", "ehrTerm": "Acute onset" },
                { "option": "B", "text": "3–5 days", "ehrTerm": "Subacute onset" },
                { "option": "C", "text": "6–10 days", "ehrTerm": "Prolonged illness" },
                { "option": "D", "text": "More than 10 days", "ehrTerm": "Chronic symptoms" }
            ]
        }
    ]
}

JSON format only.
"""

@router.post("/generate_offline_report")
async def generate_offline_report(request: OfflineReportRequest):
    """
    Endpoint to generate a JSON-formatted medical diagnosis report using LLM and save it as a file.
    """
    try:
        # Generate JSON report using LLM
        prompt = PromptTemplate(
            input_variables=["name", "age", "gender", "department"],
            template=JSON_REPORT_PROMPT
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        json_report = await chain.arun(
            name=request.name,
            age=request.age,
            gender=request.gender,
            department=request.department
        )

        # Validate and parse JSON
        try:
            report = json.loads(json_report)
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"Failed to parse generated JSON: {e}")

        # Save JSON to a file
        report_path = Path(f"reports/{request.name.lower().replace(' ', '_')}_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return {"message": "Report generated successfully", "file_path": str(report_path)}
    except Exception as e:
        logger.error(f"JSON report error: {e}")
        raise HTTPException(500, "JSON report generation failed")