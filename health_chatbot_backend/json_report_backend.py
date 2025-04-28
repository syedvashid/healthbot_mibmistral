from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize LLM
llm = ChatOllama(
    model="biomistral",
    temperature=0.7,
    max_tokens=2000,
    timeout=30
)

# Define request model
class JSONReportRequest(BaseModel):
    conversation_history: List[Dict[str, str]]
    age: str
    gender: str
    department: str

# Prompt template for generating JSON report
JSON_PROMPT = """
Generate a JSON report for patient assessment based on the following details:

- Age: {age}
- Gender: {gender}
- Department: {department}
- Conversation History:
{conversation_history}

The JSON should include:
1. A list of 5 health-related questions, each with 4 options and their respective EHR-specific terms.
2. Auto-flagging rules for high, medium, and low risk based on question responses.

Format the output as valid JSON.
"""

@app.post("/generate_json_report")
async def generate_json_report(request: JSONReportRequest):
    try:
        # Format conversation history as a string
        conversation_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in request.conversation_history
        )
        
        # Prompt preparation
        prompt = PromptTemplate(
            input_variables=["age", "gender", "department", "conversation_history"],
            template=JSON_PROMPT
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate JSON report
        json_report = await chain.arun(
            age=request.age,
            gender=request.gender,
            department=request.department,
            conversation_history=conversation_history
        )
        
        return {"json_report": json_report.strip()}
    
    except Exception as e:
        logger.error(f"Error generating JSON report: {str(e)}")
        raise HTTPException(500, "Failed to generate JSON report")