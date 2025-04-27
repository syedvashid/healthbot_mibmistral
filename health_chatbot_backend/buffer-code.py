from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain
import logging
import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from fastapi.responses import StreamingResponse
import os
from datetime import datetime

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Setup FastAPI App
app = FastAPI()

# 2. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Pydantic Model for Input Validation
class AnalysisRequest(BaseModel):
    current_symptoms: str = Field(..., description="Description of the user's current symptoms.")
    conversation_history: List[Dict[str, str]] = Field(..., description="List of previous conversation turns.")
    generate_report: bool = Field(False, description="Whether to generate a report.")
    generate_department_suggestion: bool = Field(False, description="Whether to generate a department suggestion.")

# 4. Initialize Ollama LLM
llm = ChatOllama(model="biomistral", temperature=0.9, max_tokens=50)

# 5. LangChain Setup: Prompt Template and Memory for Questions
system_prompt = """You are a helpful assistant designed to gather information about a user's health symptoms. 
    Ask concise and relevant questions. If the user asks a question that is not related to health, respond with: 
    "This chatbot is for health-related queries only." Then follow up with a relevant health question."""

human_template = "{current_symptoms}"
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(human_template),
])

# 6. New Prompt Template for Department Suggestion
department_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="""
You are a medical assistant. Based on the user's conversation at the end of chat, identify which medical department they should consult.
Choose from: Cardiology, Neurology, General Medicine, Orthopedics, Dermatology, ENT, Psychiatry, Gynecology, Gastroenterology.

Conversation:
{conversation_history}

Answer in one line with only the department name.
""",
)

# 7. Prompt Template for Report Generation
report_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="""
Generate a concise medical report summarizing the following conversation. Include the user's symptoms, 
    any questions you asked, and the final department suggested. Format the report for a doctor to quickly review.

Conversation:
{conversation_history}

Report:
""",
)

# 8. Define the /analyze POST Endpoint
@app.post("/analyze")
async def analyze_symptoms(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        # Initialize memory for this request.
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load conversation history into memory
        for message in request.conversation_history:
            if message["role"] == "human":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "ai":
                memory.chat_memory.add_ai_message(message["content"])

        # Create LLMChain to generate follow-up questions
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
        response = await chain.arun(current_symptoms=request.current_symptoms)

        # Get updated conversation history
        updated_history = memory.load_memory_variables({})["chat_history"]

        # Handle report generation and department suggestion only if requested
        report_text = ""
        if request.generate_report:
            report_text = await generate_report(conversation_history=updated_history)

        suggested_department = ""
        if request.generate_department_suggestion:
            conversation_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in updated_history])
            department_chain = LLMChain(llm=llm, prompt=department_prompt, verbose=True)
            suggested_department = await department_chain.arun(conversation_history=conversation_str)

        # Return result
        result = {
            "followup_questions": response,
            "suggested_department": suggested_department.strip() if suggested_department else "",
            "conversation_history": updated_history,
            "report": report_text,
        }
        logger.info(f"Response: {result}")

        if request.generate_report:
            background_tasks.add_task(generate_report_and_save_pdf, updated_history)

        return result

    except Exception as e:
        error_message = f"Error processing request: {e}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


def generate_report_and_save_pdf(conversation_history: str):
    """
    Generates a PDF report from the conversation history and saves it to the server.
    """
    try:
        report_text = generate_report(conversation_history) # Removed await

        # Create a buffer to store the PDF data
        buffer = BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        Story = []
        # Add a title
        Story.append(Paragraph("Medical Report", styles['Title']))
        Story.append(Paragraph(report_text, styles['Normal']))
        doc.build(Story)

        # Define the directory and filename
        save_directory = "reports"
        os.makedirs(save_directory, exist_ok=True)
        filename = f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(save_directory, filename)

        # Save the PDF to a file
        with open(filepath, "wb") as f:
            f.write(buffer.getvalue())

        logger.info(f"PDF report generated and saved to {filepath}")

    except Exception as e:
        error_message = f"Error generating and saving PDF report: {e}"
        logger.error(error_message)


async def generate_report(conversation_history: str) -> str:
    """Generates the report text."""
    llm = ChatOllama(model="biomistral", temperature=0.5, max_tokens=250)
    report_chain = LLMChain(llm=llm, prompt=report_prompt, verbose=True)
    report_text = await report_chain.arun(conversation_history=conversation_history)
    return report_text
