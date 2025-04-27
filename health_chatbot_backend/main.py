from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from langchain.chains import LLMChain
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from fastapi.responses import StreamingResponse
import asyncio

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    conversation_history: List[Dict[str, str]] = []

class HistoryRequest(BaseModel):
    conversation_history: List[Dict[str, str]]

# Initialize Ollama
llm = ChatOllama(
    model="biomistral",
    temperature=0.7,
    max_tokens=200,
    timeout=30
)

# System Prompts
MEDICAL_PROMPT = """You are a medical assistant. Ask concise follow-up questions about symptoms. 
If the query is non-medical, respond: "I specialize only in health-related topics." Then ask a health question."""

DEPARTMENT_PROMPT = """Analyze this health conversation and suggest ONE most relevant medical department:
Cardiology, Neurology, General Medicine, Orthopedics, Dermatology, ENT, Psychiatry, Gynecology, Gastroenterology.

Conversation:
{conversation_history}

Return ONLY the department name."""

REPORT_PROMPT = """Generate a professional medical report:

**Patient Summary**
- Chief Complaint: {chief_complaint}

**Clinical History**
{history}

**Assessment**
- Symptom analysis
- Potential concerns

**Recommendations**
- Suggested next steps

Format this for doctor review. Be concise yet comprehensive.

Conversation:
{conversation_history}"""

# Core Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        for msg in request.conversation_history:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(MEDICAL_PROMPT),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        response = await chain.arun(user_input=request.user_input)
        
        return {"response": response.strip()}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, "Chat processing failed")

# Department Suggestion
@app.post("/suggest_department")
async def suggest_department(request: HistoryRequest):
    try:
        conv_history = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in request.conversation_history
        )
        
        prompt = PromptTemplate(
            input_variables=["conversation_history"],
            template=DEPARTMENT_PROMPT
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        department = await chain.arun(conversation_history=conv_history)
        return {"department": department.strip()}
    
    except Exception as e:
        logger.error(f"Department error: {str(e)}")
        return {"department": "General Medicine"}

# PDF Report Generation
@app.post("/generate_report")
async def generate_report(request: HistoryRequest):
    try:
        # Extract chief complaint
        chief_complaint = next(
            (msg["content"] for msg in request.conversation_history 
             if msg["role"] == "user"),
            "Not specified"
        )
        
        conv_history = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in request.conversation_history
        )
        
        # Generate report text
        prompt = PromptTemplate(
            input_variables=["chief_complaint", "history", "conversation_history"],
            template=REPORT_PROMPT
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        report_text = await chain.arun(
            chief_complaint=chief_complaint,
            history="From conversation",
            conversation_history=conv_history
        )
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("Medical Consultation Report", styles['Title']))
        for paragraph in report_text.split('\n\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=medical_report.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Report error: {str(e)}")
        raise HTTPException(500, "Report generation failed")