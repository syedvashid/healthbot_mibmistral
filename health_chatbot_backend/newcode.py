from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory  # Removed unused import
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain
import logging
import sys
import re  # Import the regular expression module
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from fastapi.responses import StreamingResponse

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Setup FastAPI App
app = FastAPI()

# 2. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, for development. Change in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Pydantic Model for Input Validation
class AnalysisRequest(BaseModel):
    current_symptoms: str = Field(..., description="Description of the user's current symptoms.")
    conversation_history: List[Dict[str, str]] = Field(..., description="List of previous conversation turns.")

# 4. Initialize Ollama LLM
llm = ChatOllama(model="biomistral", temperature=0.5, max_tokens=250)  # Increased max_tokens to handle report generation

# 5. LangChain Setup: Prompt Template and Memory for Questions
system_prompt = """You are a helpful assistant designed to gather information about a user's health symptoms. 
    Ask concise and relevant questions. If the user asks a question that is not related to health, 
    respond with: "This chatbot is for health-related queries only."  Then follow up with a relevant health question."""

human_template = "{current_symptoms}"
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(human_template),
])

# 6. New Prompt Template for Department Suggestion
department_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="""
You are a medical assistant. Based on the user's conversation, identify which medical department they should consult.
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
Generate a concise medical report summarizing the following conversation.  Include the user's symptoms, 
    any questions you asked, and the final department suggested.  Format the report for a doctor to quickly review.

Conversation:
{conversation_history}

Report:
""",
)


# 8. Define the /analyze POST Endpoint
@app.post("/analyze")
async def analyze_symptoms(request: AnalysisRequest):
    try:
        # Initialize memory for this request.  Important to do it here!
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load conversation history into memory
        for message in request.conversation_history:
            if message["role"] == "human":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "ai":
                memory.chat_memory.add_ai_message(message["content"])

        # Check for irrelevant questions
        if not is_health_related(request.current_symptoms):
            response = "This chatbot is for health-related queries only.  Can you describe your current symptoms?"
            updated_history = memory.load_memory_variables({})["chat_history"]
            result = {
                "followup_questions": response,
                "suggested_department": "General Medicine",  # Default department
                "conversation_history": updated_history,
            }
            logger.info(f"Response: {result}")
            return result

        # Create LLMChain to generate follow-up questions
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
        response = await chain.arun(current_symptoms=request.current_symptoms)

        # Get updated conversation history
        updated_history = memory.load_memory_variables({})["chat_history"]

        # Combine history into string for department suggestion
        conversation_str = "\n".join(
            [f"{msg.type.upper()}: {msg.content}" for msg in updated_history]
        )

        # Create second chain for department suggestion
        department_chain = LLMChain(llm=llm, prompt=department_prompt, verbose=True)
        suggested_department = await department_chain.arun(conversation_history=conversation_str)

        # Create chain for report generation
        report_chain = LLMChain(llm=llm, prompt=report_prompt, verbose=True)
        report_text = await report_chain.arun(conversation_history=conversation_str)


        # Return result
        result = {
            "followup_questions": response,
            "suggested_department": suggested_department.strip(),
            "conversation_history": updated_history,  # Return the history for display
            "report": report_text
        }
        logger.info(f"Response: {result}")
        return result

    except Exception as e:
        error_message = f"Error processing request: {e}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)



@app.post("/generate_pdf")
async def generate_pdf(request_data: Dict[str, str]):
    """
    Generates a PDF report from the given text.
    """
    report_text = request_data.get("report_text", "")
    if not report_text:
        raise HTTPException(status_code=400, detail="Report text is required.")

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

    # Prepare response with PDF data
    headers = {
        "Content-Disposition": 'attachment; filename="medical_report.pdf"',
        "Content-Type": "application/pdf",
    }
    return StreamingResponse(buffer.getvalue(), headers=headers)



def is_health_related(text: str) -> bool:
    """
    Checks if the given text is related to health.  Uses a simple keyword check.
    """
    health_keywords = [
        "symptom", "pain", "fever", "cough", "cold", "flu", "injury", "disease",
        "condition", "illness", "health", "medical", "doctor", "hospital",
        "treatment", "medicine", "ওষুধ", "শরীর", "অসুবিধা", "ডাক্তার", "হাসপাতাল", "চিকিৎসা", "ঔষধ", #Bengali
        "ज्वर", "दर्द", "बीमारी", "इलाज", "दवा", "स्वास्थ्य", "चिकित्सा", "डॉक्टर", "अस्पताल", #Hindi
        "জ্বর", "গায়ে ব্যাথা", "অসুস্থ", "ডাক্তাৰ", "দৰব", "বেমাৰ", "চিকিৎসা",  # Assamese
        "jwar", "dard", "bimaaree", "ilaaj", "dava", "svaasthya", "chikitsa", "doktar", "haspataal" #Maithili
    ]
    text = text.lower()
    for keyword in health_keywords:
        if keyword in text:
            return True
    return False


# new feature add by gpt
@app.post("/generate_report")
async def generate_report(request_data: Dict[str, Any]):
    """
    Generates a medical report summarizing the entire conversation history.
    """
    conversation_history = request_data.get("conversation_history", [])

    # Convert conversation history to string format
    conversation_str = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history]
    )

    # Use a pre-existing prompt to generate a summary report
    report_chain = LLMChain(llm=llm, prompt=report_prompt, verbose=True)
    report_text = await report_chain.arun(conversation_history=conversation_str)

    return {"report": report_text}
