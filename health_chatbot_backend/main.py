from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict,Any
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
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph ,Spacer
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import asyncio
from reportlab.lib.enums import TA_LEFT
import re
import json

import os
from dotenv import load_dotenv
from database import get_db_connection

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
    language: str  # New field for language support

class HistoryRequest(BaseModel):
    name: str
    gender: str
    age: int
    language: str  # New field for language
    conversation_history: List[Dict[str, str]]

class OfflineReportRequest(BaseModel):
    name: str
    age: int
    gender: str
    department: str
    language: str  # New field for language
    responses: List[Dict]


# ===== NEW PYDANTIC MODELS =====
class LocationRequest(BaseModel):
    city: str = None
    department: str = None  
    doctor_name: str = None
    language: str

# Initialize Ollama
llm = ChatOllama(
    model="llama3",
    temperature=0.7,
    max_tokens=500,
    timeout=30
)

# Enhanced Agentic System Prompts
GREETING_AGENT_PROMPT = """You are an intelligent medical assistant greeting agent. Generate a warm, professional greeting and ask the user what they need help with today.

Instructions:
- Generate a personalized greeting in {language}
- Ask how you can help them today
- Mention that you can help with medical diagnosis questions or appointment booking
- Ask user to chose between diagnoisis or appointment booking.
- Keep it natural and conversational
- Don't use rigid options like "A" or "B"

Generate a friendly greeting message."""

INTENT_DETECTION_PROMPT = """You are an intelligent intent detection agent. Analyze the user's message and determine their intent.

User message: {user_input}
Language: {language}
Current conversation context: {context}

Possible intents:
1. DIAGNOSIS - User wants medical diagnosis, health questions, symptoms analysis, medical consultation ,health diagnosis.
2. APPOINTMENT - User wants to book appointment, find doctor, schedule consultation 
3. SWITCH_TO_APPOINTMENT - User wants to switch from diagnosis to appointment booking by askin doctor name , department or appointment is mentioned , asking for doctor suggestion.
4. SWITCH_TO_DIAGNOSIS - User wants to switch from appointment to diagnosis by saying proble ,unable to understand, asking for medical questions.
5. UNCLEAR - Intent is not clear, need more information.

Analyze the message and return ONLY one of these words: DIAGNOSIS, APPOINTMENT, SWITCH_TO_APPOINTMENT, SWITCH_TO_DIAGNOSIS, or UNCLEAR"""

MEDICAL_PROMPT = """You are a professional medical assistant. Based on conversation history and preferred language, generate medical questions to gather information about the patient's condition.

Current question count: {question_count}
Total questions asked so far: {question_count}/5

Instructions:
- If question_count < 5: Generate the next multiple-choice question with exactly 4 options (A, B, C, D)
- Don't repeat previous questions and their answers simple generate question and their options.
- Each option must include **EHR-specific terminology** in parentheses
- The format for options should be: "A. Description (EHR Term)"
- Add a **new line** between each question and between each option for better readability
- All questions and options must be in the selected language: {language}
- If question_count >= 5: Instead of generating more questions, recommend consulting a doctor and suggest booking an appointment.Tell user to type "Book Appointment or say "Appointment" for confermation.

Conversation History:
{conversation_history}
Return the appropriate response based on question count."""



SMART_APPOINTMENT_PROMPT = """You are an intelligent appointment booking assistant. 

Current conversation:
{conversation_history}

Available doctors from database:
{doctors_info}

User's latest message: {user_input}
Language: {language}

CRITICAL INSTRUCTIONS:
- When doctors are available, you MUST present ALL doctors from the database results
- NEVER select just one doctor - always show the complete list
- Format each doctor as: "• Dr. [Name] - [Department] - Available: [Timings]"
- After listing all doctors, ask user to choose which doctor they prefer
- If this is the start of appointment flow, ask what type of doctor or department they need
- Don't invent doctors not in the database
- If no doctors found, ask for different department or doctor name and tell them to ensure the doctor name/city  spelling is correct.
- Be helpful and guide the user naturally
- Respond in {language}

EXAMPLE FORMAT when multiple doctors available:
"Here are all our available cardiologists:

- Dr. [Name1] - Cardiology - Available: [Timings1]
- Dr. [Name2] - Cardiology - Available: [Timings2] 
- Dr. [Name3] - Cardiology - Available: [Timings3]

Which doctor would you like to book an appointment with?"

Generate appropriate response based on the context."""






LOCATION_COLLECTION_PROMPT = """You are a location and preference collection agent for appointment booking.

Current conversation: {conversation_history}
User input: {user_input}
Language: {language}

Current Status:
- City collected: {city_status}
- Department/Doctor preference collected: {preference_status}

INSTRUCTIONS:
1. If CITY is missing: Ask for city/location in a friendly way

2. If DEPARTMENT/DOCTOR preference is missing: Ask user to specify either:
   - Department they need (Cardiology, Pediatrics, etc.)
   - OR specific doctor name they want to see
3. Ask for both at a time .       
3. If BOTH collected: Confirm and proceed to search
4 Prefer/ current input department or name or location instead using from previous chat histroy

Available cities in our system: Kanpur, Orai, Jhansi

IMPORTANT:
- Be conversational and helpful
- Ask ONE thing at a time (city first, then preference)
- Respond in {language}
- Don't overwhelm user with too many options

Generate appropriate response based on what's missing."""

ENHANCED_DOCTOR_DISPLAY_PROMPT = """You are displaying doctor search results to help user choose.

Search Results for: {search_criteria}
City: {city}
{doctors_info}

INSTRUCTIONS:
- Present ALL doctors found in an organized, easy-to-read format
- Include complete information: name, department, location, timings
- Use emojis for better readability (📍 for location, 📞 for contact, 🕒 for timings)
- After listing all doctors, ask user which doctor they prefer
- If no doctors found, ask for different department or doctor in their city. 
- If no doctors found, Suggest user to ensure the doctor's name/city spelling is correct.
- Be helpful and guide the user naturally
- Never invent doctors from your side always use from database
- Respond in {language}

EXAMPLE FORMAT:
"Here are the available doctors in [City] for [Department/Search]:

🏥 **Dr. [Name]** - [Department]
   📍 [Location]
   🕒 Available: [Timings]

🏥 **Dr. [Name]** - [Department]  
   📍 [Location]
   🕒 Available: [Timings]

Which doctor would you like to book an appointment with?"

Generate the response showing all doctors."""







DEPARTMENT_PROMPT = """Analyze this health conversation and suggest ONE most relevant medical department:
Cardiology, Neurology, General Medicine, Orthopedics, Dermatology, ENT, Psychiatry, Gynecology, Gastroenterology.

Conversation:
{conversation_history}

Return ONLY the department name."""

REPORT_PROMPT = """Generate a comprehensive and professional pre-medical consultation assessment report with structured formatting and clarity. The report should include the following sections:

**Questions and Responses**
- Include all questions asked during the consultation along with the response provided by the patient in {language}.
- Each question should be clearly listed with its text and the available options (A, B, C, D) displayed on separate lines in {language}.
- Highlight the selected option on its own line in **bold** for emphasis.
- Do not invent or assume any additional questions beyond those {conversation_history}.

**Patient Summary**
- Provide a concise summary of the patient's condition based on the selected responses {language}.
- Reference specific questions and options to justify the overview.
- Chief Complaint: {chief_complaint}.

**Clinical History**
{history}

**Assessment**
- Evaluate the symptoms described by the patient and identify any potential areas of concern.
- Ensure consistency between the analysis and the responses provided to the questions.

**Recommendations**
- Based on the selected responses, classify the case as **High Risk**, **Medium Risk**, or **Low Risk**.
- Justify the classification using system-defined rules.

**Formatting Guidelines**
- Add proper line spacing between sections to ensure readability.
- Use **bold headings** and properly indent the content under each heading.
- Maintain a professional tone and concise language appropriate for medical review.
"""

OFFLINE_REPORT_PROMPT = """ Based on the following patient details:
            
            - Age: {age}
            - Gender: {gender}
            - Problem: {department}
            - Responses: {responses}
            - Language: {language}
 - Generate text (questions and their options) must be in specific {language}. 
 Generate  5 questions to gather information about the patient's condition. Each question should have exactly 4 options in Language .
 Provide EHR-specific terminology in parentheses for each option. 
 Help with auto flagging rules for high risk cases. 
 Return the questions and options in JSON format.
        
Provide a concise yet professional summary for doctor review."""

@app.get("/")
async def root():
    print("Function: root")
    return {"message": "Welcome to the Enhanced Agentic Health Chatbot Backend API!"}

# Enhanced Agentic Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    print("Function: chat")
    try:
        # 1. Handle initial greeting - Generate dynamic greeting
        if not request.conversation_history:
            greeting_response = await generate_greeting(request.language)
            return {"response": greeting_response}

        # 2. Check current flow and question count
        current_flow = get_current_flow(request.conversation_history)
        question_count = count_questions_asked(request.conversation_history)
        
        # 3. Detect user intent (including flow switching)
        conv_context = get_conversation_context(request.conversation_history)
        intent = await detect_user_intent(request.user_input, request.language, conv_context)
        
        # 4. Handle flow switching
        if intent == "SWITCH_TO_APPOINTMENT":
            # Switch from diagnosis to appointment
            update_flow_marker(request.conversation_history, "appointment")
            return await handle_smart_appointment_flow(request)
        
        elif intent == "SWITCH_TO_DIAGNOSIS":
            # Switch from appointment to diagnosis
            update_flow_marker(request.conversation_history, "diagnosis")
            return await handle_diagnosis_flow(request)
        
        # 5. Handle existing flows
        if current_flow == "diagnosis":
            # Check if we should transition to appointment after 5 questions
            if question_count >= 5 and intent == "APPOINTMENT":
                update_flow_marker(request.conversation_history, "appointment")
                return await handle_smart_appointment_flow(request)
            else:
                return await handle_diagnosis_flow(request, question_count)
        
        elif current_flow == "appointment":
            # Check if user wants diagnosis instead
            if intent == "DIAGNOSIS":
                update_flow_marker(request.conversation_history, "diagnosis")
                return await handle_diagnosis_flow(request)
            else:
                return await handle_smart_appointment_flow(request)
        
        else:
            # 6. No flow determined yet - use intelligent intent detection
            if intent == "DIAGNOSIS":
                request.conversation_history.append({
                    "role": "system", 
                    "content": "selected_flow: diagnosis"
                })
                return await handle_diagnosis_flow(request)
            
            elif intent == "APPOINTMENT":
                request.conversation_history.append({
                    "role": "system", 
                    "content": "selected_flow: appointment"
                })
                return await handle_smart_appointment_flow(request)
            
            else:  # UNCLEAR intent
                clarification_response = await generate_clarification(request.user_input, request.language)
                return {"response": clarification_response}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, "Chat processing failed")

# Agentic Helper Functions
async def generate_greeting(language: str) -> str:
    print("Function: generate_greeting")
    """Generate dynamic, personalized greeting"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(GREETING_AGENT_PROMPT.format(language=language))
        ])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(input="")
        return response.strip()
    except Exception as e:
        logger.error(f"Greeting generation error: {str(e)}")
        return "Hello! How can I help you today? I can assist with medical diagnosis questions or appointment booking."

async def detect_user_intent(user_input: str, language: str, context: str = "") -> str:
    print("Function: detect_user_intent")
    """Intelligently detect user's intent including flow switching"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                INTENT_DETECTION_PROMPT.format(
                    user_input=user_input, 
                    language=language,
                    context=context
                )
            )
        ])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(input="")
        
        # Extract intent from response
        intent = response.strip().upper()
        valid_intents = ["DIAGNOSIS", "APPOINTMENT", "SWITCH_TO_APPOINTMENT", "SWITCH_TO_DIAGNOSIS", "UNCLEAR"]
        if intent in valid_intents:
            return intent
        else:
            return "UNCLEAR"
            
    except Exception as e:
        logger.error(f"Intent detection error: {str(e)}")
        return "UNCLEAR"

async def generate_clarification(user_input: str, language: str) -> str:
    print("Function: generate_clarification")
    """Generate clarification message when intent is unclear"""
    clarification_prompt = f"""The user said: "{user_input}"

Generate a friendly clarification message in {language} asking whether they want:
1. Medical diagnosis/health questions
2. Appointment booking with doctors

Keep it conversational and helpful."""
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(clarification_prompt)
        ])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(input="")
        return response.strip()
    except Exception as e:
        logger.error(f"Clarification generation error: {str(e)}")
        return "I'd be happy to help! Could you let me know if you need help with medical diagnosis questions or booking an appointment with a doctor?"

def get_current_flow(conversation_history: List[Dict[str, str]]) -> str:
    print("Function: get_current_flow")
    """Extract current flow from conversation history"""
    # Get the most recent flow marker
    for msg in reversed(conversation_history):
        if msg["role"] == "system" and "selected_flow" in msg.get("content", ""):
            return msg["content"].split(":")[1].strip()
    return None

def count_questions_asked(conversation_history: List[Dict[str, str]]) -> int:
    print("Function: count_questions_asked")
    """Count how many medical questions have been asked"""
    question_count = 0
    for msg in conversation_history:
        if msg["role"] == "assistant":
            content = msg.get("content", "").lower()
            # Check if message contains multiple choice options (A., B., C., D.)
            if "a." in content and "b." in content and "c." in content and "d." in content:
                question_count += 1
    return question_count

def get_conversation_context(conversation_history: List[Dict[str, str]]) -> str:
    print("Function: get_conversation_context")
    """Get conversation context for better intent detection"""
    current_flow = get_current_flow(conversation_history)
    question_count = count_questions_asked(conversation_history)
    
    context = f"Current flow: {current_flow or 'none'}, Questions asked: {question_count}"
    
    # Add recent conversation context
    recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
    recent_context = " | ".join([
        f"{msg['role']}: {msg['content'][:50]}..." 
        for msg in recent_messages if msg['role'] != 'system'
    ])
    
    return f"{context} | Recent: {recent_context}"

def update_flow_marker(conversation_history: List[Dict[str, str]], new_flow: str):
    print("Function: update_flow_marker")
    """Update flow marker in conversation history"""
    # Remove old flow markers
    conversation_history[:] = [
        msg for msg in conversation_history 
        if not (msg["role"] == "system" and "selected_flow" in msg.get("content", ""))
    ]
    # Add new flow marker
    conversation_history.append({
        "role": "system", 
        "content": f"selected_flow: {new_flow}"
    })

async def handle_diagnosis_flow(request: ChatRequest, question_count: int = None):
    print("Function: handle_diagnosis_flow")
    """Handle diagnosis flow with question counting and transition logic"""
    try:
        if question_count is None:
            question_count = count_questions_asked(request.conversation_history)
        
        conv_history = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in request.conversation_history if msg['role'] != "system"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                MEDICAL_PROMPT.format(
                    conversation_history=conv_history,
                    language=request.language,
                    question_count=question_count
                )
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(user_input=request.user_input)
        # print(response)
        # If we've asked 5 questions and system suggests appointment, prepare for potential flow switch
        if question_count >= 5 and ("appointment" in response.lower() or "book" in response.lower()):
            # Add a subtle hint that flow switching is possible without forcing it
            response += "\n\nYou can also type 'yes' or 'book appointment' if you'd like to schedule a consultation now."
        
        return {"response": response.strip()}
        
    except Exception as e:
        logger.error(f"Diagnosis flow error: {str(e)}")
        raise HTTPException(500, "Diagnosis processing failed")

async def handle_smart_appointment_flow(request: ChatRequest):
    """Enhanced appointment flow with location-based search"""
    print("Function: handle_smart_appointment_flow - Enhanced Version")
    
    try:
        # Get current appointment state
        state = get_appointment_state(request.conversation_history)
        
        print(f"Current appointment state: {state}")
        
        # Handle different steps of appointment booking
        if state["step"] == "needs_city":
            return await collect_location_info(request)
        
        elif state["step"] == "needs_preference":
            return await collect_location_info(request)
        
        elif state["step"] == "ready_to_search":
            return await search_and_display_doctors(request)
        
        else:  # start step
            # Initial appointment flow - ask for city first
            return await collect_location_info(request)
            
    except Exception as e:
        logger.error(f"Enhanced appointment flow error: {str(e)}")
        raise HTTPException(500, "Enhanced appointment processing failed")
import re

async def location_based_doctor_search(city: str = None, department: str = None, doctor_name: str = None) -> list:
    """Enhanced doctor search with location and multiple criteria"""
    print("Function: location_based_doctor_search")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Build dynamic query based on available parameters
        query_parts = []
        params = []
        
        if city:
            query_parts.append("LOWER(Location) = LOWER(%s)")
            params.append(city)
        
        if doctor_name:
            query_parts.append("LOWER(name) LIKE %s")
            params.append(f"%{doctor_name.lower()}%")
        elif department:
            query_parts.append("LOWER(department) LIKE %s")
            params.append(f"%{department.lower()}%")
        
        # Construct final query
        base_query = "SELECT * FROM doctors"
        if query_parts:
            base_query += " WHERE " + " AND ".join(query_parts)
        base_query += " ORDER BY department, name"
        
        print(f"Query: {base_query}")
        print(f"Params: {params}")
        
        cursor.execute(base_query, params)
        doctors_list = cursor.fetchall() or []
        
        cursor.close()
        conn.close()
        
        return doctors_list
        
    except Exception as e:
        logger.error(f"Location-based doctor search error: {str(e)}")
        return []



def get_appointment_state(conversation_history: List[Dict[str, str]]) -> dict:
    """Extract appointment booking state from conversation history"""
    print("Function: get_appointment_state")
    
    state = {
        "city": None,
        "department": None,
        "doctor_name": None,
        "step": "start"  # start, needs_city, needs_preference, ready_to_search, showing_results
    }
    
    # Parse conversation for collected information
    for msg in conversation_history:
        content = msg.get("content", "").lower()
        
        # Check for city mentions
        cities = ["kanpur", "orai", "jhansi"]
        for city in cities:
            if city in content:
                state["city"] = city.title()
                break
        
        # Check for department mentions
        departments = ["cardiology", "cardiologist", "pediatric", "pediatrician", 
                      "orthopedic", "gynecologist", "dermatologist", "ent", 
                      "neurologist", "psychiatrist", "dentist", "general physician"]
        for dept in departments:
            if dept in content:
                state["department"] = dept
                break
        
        # Check for doctor name mentions (Dr. followed by name)
        if "dr." in content or "doctor" in content:
            import re
            name_match = re.search(r'dr\.?\s+([a-zA-Z\s]+)', content, re.IGNORECASE)
            if name_match:
                state["doctor_name"] = name_match.group(1).strip()
    
    # Determine current step
    if not state["city"]:
        state["step"] = "needs_city"
    elif not state["department"] and not state["doctor_name"]:
        state["step"] = "needs_preference"  
    else:
        state["step"] = "ready_to_search"
    
    return state

def extract_user_preferences(user_input: str) -> dict:
    """Extract city, department, or doctor name from user input"""
    print("Function: extract_user_preferences")
    
    user_input_lower = user_input.lower()
    preferences = {"city": None, "department": None, "doctor_name": None}
    
    # Extract city
    cities = ["kanpur", "orai", "jhansi"]
    for city in cities:
        if city in user_input_lower:
            preferences["city"] = city.title()
            break
    
    # Extract department
    dept_mapping = {
        "heart": "Cardiologist", "cardio": "Cardiologist", "cardiologist": "Cardiologist",
        "child": "Pediatrician", "pediatric": "Pediatrician", "pediatrician": "Pediatrician",
        "bone": "Orthopedic", "orthopedic": "Orthopedic", "ortho": "Orthopedic",
        "skin": "Dermatologist", "dermatologist": "Dermatologist",
        "ear": "ENT Specialist", "nose": "ENT Specialist", "throat": "ENT Specialist", "ent": "ENT Specialist",
        "brain": "Neurologist", "neurologist": "Neurologist", "neuro": "Neurologist",
        "mental": "Psychiatrist", "psychiatrist": "Psychiatrist", "psychology": "Psychiatrist",
        "teeth": "Dentist", "dental": "Dentist", "dentist": "Dentist",
        "general": "General Physician", "physician": "General Physician", "family": "General Physician",
        "women": "Gynecologist", "gynecologist": "Gynecologist", "gyno": "Gynecologist"
    }
    
    for keyword, department in dept_mapping.items():
        if keyword in user_input_lower:
            preferences["department"] = department
            break
    
    # Extract doctor name
    import re
    name_patterns = [
        r'dr\.?\s*([a-zA-Z\s]+)',
        r'doctor\s+([a-zA-Z\s]+)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            preferences["doctor_name"] = match.group(1).strip()
            break
    
    return preferences

# ===== ENHANCED APPOINTMENT FLOW FUNCTIONS =====

async def collect_location_info(request: ChatRequest):
    """Collect city and preference information step by step"""
    print("Function: collect_location_info")
    
    try:
        # Get current state
        state = get_appointment_state(request.conversation_history)
        
        # Extract any new preferences from user input
        user_prefs = extract_user_preferences(request.user_input)
        
        # Update state with new information
        if user_prefs["city"]:
            state["city"] = user_prefs["city"]
        if user_prefs["department"]:
            state["department"] = user_prefs["department"]
        if user_prefs["doctor_name"]:
            state["doctor_name"] = user_prefs["doctor_name"]
        
        # Determine what information is still needed
        city_status = "✅ Collected" if state["city"] else "❌ Missing"
        preference_status = "✅ Collected" if (state["department"] or state["doctor_name"]) else "❌ Missing"
        
        # Generate appropriate response
        conv_history = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in request.conversation_history if msg['role'] != "system"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                LOCATION_COLLECTION_PROMPT.format(
                    conversation_history=conv_history,
                    user_input=request.user_input,
                    language=request.language,
                    city_status=city_status,
                    preference_status=preference_status
                )
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(user_input=request.user_input)
        
        return {"response": response.strip()}
        
    except Exception as e:
        logger.error(f"Location collection error: {str(e)}")
        raise HTTPException(500, "Location collection failed")

async def search_and_display_doctors(request: ChatRequest):
    """Search doctors based on collected criteria and display results"""
    print("Function: search_and_display_doctors")
    
    try:
        # Get current state
        state = get_appointment_state(request.conversation_history)
        
        # Extract any additional preferences from current input
        user_prefs = extract_user_preferences(request.user_input)
        if user_prefs["city"]:
            state["city"] = user_prefs["city"]
        if user_prefs["department"]:
            state["department"] = user_prefs["department"]
        if user_prefs["doctor_name"]:
            state["doctor_name"] = user_prefs["doctor_name"]
        
        # Perform doctor search
        doctors_list = await location_based_doctor_search(
            city=state["city"],
            department=state["department"],
            doctor_name=state["doctor_name"]
        )
        
        # Format search criteria for display
        search_criteria = []
        if state["doctor_name"]:
            search_criteria.append(f"Doctor: {state['doctor_name']}")
        elif state["department"]:
            search_criteria.append(f"Department: {state['department']}")
        
        search_criteria_text = " & ".join(search_criteria)
        
        # Format doctors info for LLM
        if doctors_list:
            doctors_text = f"FOUND {len(doctors_list)} DOCTORS:\n" + "\n".join([
                f"🏥 Dr. {doc['name']} - {doc['department']}\n   📍 {doc['Location']}\n   🕒 {doc.get('timings', 'Contact for timings')}"
                for doc in doctors_list
            ])
        else:
            doctors_text = "No doctors found matching the criteria."
        
        # Generate response
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                ENHANCED_DOCTOR_DISPLAY_PROMPT.format(
                    search_criteria=search_criteria_text,
                    city=state["city"] or "your location",
                    doctors_info=doctors_text,
                    language=request.language
                )
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(user_input=request.user_input)
        
        return {"response": response.strip()}
        
    except Exception as e:
        logger.error(f"Doctor search and display error: {str(e)}")
        raise HTTPException(500, "Doctor search failed")
    







    





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
        # Extract name, gender, and age from the request
        name = request.name
        gender = request.gender
        age = request.age
        language = request.language  # New field for language support
        
        # Extract chief complaint
        chief_complaint = next(
            (msg["content"] for msg in request.conversation_history 
             if msg["role"] == "user"),
            "Not specified"
        )
        
        # Build conversation history
        conv_history = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in request.conversation_history
        )

        # Prepare LangChain LLM call
        prompt = PromptTemplate(
            input_variables=["chief_complaint", "history", "conversation_history","language"],
            template=REPORT_PROMPT
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        report_text = await chain.arun(
            chief_complaint=chief_complaint,
            history="From conversation",
            conversation_history=conv_history,
            language=language  # Include language in the prompt
        )

        # PDF Generation with styling
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        # Styles
        base_styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'Heading',
            parent=base_styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=12,
            leftIndent=0,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        body_style = ParagraphStyle(
            'Body',
            parent=base_styles['Normal'],
            fontSize=11,
            leading=16,
            leftIndent=20
        )

        story = []

        # Title
        story.append(Paragraph("Medical Consultation Report", heading_style))
        story.append(Spacer(1, 12))

        # Patient Details Section
        story.append(Paragraph("Patient Details", heading_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Name: {name}", body_style))
        story.append(Paragraph(f"Gender: {gender}", body_style))
        story.append(Paragraph(f"Age: {age}", body_style))
        story.append(Spacer(1, 12))

        # Split report into sections and format
        for paragraph in report_text.split('\n\n'):
            stripped = paragraph.strip()
            if not stripped:
                continue
            if stripped.endswith(":"):  # Assume it's a heading
                story.append(Spacer(1, 10))
                story.append(Paragraph(stripped, heading_style))
            else:
                story.append(Paragraph(stripped, body_style))
            story.append(Spacer(1, 6))

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

# Offline Report Generation
@app.post("/generate_offline_report")
async def generate_offline_report(request: OfflineReportRequest):
    try:
        prompt = PromptTemplate(
            input_variables=["name", "age", "gender", "department", "language", "responses"],
            template=OFFLINE_REPORT_PROMPT
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        report_content = await chain.arun(
            name=request.name,
            age=request.age,
            gender=request.gender,
            department=request.department,
            language=request.language,  # Include language in the prompt
            responses=request.responses,
        )

        report = {
            "Patient Details": {
                "Name": request.name,
                "Age": request.age,
                "Gender": request.gender,
                "Department": request.department,
                "Language": request.language,  # Include language in the JSON response
            },
            "Report": report_content,
            "Remarks": "This is an auto-generated offline medical  with language consideration.",
        }

        return JSONResponse(
            content=report,
            headers={"Content-Disposition": "attachment; filename=offline_report.json"}
        )
    except Exception as e:
        logger.error(f"Error in /generate_offline_report endpoint: {str(e)}")
        raise HTTPException(500, "Offline report generation failed")
