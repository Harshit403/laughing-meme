from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import re
import time
from typing import List, Dict, Optional
from vector_store import OptimizedVectorStore, generate_answer, run_async_task
import uvicorn
from vector_store import EMBEDDING_DIMENSION   # already == 1024
vector_store = OptimizedVectorStore(EMBEDDING_DIMENSION)

# Initialize FastAPI app
app = FastAPI(title="Live Chat Widget API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create chats directory if it doesn't exist
os.makedirs("chats", exist_ok=True)

# Initialize vector store
#vector_store = OptimizedVectorStore(384)  # EMBEDDING_DIMENSION

# Load documents from document.txt
def load_documents():
    try:
        with open("document.txt", "r", encoding="utf-8") as f:
            content = f.read()
            # Split content into documents by double newlines
            documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]
            if documents:
                vector_store.add_documents(documents)
                print(f"Loaded {len(documents)} documents into vector store")
            else:
                print("No documents found in document.txt")
    except FileNotFoundError:
        print("document.txt not found. Vector store will be empty.")
    except Exception as e:
        import traceback
        traceback.print_exc()      # full stack-trace to console

    #except Exception as e:
   #     print(f"Error loading documents: {e}")

# Load documents at startup
load_documents()

# Pydantic models for request/response
class UserInfo(BaseModel):
    name: str
    email: str
    mobile: str

class ChatMessage(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class ChatStartResponse(BaseModel):
    session_id: str
    message: str

# Helper functions for file operations
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_chat_file_path(session_id: str) -> str:
    """Get the file path for a chat session"""
    sanitized_name = sanitize_filename(session_id)
    return os.path.join("chats", f"{sanitized_name}.txt")

def save_user_info(session_id: str, user_info: UserInfo):
    """Save user information to chat file"""
    file_path = get_chat_file_path(session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("USER INFORMATION:\n")
        f.write(f"Name: {user_info.name}\n")
        f.write(f"Email: {user_info.email}\n")
        f.write(f"Mobile: {user_info.mobile}\n\n")
        f.write("CHAT HISTORY:\n")

def save_message(session_id: str, role: str, message: str):
    """Save a message to the chat file"""
    file_path = get_chat_file_path(session_id)
    with open(file_path, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {role}: {message}\n")

def load_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Load chat history from file"""
    file_path = get_chat_file_path(session_id)
    chat_history = []
    
    if not os.path.exists(file_path):
        return chat_history
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Skip user information section
    start_chat = False
    for line in lines:
        if line.strip() == "CHAT HISTORY:":
            start_chat = True
            continue
        
        if start_chat and line.strip():
            # Parse message line: [timestamp] role: message
            match = re.match(r'\[([^\]]+)\] (\w+): (.+)', line.strip())
            if match:
                timestamp, role, message = match.groups()
                chat_history.append({
                    "role": role,
                    "content": message
                })
    
    return chat_history

# API endpoints
@app.get("/")
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_chat", response_model=ChatStartResponse)
async def start_chat(user_info: UserInfo):
    """Start a new chat session with user information"""
    try:
        # Save user information
        save_user_info(user_info.name, user_info)
        
        # Return session ID (user name)
        return ChatStartResponse(
            session_id=user_info.name,
            message="Chat session started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")

@app.post("/send_message", response_model=ChatResponse)
async def send_message(chat_message: ChatMessage):
    """Send a message and get a response from the agent"""
    try:
        # Load chat history
        chat_history = load_chat_history(chat_message.session_id)
        
        # Save user message
        save_message(chat_message.session_id, "User", chat_message.message)
        
        # Generate agent response
        response = await generate_answer(
            query=chat_message.message,
            vector_store=vector_store,
            chat_history=chat_history
        )
        
        # Save agent response
        save_message(chat_message.session_id, "Agent", response)
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
