from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model import model
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuestionRequest(BaseModel):
    topic: str
    difficulty: str = "medium"  # default difficulty

class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

@app.post("/generate-questions/", response_model=List[Question])
async def generate_questions(request: QuestionRequest):
    logger.info(f"Received request for topic: {request.topic}, difficulty: {request.difficulty}")
    try:
        logger.info("Calling model to generate questions...")
        questions = await model.generate_questions(request.topic, request.difficulty)
        logger.info(f"Generated {len(questions)} valid questions")
        
        if not questions:
            logger.error("No valid questions were generated")
            raise HTTPException(
                status_code=500,
                detail="No valid questions were generated. Please try again."
            )
        return questions
    except RuntimeError as e:
        logger.error(f"RuntimeError: {str(e)}")
        if "Ollama service not available" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not available. Please ensure Ollama is running by executing 'ollama serve' in a separate terminal."
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/health")
def model_health():
    try:
        ollama_status = model.check_ollama_status()
        return {
            "status": "healthy", 
            "model_loaded": True,
            "ollama_status": ollama_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }