from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model import model
import logging
import sys
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

async def generate_questions(context, num_questions=5):
    """Generate questions directly using phi3 model with internal validation"""
    system_prompt = """Given a context, generate thought-provoking questions. Each question should:
    1. Be clear and concise
    2. Test understanding of key concepts
    3. Follow format: "Q: [question]"
    4. Focus on important information
    5. Be answerable from the context
    
    Example format:
    Q: What is the main challenge in implementing X?
    Q: How does Y affect the overall system?
    """
    
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nGenerate {num_questions} questions:\n"
    
    try:
        # Generate questions in a single pass
        response = await generate_with_phi3(prompt)
        questions = parse_questions(response)
        
        # Validate and filter questions
        valid_questions = []
        for q in questions:
            if validate_question(q, context):
                valid_questions.append(q)
            if len(valid_questions) >= num_questions:
                break
                
        # If we don't have enough valid questions, generate more
        if len(valid_questions) < num_questions:
            additional_needed = num_questions - len(valid_questions)
            additional_prompt = f"{prompt}\nGenerate {additional_needed} more questions, different from:\n"
            for q in valid_questions:
                additional_prompt += f"{q}\n"
            
            additional_response = await generate_with_phi3(additional_prompt)
            additional_questions = parse_questions(additional_response)
            
            for q in additional_questions:
                if validate_question(q, context):
                    valid_questions.append(q)
                if len(valid_questions) >= num_questions:
                    break
        
        return valid_questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return []

def validate_question(question, context):
    """Validate a single question against quality criteria"""
    # Remove 'Q: ' prefix if present
    q_text = question[3:] if question.startswith('Q: ') else question
    
    # Basic validation checks
    if len(q_text) < 10 or len(q_text) > 200:  # Length check
        return False
    
    if not q_text.endswith('?'):  # Must end with question mark
        return False
        
    # Check if question contains key terms from context
    context_words = set(re.findall(r'\w+', context.lower()))
    question_words = set(re.findall(r'\w+', q_text.lower()))
    
    # Question should contain at least some context keywords
    if len(question_words.intersection(context_words)) < 2:
        return False
        
    # Avoid overly simple questions
    simple_starts = ['what is', 'who is', 'where is']
    if any(q_text.lower().startswith(start) for start in simple_starts):
        return len(q_text) > 30  # Allow if question is more detailed
        
    return True

def parse_questions(response):
    """Parse questions from model response"""
    questions = []
    
    # Split by newline and look for question format
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Q:') or line.startswith('Question:'):
            # Standardize format to 'Q: '
            q = line.replace('Question:', 'Q:').strip()
            if not q.startswith('Q: '):
                q = 'Q: ' + q[2:].strip()
            questions.append(q)
            
    return questions

async def generate_with_phi3(prompt):
    """Generate text using the phi3 model"""
    try:
        from .model import get_model_config
        
        # Get optimized configuration
        system_info = {
            'is_cloud': False  # Set based on your environment
        }
        config = get_model_config(system_info)
        
        # Initialize model with optimized settings
        model_name = "microsoft/phi-2"  # You can update this to phi-3 when available
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config['cuda'] else torch.float32,
            device_map="auto" if config['cuda'] else None
        )

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config['ctx_size'])
        if config['cuda']:
            inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=config['num_predict'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error in phi3 generation: {str(e)}")
        raise