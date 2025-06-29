import ollama
import os
import platform
import time
import logging
import sys
import socket
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# === System Configuration ===
def get_system_config():
    is_lambda = bool(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    system_info = {
        'is_cloud': is_lambda,
        'cpu_count': os.cpu_count() or 1,
        'platform': platform.system(),
        'cuda_available': True
    }
    return system_info

# === Model Configuration ===
def get_model_config(system_info):
    if system_info['is_cloud']:
        return {
            "num_predict": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "cuda": False,
            "num_thread": 2
        }
    else:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        return {
            "num_predict": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "cuda": cuda_available,
            "num_thread": 8 if cuda_available else 4,
            "batch_size": 16 if cuda_available else 8,
            "gpu_layers": 32 if cuda_available else 0,
            "mmap": True,
            "seed": 42,
            "ctx_size": 2048,
            "f16_kv": cuda_available
        }

def check_ollama_status():
    """Check if Ollama service is running and responsive"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()
        
        if result != 0:
            return {"status": "not_running", "error": "Ollama service is not running"}
            
        models = ollama.list()
        return {"status": "running", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}

GEMINI_RATE_LIMIT = 10  # requests per minute
GEMINI_RATE_WINDOW = 60  # seconds
_last_gemini_calls = []  # Track timestamps of API calls

async def _wait_for_rate_limit():
    """Wait if we're approaching the rate limit"""
    now = time.time()
    # Remove calls older than our window
    while _last_gemini_calls and _last_gemini_calls[0] < now - GEMINI_RATE_WINDOW:
        _last_gemini_calls.pop(0)
    
    if len(_last_gemini_calls) >= GEMINI_RATE_LIMIT:
        # Calculate how long to wait
        wait_time = _last_gemini_calls[0] + GEMINI_RATE_WINDOW - now + 1  # Add 1s buffer
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
    
    _last_gemini_calls.append(now)

async def verify_answer_with_gemini(question, options, proficiency_level):
    """Verify the correct answer using Gemini API"""
    try:
        # Configure model
        model = genai.GenerativeModel('gemini-pro')  # Use standard model as flash has stricter limits
        
        # Safety timeout and retries
        max_retries = 3
        retry_delay = 2  # Base delay for exponential backoff
        
        prompt = f"""Respond with ONLY a single letter A, B, C, or D representing the most correct answer.

Question: {question}
Level: {proficiency_level}

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""

        # Implement retry logic with rate limiting
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making call
                await _wait_for_rate_limit()
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt, 
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent answers
                        candidate_count=1,
                        max_output_tokens=1
                    )
                )
                
                # Check if generation succeeded
                if response and response.text:
                    answer = response.text.strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        index = ord(answer) - ord('A')
                        return options[index]
                    else:
                        logger.warning(f"Invalid Gemini response format on attempt {attempt + 1}: {answer}")
                else:
                    logger.warning(f"Empty Gemini response on attempt {attempt + 1}")
                    
                # Wait before retrying with exponential backoff
                if attempt < max_retries - 1:
                    retry_wait = retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(retry_wait)
                
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    # Get retry delay from error message if available
                    retry_seconds = 60  # Default to 60 seconds if we can't parse the delay
                    try:
                        import re
                        retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)\s*}', str(e))
                        if retry_match:
                            retry_seconds = int(retry_match.group(1))
                    except Exception:
                        pass
                    
                    logger.warning(f"Rate limit hit, waiting {retry_seconds} seconds before retry")
                    await asyncio.sleep(retry_seconds)
                else:
                    logger.warning(f"Gemini API error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_wait = retry_delay * (2 ** attempt)
                        await asyncio.sleep(retry_wait)
                continue
                    
        # If we get here, all retries failed
        logger.error("All Gemini verification attempts failed")
        return None
            
    except Exception as e:
        logger.error(f"Critical error in Gemini verification: {str(e)}")
        return None

async def generate_questions(topic, proficiency_level):
    """Generate questions asynchronously with proper error handling"""
    logger.info(f"Generating questions for topic: {topic}, level: {proficiency_level}")
    
    try:
        # Check Ollama status first
        status = check_ollama_status()
        if status["status"] != "running":
            error_msg = f"Ollama service is not ready: {status['error']}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        all_valid_questions = []
        max_attempts = 3
        gemini_cooldown = 1  # Cooldown between Gemini API calls in seconds
        
        while len(all_valid_questions) < 10 and max_attempts > 0:
            try:
                remaining = 10 - len(all_valid_questions)
                attempt_target = min(remaining + 2, 5)  # Generate a few extra to account for invalid ones
                logger.info(f"Attempting to generate {attempt_target} questions. Attempts remaining: {max_attempts}")
                
                # Format the prompt
                prompt = f"""
                You are an expert exam creator. Follow the format EXACTLY.

                Task:
                Generate exactly {attempt_target} multiple-choice questions (MCQs) on the topic: "{topic}"
                Proficiency level: {proficiency_level}

                REQUIREMENTS:
                1. Generate high-quality, challenging questions
                2. Each question must have one clear, unambiguous correct answer
                3. All options must be plausible but only one should be correct
                4. Keep questions concise and direct

                STRICT FORMAT:
                1. Each question MUST start with "Q1:", "Q2:", etc.
                2. Each question MUST have EXACTLY 4 options labeled as "A)", "B)", "C)", "D)"
                3. Each option must be on a new line
                4. No correct answers or explanations - just questions and options

                Example format (follow this EXACTLY):
                Q1: What is the capital of France?
                A) London
                B) Berlin
                C) Madrid
                D) Paris
                """

                system_config = get_system_config()
                model_options = get_model_config(system_config)
                start_time = time.time()

                # Generate questions using Ollama - make it async
                response = await asyncio.to_thread(
                    ollama.chat,
                    model="gemma3:1b",
                    messages=[{"role": "user", "content": prompt}],
                    options=model_options
                )
                
                generation_time = time.time() - start_time
                logger.info(f"Initial generation took {generation_time:.2f} seconds")
                
                raw_result = response['message']['content']
                logger.info("Successfully generated response from model")
                
                # Parse questions
                current_question = None
                questions_batch = []
                
                for line in raw_result.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('Q'):
                        if current_question and current_question['options'] and len(current_question['options']) == 4:
                            questions_batch.append(current_question)
                        
                        current_question = {
                            'question': line.split(': ', 1)[1] if ': ' in line else line,
                            'options': [],
                            'correct_answer': ''
                        }
                        
                    elif current_question and line[0] in 'ABCD' and line[1] == ')':
                        option_text = line[2:].strip()
                        if option_text:
                            current_question['options'].append(option_text)

                # Process the last question in the batch
                if current_question and current_question['options'] and len(current_question['options']) == 4:
                    questions_batch.append(current_question)

                # Verify answers with rate limiting
                for q in questions_batch:
                    if len(all_valid_questions) >= 10:
                        break
                        
                    correct_answer = await verify_answer_with_gemini(
                        q['question'],
                        q['options'],
                        proficiency_level
                    )
                    
                    if correct_answer:
                        q['correct_answer'] = correct_answer
                        all_valid_questions.append(q)
                        await asyncio.sleep(gemini_cooldown)  # Async rate limiting
                
            except Exception as e:
                logger.error(f"Error during generation attempt: {str(e)}")

            max_attempts -= 1
            if len(all_valid_questions) < 10 and max_attempts > 0:
                retry_delay = 2 * (3 - max_attempts)  # Exponential backoff
                logger.info(f"Waiting {retry_delay} seconds before next attempt...")
                await asyncio.sleep(retry_delay)  # Async sleep

        if len(all_valid_questions) < 10:
            logger.warning(f"Could only generate {len(all_valid_questions)} valid questions after all attempts")
        else:
            logger.info("Successfully generated all 10 questions")

        return all_valid_questions[:10]  # Return exactly 10 questions or all we have

    except Exception as e:
        logger.error(f"Critical error in generate_questions: {str(e)}")
        raise  # Re-raise to let FastAPI handle the error response
