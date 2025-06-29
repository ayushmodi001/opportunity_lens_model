import ollama
import os
import platform
import time
import logging
import sys
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        # AWS Lambda optimization settings
        return {
            "num_predict": 1024,  # Reduced for faster generation
            "temperature": 0.8,   # Slightly increased for better variation
            "top_p": 0.9,        # Reduced for more focused output
            "cuda": False,
            "num_thread": 2
        }
    else:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        # Optimized settings for faster generation
        return {
            "num_predict": 1024,  # Reduced from 2048
            "temperature": 0.8,   # Balanced for reliability
            "top_p": 0.9,        # More focused sampling
            "cuda": cuda_available,
            "num_thread": 6 if cuda_available else 4,  # Optimized thread count
            "batch_size": 8,      # Reduced for faster processing
            "gpu_layers": 24 if cuda_available else 0,  # Optimized GPU layer count
            "mmap": True,
            "seed": 42,
            "ctx_size": 1024,     # Reduced context window
            "f16_kv": cuda_available
        }

def check_ollama_status():
    """Check if Ollama service is running and responsive"""
    try:
        # Try to connect to Ollama's default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()
        
        if result != 0:
            return {"status": "not_running", "error": "Ollama service is not running"}
            
        # Check if we can list models
        models = ollama.list()
        return {"status": "running", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# === Prompt Template ===
async def generate_questions(topic, proficiency_level):
    logger.info(f"Generating questions for topic: {topic}, level: {proficiency_level}")
    
    # Check Ollama status first
    status = check_ollama_status()
    if status["status"] != "running":
        error_msg = f"Ollama service is not ready: {status['error']}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    all_valid_questions = []
    max_attempts = 3  # Maximum number of attempts to get all 10 questions
    
    while len(all_valid_questions) < 10 and max_attempts > 0:
        remaining = 10 - len(all_valid_questions)
        logger.info(f"Attempting to generate {remaining} more questions. Attempts remaining: {max_attempts}")
        
        # Modify prompt to request remaining number of questions
        prompt = f"""
            You are an expert exam creator. Follow the format EXACTLY.

            Task:
            Generate exactly {remaining} multiple-choice questions (MCQs) on the topic: "{topic}"
            Proficiency level: {proficiency_level}

            STRICT FORMAT REQUIREMENTS:
            1. Each question MUST start with "Q1:", "Q2:", etc.
            2. Each question MUST have EXACTLY 4 options labeled as "A)", "B)", "C)", "D)"
            3. EXACTLY ONE option must be marked with (✓) to indicate the correct answer
            4. Each option must be on a new line
            5. Do not add any extra text or explanations
            6. GENERATE EXACTLY {remaining} QUESTIONS, NO MORE, NO LESS

            Example format (follow this EXACTLY):
            Q1: What is the capital of France?
            A) London
            B) Berlin
            C) Madrid
            D) Paris (✓)

            Generate exactly {remaining} questions following this format precisely.
            DO NOT SKIP NUMBERS - start from Q1 and continue sequentially.
        """

        # === Initialize System and Model Configuration ===
        system_config = get_system_config()
        logger.info(f"System config: {system_config}")
        
        model_options = get_model_config(system_config)
        logger.info(f"Model options: {model_options}")

        # === Send to Ollama Model with Performance Monitoring and Retries ===
        start_time = time.time()
        retry_delay = 2  # seconds
        
        try:
            logger.info(f"Generating {remaining} questions...")
            response = ollama.chat(
                model="phi3:3.8b",
                messages=[{"role": "user", "content": prompt}],
                options=model_options
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            logger.info(f"Generation took {generation_time:.2f} seconds")
            
            raw_result = response['message']['content']
            logger.info("Successfully generated response from model")
            
            # Parse and validate questions
            questions = []
            current_question = None
            
            for line in raw_result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Handle question lines
                if line.startswith('Q'):
                    if current_question and current_question['options']:
                        questions.append(current_question)
                    current_question = {
                        'question': line.split(': ', 1)[1] if ': ' in line else line,
                        'options': [],
                        'correct_answer': ''
                    }
                    
                # Handle option lines
                elif current_question and line[0] in 'ABCD' and line[1] == ')':
                    option_text = line[2:].strip()
                    is_correct = '(✓)' in option_text
                    clean_option = option_text.replace('(✓)', '').strip()
                    
                    if clean_option:
                        current_question['options'].append(clean_option)
                        if is_correct:
                            current_question['correct_answer'] = clean_option

            # Add the last question if it's complete
            if current_question and current_question['options']:
                questions.append(current_question)

            # Validate and add to collection
            for q in questions:
                if len(q['options']) == 4 and q['correct_answer'] and len(all_valid_questions) < 10:
                    # Ensure no duplicate questions
                    is_duplicate = any(
                        existing['question'].lower() == q['question'].lower() 
                        for existing in all_valid_questions
                    )
                    if not is_duplicate:
                        all_valid_questions.append(q)

            logger.info(f"Valid questions so far: {len(all_valid_questions)}")
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
        
        max_attempts -= 1
        if len(all_valid_questions) < 10 and max_attempts > 0:
            logger.info(f"Waiting {retry_delay} seconds before next attempt...")
            time.sleep(retry_delay)
            retry_delay *= 2

    if len(all_valid_questions) < 10:
        logger.warning(f"Could only generate {len(all_valid_questions)} valid questions after all attempts")
    else:
        logger.info("Successfully generated all 10 questions")

    return all_valid_questions[:10]  # Return exactly 10 questions or all we have
