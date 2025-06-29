# Phi-3 Question Generator API

A fast and efficient multiple-choice question generator using the Phi-3 language model, optimized for performance and reliability.

## Features

- Generates high-quality multiple-choice questions (MCQs)
- Optimized for both cloud and local environments
- CUDA support for faster generation
- Automatic validation and quality control
- Configurable difficulty levels
- No external API dependencies

## Prerequisites

- Python 3.12+
- Ollama installed and running locally
- Required Python packages (install via `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd phi3_model
```

2. Create and activate a virtual environment:
```bash
python -m venv myvenv
./myvenv/Scripts/Activate.ps1  # For PowerShell
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is installed and running with the phi3 model:
```bash
ollama run phi3:3.8b
```

## API Usage

### Generate Questions

Endpoint: `/generate`
Method: `POST`

Request Body:
```json
{
    "topic": "string",           // The topic to generate questions about
    "proficiency_level": "string" // Difficulty level (e.g., "beginner", "intermediate", "advanced")
}
```

Example Response:
```json
{
    "questions": [
        {
            "question": "What is the capital of France?",
            "options": [
                "London",
                "Berlin",
                "Madrid",
                "Paris"
            ],
            "correct_answer": "Paris"
        },
        // ... more questions
    ]
}
```

### Question Format

Each generated question follows a strict format:
1. Question text starting with "Q1:", "Q2:", etc.
2. Exactly 4 options labeled as "A)", "B)", "C)", "D)"
3. One correct answer marked with (✓)
4. Each option on a new line

### Configuration Options

Model configuration can be customized in `model.py`:

```python
{
    "num_predict": 1024,     # Token prediction limit
    "temperature": 0.8,      # Control randomness (0.0-1.0)
    "top_p": 0.9,           # Nucleus sampling parameter
    "cuda": true,           # Enable GPU acceleration
    "num_thread": 6,        # Number of CPU threads
    "batch_size": 8,        # Batch size for generation
    "gpu_layers": 24,       # Number of GPU layers
    "ctx_size": 1024       # Context window size
}
```

## Performance Optimization

The system automatically detects and optimizes for:
- CUDA availability for GPU acceleration
- Cloud vs local environment
- Available CPU cores
- Memory constraints

## Error Handling

The API includes:
- Automatic retry mechanism (up to 3 attempts)
- Input validation
- Question format verification
- Duplicate question detection
- Service status monitoring

## Logging

Comprehensive logging is available with:
- Generation timing
- Success/failure status
- Question validation results
- System configuration details

## Environment Variables

Optional environment variables:
- `AWS_LAMBDA_FUNCTION_NAME`: Detected for cloud deployment
- Custom configuration can be added through environment variables

## Development

To run in development mode:
```bash
uvicorn main:app --reload
```

To run tests:
```bash
pytest tests/
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
