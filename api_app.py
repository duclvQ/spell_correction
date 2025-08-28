from _hashlib import new
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict
from misspell_detection import BertSpellChecker
from pyvi import ViTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vietnamese Spell Checker API",
    description="API for checking and correcting Vietnamese text spelling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class SpellCheckRequest(BaseModel):
    text: str

class ErrorDetail(BaseModel):
    op: str
    orig: str
    repl: str
    pos: tuple

class SpellCheckSummary(BaseModel):
    total_sentences: int
    sentences_with_errors: int
    total_words_corrected: int

class SpellCheckResponse(BaseModel):
    status: str
    original_text: str
    new_text: str
    replacement: List[ErrorDetail]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ErrorResponse(BaseModel):
    status: str
    message: str

class SpellCheckerAPI:
    def __init__(self):
        """Initialize the spell checker API"""
        logger.info("Initializing Spell Checker API...")
        self.spell_checker = BertSpellChecker()
    
    def check_spelling(self, text: str) -> Dict:
        """
        Check spelling in the given text
        
        Args:
            text (str): Input text to check for misspellings
            
        Returns:
            dict: Results containing original text, errors, and corrected version
        """
        try:
            if not text or not text.strip():
                return {
                    "status": "error",
                    "message": "Input text is empty"
                }
            
            logger.info(f"Processing text: {text[:100]}...")
         
            errors, corrected_sentence = self.spell_checker(text)
            results = {
                "status": "success",
                "original_text": text,
                "new_text": corrected_sentence,
                "replacement": errors
            }
            print("results", results)
            return results

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing text: {str(e)}"
            }

# Initialize the spell checker
spell_checker = SpellCheckerAPI()

# FastAPI endpoints
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vietnamese Spell Checker API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            textarea { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 4px; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vietnamese Spell Checker API</h1>
            <p>Enter Vietnamese text to check for spelling errors:</p>
            
            <textarea id="inputText" rows="6" placeholder="Enter your Vietnamese text here...">
Tổng bí thư Tôn Lâm và thủ tướng nguyễn Minh Chính có bài phát biểu tại hội nghị chuyển đổi số quốc gia của Việt Nam
            </textarea>
            
            <button onclick="checkSpelling()">Check Spelling</button>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
        async function checkSpelling() {
            const inputText = document.getElementById('inputText').value.trim();
            const resultDiv = document.getElementById('result');
            
            if (!inputText) {
                resultDiv.innerHTML = '<p class="error">Please enter some text to check.</p>';
                resultDiv.style.display = 'block';
                return;
            }
            
            try {
                const response = await fetch('/api/check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: inputText
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    let html = '<h3>Results:</h3>';
                    html += `<p><strong>Original:</strong> ${result.original_text}</p>`;
                    html += `<p><strong>Corrected:</strong> ${result.corrected_text}</p>`;
                    html += `<p><strong>Total errors found:</strong> ${result.total_errors}</p>`;
                    
                    if (result.sentences && result.sentences.length > 0) {
                        html += '<h4>Details:</h4>';
                        result.sentences.forEach((sentence, index) => {
                            if (sentence.has_errors) {
                                html += `<p><strong>Sentence ${index + 1}:</strong> ${sentence.errors.length} error(s)</p>`;
                                sentence.errors.forEach(error => {
                                    html += `<p style="margin-left: 20px;">• "${error.original_word}" → "${error.suggestion}"</p>`;
                                });
                            }
                        });
                    }
                    
                    resultDiv.innerHTML = html;
                    resultDiv.className = 'result success';
                } else {
                    resultDiv.innerHTML = `<p class="error">Error: ${result.message}</p>`;
                    resultDiv.className = 'result error';
                }
                
                resultDiv.style.display = 'block';
                
            } catch (error) {
                resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/check", response_model=SpellCheckResponse)
async def api_check_spelling(request: SpellCheckRequest):
    """API endpoint for spell checking"""
    try:
        result = spell_checker.check_spelling(request.text)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return result
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Vietnamese Spell Checker API",
        "version": "1.0.0"
    }

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the spell checker on startup"""
    logger.info("Starting Vietnamese Spell Checker API...")

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting Vietnamese Spell Checker API with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
