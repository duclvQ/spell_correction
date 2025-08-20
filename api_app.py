from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from mis_detection import BertSpellChecker
from pyvi import ViTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

class SpellCheckerAPI:
    def __init__(self):
        """Initialize the spell checker API"""
        logger.info("Initializing Spell Checker API...")
        self.spell_checker = BertSpellChecker()
    
    def check_spelling(self, text):
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
            
            # Split text into sentences
            sentences = text.split('. ')
            # sentences = [text]
            results = []
            all_errors = []
            corrected_sentences = []
            
            for sentence_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Get spell check results for this sentence
                corrected_sentence, errors = self.spell_checker(sentence)
                
                if errors:
                    # Process errors and create corrected version
                    sentence_errors = []
                    for word_idx, original_word, suggestion in errors:
                        sentence_errors.append({
                            "word_index": word_idx,
                            "original_word": original_word,
                            "suggestion": suggestion,
                            "position_in_sentence": word_idx
                        })
                    
                    results.append({
                        "sentence_index": sentence_idx,
                        "original_sentence": sentence,
                        "corrected_sentence": corrected_sentence,
                        "errors": sentence_errors,
                        "has_errors": True
                    })
                    
                    all_errors.extend(sentence_errors)
                    corrected_sentences.append(corrected_sentence)
                else:
                    # No errors in this sentence
                    results.append({
                        "sentence_index": sentence_idx,
                        "original_sentence": sentence,
                        "corrected_sentence": sentence,
                        "errors": [],
                        "has_errors": False
                    })
                    corrected_sentences.append(sentence)
            
            # Create full corrected text
            corrected_text = '. '.join(corrected_sentences)
            
            return {
                "status": "success",
                "original_text": text,
                "corrected_text": corrected_text,
                "total_errors": len(all_errors),
                "sentences": results,
                "summary": {
                    "total_sentences": len(results),
                    "sentences_with_errors": len([r for r in results if r["has_errors"]]),
                    "total_words_corrected": len(all_errors)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing text: {str(e)}"
            }

# Initialize the spell checker
spell_checker = SpellCheckerAPI()

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def api_check_spelling():
    """API endpoint for spell checking"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'text' parameter in request body"
            }), 400
        
        input_text = data['text']
        result = spell_checker.check_spelling(input_text)
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Vietnamese Spell Checker API",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    logger.info("Starting Vietnamese Spell Checker API...")
    app.run(debug=True, host='0.0.0.0', port=5001)
