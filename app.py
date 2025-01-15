import os
import logging

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from langsmith import traceable

# If you're using a RAG pipeline, uncomment these:
from src.generation.rag_chain import RAGPipeline
from src.config_loader import load_config, set_environment_variables

app = Flask(
    __name__, 
    template_folder='frontend',  # where index.html is
    static_folder='frontend',
    static_url_path=''
)
CORS(app)

# Setup logging with more detailed output
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more visibility
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# If you have a RAG pipeline:
rag = None

def initialize_rag():
    global rag
    try:
        logger.debug("Starting RAG initialization...")
        config = load_config()
        set_environment_variables(config)
        rag = RAGPipeline(
            huggingface_token=config['huggingface']['token']
        )
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
        logger.exception("Detailed error:")  # This will log the full traceback
        raise

def list_local_preprocessed_docs(folder_path="data/preprocessed"):
    """
    Return a sorted list of filenames in data/preprocessed, 
    stripped of their .json extension (only .json files).
    """
    doc_names = []
    if not os.path.exists(folder_path):
        logger.warning(f"Folder path {folder_path} does not exist.")
        return doc_names

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            # remove trailing '.json' from filename
            doc_name = filename[:-5]  # e.g. "myfile.json" -> "myfile"
            doc_names.append(doc_name)
    doc_names.sort()
    return doc_names

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory."""
    logger.debug(f"Attempting to serve static file: {path}")
    try:
        return send_from_directory('frontend', path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}")
        return str(e), 500

@app.route('/')
def home():
    """
    Render the main chat interface (index.html),
    passing in the list of local doc names so the greeting
    can display them.
    """
    logger.debug("Handling request to home route '/'")
    try:
        # Check if frontend directory exists
        if not os.path.exists('frontend'):
            logger.error("Frontend directory not found")
            return "Frontend directory not found", 500
            
        # Check if index.html exists
        if not os.path.exists('frontend/index.html'):
            logger.error("index.html not found in frontend directory")
            return "index.html not found", 500

        doc_names = list_local_preprocessed_docs("data/preprocessed")
        logger.debug(f"Found doc_names: {doc_names}")
        return render_template('index.html', doc_names=doc_names)
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        logger.exception("Detailed error:")  # This will log the full traceback
        return str(e), 500

@app.route('/chat', methods=['POST'])
@traceable  # Enable tracing for the chat endpoint
def chat():
    """
    Handle chat messages and generate responses.
    Currently a placeholder; if you have a RAG pipeline, 
    call rag.generate_response(user_message) here.
    """
    try:
        if not request.is_json:
            return jsonify({'error': "Request must be JSON"}), 400
            
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': "Empty message received."}), 400

        logger.debug(f"Processing chat message: {user_message}")
        
        if rag is None:
            logger.error("RAG pipeline not initialized")
            return jsonify({'error': "RAG pipeline not initialized."}), 503

        response = rag.generate_response(user_message)
        logger.debug("Successfully generated response")
        return jsonify({'answer': response.answer, 'sources': response.sources})

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        logger.exception("Detailed error:")  # This will log the full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag is not None
    }), 200

@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {str(error)}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
def create_app():
    try:
        logger.info("Initializing application...")
        initialize_rag()
        logger.info("Application initialization complete")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        logger.exception("Detailed error:")
        raise

# This will be used by waitress-serve
application = create_app()

if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', 8000))
        app.run(
            host='0.0.0.0',
            port=port,
            threaded=True,
            debug=True  # Enable debug mode for local development
        )
    except Exception as e:
        logger.error(f"Failed to start the application: {str(e)}")
        logger.exception("Detailed error:")
        raise