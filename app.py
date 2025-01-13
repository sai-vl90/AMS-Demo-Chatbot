import os
import logging

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from langsmith import traceable

# If you're using a RAG pipeline, uncomment these:
from src.generation.rag_chain import RAGPipeline
from src.config_loader import load_config, set_environment_variables
import os


app = Flask(
    __name__, 
    template_folder='frontend',  # where index.html is
    static_folder='frontend',
    static_url_path=''
)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# If you have a RAG pipeline:
rag = None

def initialize_rag():
    global rag
    try:
        config = load_config()
        set_environment_variables(config)
        rag = RAGPipeline(
            huggingface_token=config['huggingface']['token']
        )
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
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
    return send_from_directory('frontend', path)

@app.route('/')
def home():
    """
    Render the main chat interface (index.html),
    passing in the list of local doc names so the greeting
    can display them.
    """
    try:
        doc_names = list_local_preprocessed_docs("data/preprocessed")
        return render_template('index.html', doc_names=doc_names)
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
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
        # if rag is None:
        #     return jsonify({'error': "RAG pipeline not initialized."}), 503

        if not request.is_json:
            return jsonify({'error': "Request must be JSON"}), 400
            
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': "Empty message received."}), 400

        # Example placeholder response:
        # If you have a RAG pipeline:
        response = rag.generate_response(user_message)
        return jsonify({'answer': response.answer, 'sources': response.sources})

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def check_port_availability(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

if __name__ == '__main__':
    try:
        # If you have a RAG pipeline, uncomment:
        initialize_rag()

        port = 5000
        if not check_port_availability(port):
            logger.warning(f"Port {port} is already in use. Trying port 8000...")
            port = 8000
            if not check_port_availability(port):
                raise RuntimeError("Both ports 5000 and 8000 are in use. Please free up a port or specify a different one.")
        
        logger.info(f"Starting Flask app on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            threaded=True,
            debug=False
        )
    except Exception as e:
        logger.error(f"Failed to start the application: {str(e)}")
        raise
