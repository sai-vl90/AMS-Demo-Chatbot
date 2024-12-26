from flask import Flask, request, jsonify, render_template, send_from_directory
from src.generation.rag_chain import RAGPipeline
from src.config_loader import load_config, set_environment_variables
import os
import logging
from flask_cors import CORS

# Initialize Flask app with correct paths
app = Flask(__name__, 
    template_folder='frontend',
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

# Global variable for RAG pipeline
rag = None

def initialize_rag():
    """Initialize the RAG pipeline with configuration settings."""
    global rag
    try:
        config = load_config()
        set_environment_variables(config)
        rag = RAGPipeline(
            dataset_path=config['deeplake']['dataset_path'],
            huggingface_token=config['huggingface']['token']
        )
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
        raise

# Static file handler
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory."""
    return send_from_directory('frontend', path)

@app.route('/')
def home():
    """Render the main chat interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate responses using the RAG pipeline."""
    global rag
    try:
        # Check if RAG pipeline is initialized
        if rag is None:
            return jsonify({'error': "RAG pipeline not initialized."}), 503

        # Validate request
        if not request.is_json:
            return jsonify({'error': "Request must be JSON"}), 400
            
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': "Empty message received."}), 400

        # Generate response using RAG pipeline
        logger.info(f"Generating response for message: {user_message}")
        response = rag.generate_response(user_message)
        
        # Format sources if they exist
        sources = []
        if hasattr(response, 'sources'):
            sources = [
                {
                    "content": src.get('content', ''),
                    "metadata": src.get('metadata', 'No metadata available')
                }
                for src in response.sources
            ]

        # Return formatted response
        return jsonify({
            'answer': response.answer,
            'sources': sources
        })

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def check_port_availability(port):
    """Check if the specified port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

if __name__ == '__main__':
    try:
        # Initialize RAG pipeline
        initialize_rag()
        
        # Check port availability
        port = 5000
        if not check_port_availability(port):
            logger.warning(f"Port {port} is already in use. Trying port 8000...")
            port = 8000
            if not check_port_availability(port):
                raise RuntimeError("Both ports 5000 and 8000 are in use. Please free up a port or specify a different one.")
        
        # Run the Flask app
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