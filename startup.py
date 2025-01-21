# startup.py
from app import application
from waitress import serve
import os

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    serve(application, host='0.0.0.0', port=port)
