import os
import yaml
import requests

CONFIG_PATH = "configs/credentials.yaml"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_environment_variables(config):
    """
    Reads the YAML config and sets environment variables.
    """
    # Azure Search configuration
    azure_search_config = config.get('azure_search', {})
    if azure_search_config:
        os.environ["AZURE_SEARCH_ENDPOINT"] = azure_search_config.get('endpoint', '')
        os.environ["AZURE_SEARCH_KEY"] = azure_search_config.get('key', '')
        os.environ["AZURE_SEARCH_INDEX_NAME"] = azure_search_config.get('index_name', '')

        if not os.environ["AZURE_SEARCH_ENDPOINT"] or \
           not os.environ["AZURE_SEARCH_KEY"] or \
           not os.environ["AZURE_SEARCH_INDEX_NAME"]:
            raise ValueError("Azure Search config (endpoint, key, or index_name) is missing or invalid in credentials.yaml")
    else:
        raise ValueError("Azure Search configuration not found in credentials.yaml")

    # Hugging Face configuration    
    huggingface_config = config.get('huggingface', {})
    if huggingface_config:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_config.get('token', '')
    else:
        raise ValueError("Hugging Face configuration not found in credentials.yaml")
    
    # Groq configuration
    groq_config = config.get('groq', {})
    if groq_config:
        os.environ["GROQ_API_KEY"] = groq_config.get('api_key', '')
    else:
        raise ValueError("Groq configuration not found in credentials.yaml")
    
    # Langsmith configuration - using exact keys from credentials.yaml
    langsmith_config = config.get('langsmith', {})
    if langsmith_config:
        os.environ['LANGCHAIN_TRACING_V2'] = langsmith_config.get('LANGCHAIN_TRACING_V2', 'false')
        os.environ['LANGCHAIN_ENDPOINT'] = langsmith_config.get('LANGCHAIN_ENDPOINT', '')
        os.environ['LANGCHAIN_API_KEY'] = langsmith_config.get('LANGCHAIN_API_KEY', '')
        os.environ['LANGCHAIN_PROJECT'] = langsmith_config.get('LANGCHAIN_PROJECT', '')
        
        # Validate required LangSmith settings
        if not os.environ['LANGCHAIN_API_KEY'] or not os.environ['LANGCHAIN_PROJECT']:
            raise ValueError("LangSmith API key or project name is missing in credentials.yaml")
    else:
        raise ValueError("Langsmith configuration not found in credentials.yaml")

def get_langsmith_config():
    """
    Returns the Langsmith configuration variables for use in other files
    """
    return {
        "tracing_v2": os.getenv('LANGCHAIN_TRACING_V2'),
        "endpoint": os.getenv('LANGCHAIN_ENDPOINT'),
        "api_key": os.getenv('LANGCHAIN_API_KEY'),
        "project": os.getenv('LANGCHAIN_PROJECT')
    }

def setup_langchain_env():
    """Set up LangChain environment with the loaded configuration"""
    config = load_config()
    set_environment_variables(config)
    return get_langsmith_config()

def get_langsmith_api_key():
    return os.getenv('LANGCHAIN_API_KEY')

def get_langchain_endpoint():
    return os.getenv('LANGCHAIN_ENDPOINT')

def is_tracing_enabled():
    return os.getenv('LANGCHAIN_TRACING_V2', '').lower() == 'true'

if __name__ == "__main__":
    try:
        # Load config & set environment variables
        config = load_config()
        set_environment_variables(config)
        print("\nEnvironment variables set successfully!")

        # Print out LangSmith info
        print("\nLangSmith Configuration:")
        print(f"Tracing Enabled: {is_tracing_enabled()}")
        print(f"Endpoint: {get_langchain_endpoint()}")
        print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
        print(f"API Key Status: {'Set' if get_langsmith_api_key() else 'Not Set'}")

    except Exception as e:
        print(f"Error setting environment variables: {str(e)}")