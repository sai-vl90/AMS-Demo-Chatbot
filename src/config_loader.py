import os
import yaml

CONFIG_PATH = "configs/credentials.yaml"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_environment_variables(config):
    # Retrieve the tokens from the config
    deeplake_token = config.get('deeplake', {}).get('token', None)
    huggingface_token = config.get('huggingface', {}).get('token', None)
    
    if deeplake_token:
        os.environ["ACTIVELOOP_TOKEN"] = deeplake_token
    else:
        raise ValueError("Deep Lake token not found in credentials.yaml")
        
    if huggingface_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token
    else:
        raise ValueError("Hugging Face token not found in credentials.yaml")

if __name__ == "__main__":
    config = load_config()
    set_environment_variables(config)
    print("Environment variables set successfully!")