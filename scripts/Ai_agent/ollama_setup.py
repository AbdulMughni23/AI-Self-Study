import os
import subprocess

def setup_ollama():
    """Ensure Ollama is installed and the Mistral model is pulled."""
    try:
        #check if Ollama is inatalled 
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Ollama is alredy installed.")
        else:
            raise subprocess.CalledProcessError(result.returncode, 'ollama --version')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ollama not found, please install Ollama fisrt: https://ollama.ai/download")
        return False

    #check if Mistral model is available 
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'mistral' in result.stdout:
            print("Mistral model is already pulled.")

        else:
            print("pulling Mistral model...")
            subprocess.run(['ollama', 'pull', 'mistral'], check=True)
            print("Mistral model pulled successfully")
    except subprocess.CalledProcessError as e:
        print(f"error pulling Mistral model: {e}")
        return False
    
    return True

if __name__ == '__main__':
    if setup_ollama():
        print("Ollama setup colmplete.")
    else:
        print("Ollama setup failed")