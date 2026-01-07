import requests

def groq_generate_text(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1000):
    """
    Call Groq API for text generation.
    
    Args:
        prompt: The input prompt
        api_key: Groq API key
        model: Model ID (default: llama-3.3-70b-versatile)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if resp.status_code == 200:
            result = resp.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error {resp.status_code}: {resp.text}"
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"