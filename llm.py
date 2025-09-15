import requests

def hf_generate_text(model_id: str, prompt: str, hf_token: str, max_tokens=200):
    """Call Hugging Face Inference API for text generation."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        try:
            return resp.json()[0]["generated_text"]
        except Exception:
            return str(resp.json())
    else:
        return f"Error {resp.status_code}: {resp.text}"
