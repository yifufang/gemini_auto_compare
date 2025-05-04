import httpx
from dotenv import load_dotenv
import os

load_dotenv()

class Deepseek:
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, input):
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "based on the input, generate a concise 1 paragraph travel plan"},
                {"role": "user", "content": input}
            ],
            "stream": False
        }

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()  # Raise an error for bad responses
                data = response.json()
                return data["choices"][0]["message"]["content"] if "choices" in data and len(data["choices"]) > 0 else "No response generated."
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please set the DEEPSEEK_API_KEY environment variable.")
    else:
        deepseek = Deepseek(os.getenv("DEEPSEEK_API_KEY"))
        print(deepseek.generate_response("Plan a trip to Paris for 3 days."))

        
