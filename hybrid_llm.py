import os
from openai import OpenAI

class HybridLLMClient:
    def __init__(self):
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            self.ollama = None

        self.local_model = os.getenv("LOCAL_MODEL", "gemma:2b")
        self.cloud_model = os.getenv("CLOUD_MODEL", "gpt-4o-mini")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.cloud_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.mode = os.getenv("LLM_MODE", "local")  # 'local' or 'cloud'
        

    def set_mode(self, mode: str):
        """Switch mode at runtime."""
        if mode in ("local", "cloud"):
            self.mode = mode
        else:
            raise ValueError("LLM mode must be 'local' or 'cloud'")

    def chat(self, messages, temperature=0.3, max_tokens=2000, return_model=False):
        # Decide which model to use based on mode
        if self.mode == "local" and self.ollama:
            try:
                response = self.ollama.chat(model=self.local_model, messages=messages)
                if return_model:
                    return response["message"]["content"], self.local_model
                return response["message"]["content"]
            except Exception as e_local:
                print("Local model failed:", e_local)
                if self.mode == "local":
                    print("Falling back to cloud model")

        if self.mode == "cloud" and self.cloud_client:
            try:
                response = self.cloud_client.chat.completions.create(
                    model=self.cloud_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if return_model:
                    return response.choices[0].message.content, self.cloud_model
                return response.choices[0].message.content
            except Exception as e_cloud:
                print("Cloud model failed:", e_cloud)

        raise RuntimeError("No valid LLM available.")
    
    def available_providers(self):
        providers = []

        if self.ollama is not None:
            providers.append("local")

        if self.cloud_client is not None:
            providers.append("cloud")

        return providers


    def has_provider(self):
        return len(self.available_providers()) > 0
