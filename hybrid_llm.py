from openai import OpenAI
class HybridLLMClient:
    def __init__(self, local_model="gemma:2b", cloud_model="gpt-4o-mini", openai_key=None):
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            self.ollama = None

        self.local_model = local_model
        self.cloud_model = cloud_model
        self.openai_key = openai_key
        self.cloud_client = OpenAI(api_key=openai_key) if openai_key else None

    def chat(self, messages, temperature=0.3, max_tokens=2000, return_model=False):
        # Try local first
        if self.ollama:
            try:
                response = self.ollama.chat(model=self.local_model, messages=messages)
                if return_model:
                    return response["message"]["content"], self.local_model
                return response["message"]["content"]
            except Exception as e_local:
                print("Local model failed:", e_local)

        # Cloud fallback
        if self.cloud_client:
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

        raise RuntimeError("Both local and cloud models failed.")