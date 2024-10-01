import google.generativeai as genai
from embedding.base_embedding import BaseEmbedding


class GeminiEmbedding(BaseEmbedding):
    """Wrapper around Gemini embedding models."""

    def __init__(
        self, gemini_api_key: str, model_name: str = "text-embedding-004"
    ) -> None:
        genai.configure(api_key=gemini_api_key)
        self.model = model_name

    def generate(
        self,
        input: str,
    ) -> str:
        embedding = genai.embed_content(model=self.model,
                                content=input,
                                task_type="retrieval_document",
                                title="") # TODO: add title
        return embedding["embedding"]
