from typing import (
    Callable,
    List,
)

import google.generativeai as genai
import google.api_core.exceptions
from llm.basellm import BaseLLM
from retry import retry


class GeminiChat(BaseLLM):
    """Wrapper around Gemini Chat large language models."""

    def __init__(
        self,
        gemini_api_key: str,
        # model_name: str = "gemini-1.5-pro-latest",
        model_name: str = "gemini-1.5-flash-latest",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: List[str],
    ) -> str:
        try:
            response = self.model.generate_content(
                                                    messages,
                                                    generation_config=genai.types.GenerationConfig(
                                                        # Only one candidate for now.
                                                        candidate_count=1,
                                                        max_output_tokens=self.max_tokens,
                                                        temperature=self.temperature,
                                                    )
                                                )
            return response.text
        except (google.api_core.exceptions.TooManyRequests, google.api_core.exceptions.BadRequest, google.api_core.exceptions.Forbidden) as e:
           return str(f"Error: {e}")
        except google.api_core.exceptions.InternalServerError as e:
            return str(f"Error: {e}")
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise Exception()

    async def generateStreaming(
        self,
        messages: List[str],
        onTokenCallback=Callable[[str], None],
    ) -> str:
        result = []
        response = self.model.generate_content(
                                                    messages,
                                                    generation_config=genai.types.GenerationConfig(
                                                        # Only one candidate for now.
                                                        candidate_count=1,
                                                        max_output_tokens=self.max_tokens,
                                                        temperature=self.temperature,
                                                    ),
                                                    stream=True
                                                )
        result = []
        for chunk in response:
            result.append(chunk.text)
            await onTokenCallback(chunk.text)
        return result

    def num_tokens_from_string(self, string: str) -> int:
        num_tokens = self.model.count_tokens(string)
        return num_tokens

    def max_allowed_token_length(self) -> int:
        # TODO: list all models and their max tokens from api
        return 1048576 # flash
        # return 2097152 # pro