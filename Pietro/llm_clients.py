"""
LLM client implementations for Anthropic and OpenAI APIs.
"""

import os
import time
from typing import Dict, Optional
from abc import ABC, abstractmethod
import anthropic
from openai import OpenAI
from config import API_SETTINGS


class LLMClient(ABC):

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.max_retries = API_SETTINGS['max_retries']
        self.retry_delay = API_SETTINGS['retry_delay']
        self.timeout = API_SETTINGS['timeout']

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text from the given prompt."""
        pass

    def generate_to_length(self, prompt: str, target_word_count: int,
                          temperature: float = 0.7, max_tokens: int = 2000,
                          max_iterations: int = 5) -> str:
   
        full_text = ""
        iteration = 0

        while iteration < max_iterations:
            word_count = len(full_text.split()) if full_text else 0

            if word_count >= target_word_count:
                print(f"  Target reached: {word_count} words")
                break

            words_needed = target_word_count - word_count
            print(f"  Iteration {iteration + 1}: {word_count}/{target_word_count} words (need {words_needed} more)")

            if iteration == 0:
                chunk = self.generate(prompt, temperature, max_tokens)
            else:
                continuation_prompt = self._create_continuation_prompt(
                    original_prompt=prompt,
                    text_so_far=full_text,
                    words_needed=words_needed
                )
                chunk = self.generate(continuation_prompt, temperature, max_tokens)

            full_text += chunk
            iteration += 1

        return full_text

    def _create_continuation_prompt(self, original_prompt: str,
                                   text_so_far: str, words_needed: int) -> str:
        return f"""Continue the story below. You have written {len(text_so_far.split())} words so far, and need to write approximately {words_needed} more words to reach the target length.

IMPORTANT: Continue seamlessly from where the text ends. Maintain the same punctuation style, narrative voice, and story coherence.

Original instructions:
{original_prompt}

Story so far:
{text_so_far}

Continue the story:"""

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-3-5-sonnet-20241022'):
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using Claude."""

        def _make_request():
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text

        return self._retry_with_backoff(_make_request)


class OpenAIClient(LLMClient):
    """Client for OpenAI's GPT API."""

    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4'):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        super().__init__(api_key, model)
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using GPT."""

        def _make_request():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_make_request)


def get_client(provider: str, model: Optional[str] = None) -> LLMClient:
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: 'anthropic' or 'openai'
        model: Optional model name (uses default if not specified)

    Returns:
        LLMClient instance
    """
    if provider.lower() == 'anthropic':
        return AnthropicClient(model=model) if model else AnthropicClient()
    elif provider.lower() == 'openai':
        return OpenAIClient(model=model) if model else OpenAIClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'anthropic' or 'openai'")


# Convenience functions for quick access
def generate_with_claude(prompt: str, model: str = 'claude-3-5-sonnet-20241022',
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Generate text with Claude."""
    client = AnthropicClient(model=model)
    return client.generate(prompt, temperature, max_tokens)


def generate_with_gpt(prompt: str, model: str = 'gpt-4',
                     temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Generate text with GPT."""
    client = OpenAIClient(model=model)
    return client.generate(prompt, temperature, max_tokens)
