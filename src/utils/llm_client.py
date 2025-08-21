import asyncio
import logging
import os
import time

from groq import AsyncGroq, RateLimitError

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GroqClient:
    """A simple and reliable Groq API client with error handling."""

    def __init__(self, api_key: str = None):
        """
        Initializes the GroqClient.

        Args:
            api_key: The Groq API key. Defaults to GROQ_API_KEY env variable.
        """
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is required. Set it as an argument or in GROQ_API_KEY environment variable.")

        self.client = AsyncGroq(api_key=api_key)
        self._rate_limit_reset_time = 0

    async def chat_completion(
        self,
        messages: list[dict],
        model: str = "llama3-70b-8192",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> str:
        """
        Generates a chat completion using the Groq API.

        Args:
            messages: A list of message dictionaries.
            model: The model to use for the completion.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.
            retries: The number of retries for transient errors.
            backoff_factor: The factor to increase delay between retries.

        Returns:
            The content of the chat completion message.
        """
        last_exception = None
        for attempt in range(retries):
            try:
                # Basic rate limiting: wait if we hit a rate limit
                if time.time() < self._rate_limit_reset_time:
                    sleep_time = self._rate_limit_reset_time - time.time()
                    logging.warning(f"Rate limit likely active. Waiting for {sleep_time:.2f} seconds.")
                    await asyncio.sleep(sleep_time)

                chat_completion = await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return chat_completion.choices[0].message.content

            except RateLimitError as e:
                logging.warning(f"Rate limit error (attempt {attempt + 1}/{retries}): {e}")
                # Simple exponential backoff
                self._rate_limit_reset_time = time.time() + (backoff_factor ** attempt)
                last_exception = e
                await asyncio.sleep(backoff_factor ** attempt)

            except Exception as e:
                logging.error(f"An unexpected error occurred (attempt {attempt + 1}/{retries}): {e}")
                last_exception = e
                await asyncio.sleep(backoff_factor ** attempt)

        raise ConnectionError(f"Failed to get completion after {retries} retries.") from last_exception


async def main():
    """Example usage of the GroqClient."""
    try:
        client = GroqClient()
        messages = [
            {
                "role": "user",
                "content": "Explain the importance of low-latency LLMs",
            }
        ]

        logging.info("--- Using LLaMA3-70b ---")
        response = await client.chat_completion(
            messages=messages,
            model="llama3-70b-8192",
        )
        logging.info(f"Response: {response}")

        logging.info("\n--- Using Mixtral-8x7b ---")
        response_mixtral = await client.chat_completion(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=500,
        )
        logging.info(f"Response: {response_mixtral}")

    except (ValueError, ConnectionError) as e:
        logging.error(f"Failed to run example: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the example run: {e}")

if __name__ == "__main__":
    # To run this example, make sure to set the GROQ_API_KEY environment variable.
    # export GROQ_API_KEY="your_api_key_here"
    asyncio.run(main())
