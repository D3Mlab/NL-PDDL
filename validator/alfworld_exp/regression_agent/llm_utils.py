import os
import sys
import openai
from openai import OpenAI, AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4o","gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure client from environment variables
# Set OPENAI_API_KEY and optionally AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_VERSION for Azure
_azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
if _azure_endpoint:
    client = AzureOpenAI(
        azure_endpoint=_azure_endpoint,
        api_key=os.getenv('AZURE_OPENAI_API_KEY', os.getenv('OPENAI_API_KEY', '')),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
    )
else:
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY', ''),
    )

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    return response.choices[0].message.content



def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        while cur_try < 1:
            if model == "text-davinci-003":
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2)
            else:
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2)
            # dumb way to do this
            if text:
                text = text.replace(">", "") 
                if len(text.strip()) >= 5:
                    return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)