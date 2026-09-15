import os
subscription_key = str(os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY"))

## AnthropicVertex
from anthropic import AnthropicVertex # type: ignore
from google.genai.types import HttpOptions # type: ignore

client = AnthropicVertex(
    access_token=subscription_key,
    project_id="_",
    region="_",
    base_url="https://aoai-farm.bosch-temp.com/api/google/v1",
)

response = client.messages.create(
    model="claude-opus-4-1@20250805",
    max_tokens=1024,
    messages=[
      { "role": "user", "content": "Hello!" }
    ]
)

print(response.model_dump_json(indent=2))
###