import json
import logging

import anthropic

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent wrapping the Anthropic Python SDK for Claude API calls.

    Handles client initialization, LLM calls (plain text and structured JSON),
    and automatic retry for transient errors via the SDK's built-in retry logic.
    """

    def __init__(self, model="claude-opus-4-6", max_retries=3):
        self.model = model
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def _call_llm(self, system_prompt, user_message, max_tokens=4096):
        """Send a message to Claude and return the text response.

        Args:
            system_prompt: System prompt setting the agent's role/context.
            user_message: The user message to send.
            max_tokens: Maximum tokens in the response.

        Returns:
            The text content of Claude's response.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_llm_json(self, system_prompt, user_message, schema, max_tokens=4096):
        """Call Claude and parse structured JSON from the response.

        Uses the output_config format to constrain Claude's output to the
        provided JSON schema, ensuring valid and parseable results.

        Args:
            system_prompt: System prompt setting the agent's role/context.
            user_message: The user message to send.
            schema: A JSON schema dict describing the expected output structure.
            max_tokens: Maximum tokens in the response.

        Returns:
            A dict parsed from Claude's JSON response.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )
        return json.loads(response.content[0].text)
