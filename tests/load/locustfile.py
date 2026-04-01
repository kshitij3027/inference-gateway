"""Locust load test for the inference gateway.

Simulates multi-tenant bursty traffic with model and prompt length mix.
Run with: make loadtest (requires gateway running via make up or make chaos)
"""

import random

from locust import HttpUser, between, task


# --- Prompt templates of varying lengths ---

SHORT_PROMPTS = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "Say hi"}],
    [{"role": "user", "content": "Tell me a joke"}],
]

MEDIUM_PROMPTS = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a hash table is and why it is useful."},
    ],
    [
        {"role": "system", "content": "You are a coding tutor."},
        {"role": "user", "content": "How do I reverse a linked list in Python?"},
        {"role": "assistant", "content": "You can reverse a linked list iteratively."},
        {"role": "user", "content": "Show me the code."},
    ],
]

LONG_PROMPTS = [
    [
        {"role": "system", "content": "You are an expert software architect. Provide detailed, "
         "well-structured answers with code examples when appropriate. Consider "
         "trade-offs, edge cases, and production readiness in your responses."},
        {"role": "user", "content": "What is the CAP theorem?"},
        {"role": "assistant", "content": "The CAP theorem states that a distributed system "
         "can only provide two of three guarantees: Consistency, Availability, "
         "and Partition tolerance."},
        {"role": "user", "content": "How does this apply to database selection?"},
        {"role": "assistant", "content": "When choosing a database, you must consider which "
         "of the CAP properties matter most for your use case."},
        {"role": "user", "content": "Give me a detailed comparison of PostgreSQL vs Cassandra "
         "in terms of CAP properties, with specific use cases for each."},
    ],
]


def _pick_prompt():
    """Pick a random prompt with weighted length distribution: 50% short, 30% medium, 20% long."""
    roll = random.random()
    if roll < 0.50:
        return random.choice(SHORT_PROMPTS)
    if roll < 0.80:
        return random.choice(MEDIUM_PROMPTS)
    return random.choice(LONG_PROMPTS)


class TenantAlphaUser(HttpUser):
    """Steady, rate-limited tenant (10 rps, 60 rpm)."""

    wait_time = between(0.5, 2.0)
    weight = 1

    def on_start(self):
        self.api_key = "test-alpha-key"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @task(7)
    def chat_gpt_markdown(self):
        self.client.post(
            "/v1/chat/completions",
            json={"model": "mock-gpt-markdown", "messages": _pick_prompt()},
            headers=self.headers,
            name="/v1/chat/completions [mock-gpt-markdown]",
        )

    @task(3)
    def chat_claude_markdown(self):
        self.client.post(
            "/v1/chat/completions",
            json={"model": "mock-claude-markdown", "messages": _pick_prompt()},
            headers=self.headers,
            name="/v1/chat/completions [mock-claude-markdown]",
        )


class TenantBetaUser(HttpUser):
    """Bursty, unrestricted tenant (no rate limits, wildcard models)."""

    wait_time = between(0.1, 1.0)
    weight = 2  # 2x more beta users than alpha

    def on_start(self):
        self.api_key = "test-beta-key"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @task(5)
    def chat_gpt_markdown(self):
        self.client.post(
            "/v1/chat/completions",
            json={"model": "mock-gpt-markdown", "messages": _pick_prompt()},
            headers=self.headers,
            name="/v1/chat/completions [mock-gpt-markdown]",
        )

    @task(3)
    def chat_claude_markdown(self):
        self.client.post(
            "/v1/chat/completions",
            json={"model": "mock-claude-markdown", "messages": _pick_prompt()},
            headers=self.headers,
            name="/v1/chat/completions [mock-claude-markdown]",
        )

    @task(2)
    def chat_gpt_streaming(self):
        """Streaming request — reads full SSE response."""
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-gpt-markdown",
                "messages": random.choice(SHORT_PROMPTS),
                "stream": True,
            },
            headers=self.headers,
            name="/v1/chat/completions [mock-gpt-markdown] (stream)",
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # Consume the stream
                for _ in response.iter_lines():
                    pass
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
