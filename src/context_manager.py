"""Context Manager — Tracks conversation history and token budget."""


class ContextManager:
    """Maintains the message history and enforces the token window."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return the current message list."""
        return list(self.messages)

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages.clear()
