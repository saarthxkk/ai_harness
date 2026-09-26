"""Verifier — Validates agent outputs and tool results."""


class Verifier:
    """Checks that tool outputs and final answers meet acceptance criteria."""

    def verify(self, result: str) -> bool:
        """Return True if the result passes verification."""
        raise NotImplementedError("Verifier.verify() is not yet implemented.")
