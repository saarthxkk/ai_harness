"""Orchestrator — Agent loop that drives tool-use iterations."""


class Orchestrator:
    """Runs the think → act → observe loop up to max_iterations."""

    def __init__(self, model_client, tools: list, max_iterations: int = 10):
        self.model_client = model_client
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        """Execute the agent loop for the given task and return the final output."""
        raise NotImplementedError("Orchestrator.run() is not yet implemented.")
