import logging

from moltbook.context import MoltContext
from moltbook.policy import MoltPolicy


logger = logging.getLogger(__name__)


class MoltRunner:

    def __init__(
        self,
        agent_loop=None,
        memory=None,
        policy: MoltPolicy | None = None,
    ):
        self.agent_loop = agent_loop
        self.memory = memory
        self.policy = policy or MoltPolicy()

        self.context = MoltContext(
            memory=self.memory,
        )

    def run_cycle(self):

        if self.agent_loop is None:
            raise RuntimeError(
                "AgentLoop has not been configured."
            )

        logger.info("Starting Molt autonomous cycle.")

        context = self.context.build_context()

        # The exact AgentLoop invocation will be implemented
        # after inspecting its actual public interface.

        return {
            "status": "ready",
            "context": context,
        }