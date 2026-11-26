from __future__ import annotations

import logging
import sys

from moltbook.heartbeat import MoltHeartbeat
from moltbook.runner import MoltRunner
from runtime.bootstrap import create_runtime
from utils.logging import configure_logging


def main() -> None:
    configure_logging()

    logger = logging.getLogger("molt")
    runtime = None
    exit_code = 0

    try:
        logger.info("Initializing Molt runtime.")

        runtime = create_runtime()

        runner = MoltRunner(
            agent_loop=runtime.agent_loop,
            memory=runtime.memory,
            policy=runtime.policy,
        )

        logger.info("Molt runtime initialized successfully.")

        heartbeat = MoltHeartbeat(
            callback=runner.run_cycle,
            policy=runtime.policy,
        )

        heartbeat.run()

    except KeyboardInterrupt:
        logger.info("Molt interrupted by user.")

    except Exception:
        logger.exception("Molt runtime failed.")
        exit_code = 1

    finally:
        if runtime is not None:
            runtime.close()
            logger.info("Molt runtime shut down.")

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
