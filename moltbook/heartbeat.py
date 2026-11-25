import logging
import signal
import time

from moltbook.policy import MoltPolicy


logger = logging.getLogger(__name__)


class MoltHeartbeat:

    def __init__(
        self,
        callback,
        policy: MoltPolicy | None = None,
    ):
        self.callback = callback
        self.policy = policy or MoltPolicy()
        self.running = False

    def stop(self):
        logger.info("Molt heartbeat stopping.")
        self.running = False

    def run(self):

        self.running = True

        signal.signal(
            signal.SIGINT,
            lambda *_: self.stop(),
        )

        logger.info("Molt heartbeat started.")

        while self.running:

            try:
                self.callback()

            except Exception:
                logger.exception(
                    "An error occurred during Molt's cycle."
                )

            interval = self.policy.get_heartbeat_interval()

            logger.info(
                "Next heartbeat in %s seconds.",
                interval,
            )

            for _ in range(interval):
                if not self.running:
                    break

                time.sleep(1)

        logger.info("Molt heartbeat stopped.")