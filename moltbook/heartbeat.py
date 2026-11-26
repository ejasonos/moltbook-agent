from __future__ import annotations

import logging
import signal
import threading
import uuid
from datetime import datetime, timedelta, timezone

from moltbook.policy import MoltPolicy


logger = logging.getLogger("local_agent.moltbook.heartbeat")


class MoltHeartbeat:
    """Persistent, restart-aware heartbeat for the independent worker process."""

    def __init__(
        self,
        callback,
        policy: MoltPolicy | None = None,
        memory=None,
        *,
        max_backoff_seconds: int = 86400,
        lease_seconds: int = 7200,
    ):
        self.callback = callback
        self.policy = policy or MoltPolicy()
        self.memory = memory
        self.max_backoff_seconds = max(60, max_backoff_seconds)
        self.lease_seconds = max(60, lease_seconds)
        self.running = False
        self._stop_event = threading.Event()
        self._cycle_lock = threading.Lock()
        self._owner = uuid.uuid4().hex

    def stop(self):
        logger.info("Penelope_360 heartbeat stopping.")
        self.running = False
        self._stop_event.set()

    def run(self):
        self.running = True
        old_handlers = self._install_signal_handlers()
        interval = self.policy.get_heartbeat_interval()
        next_run = self._initial_schedule()
        logger.info("Penelope_360 heartbeat started | interval_seconds=%s", interval)
        try:
            while self.running and not self._stop_event.is_set():
                if next_run:
                    delay = max(0.0, (next_run - datetime.now(timezone.utc)).total_seconds())
                    if delay and self._stop_event.wait(delay):
                        break
                if not self.running or self._stop_event.is_set():
                    break

                if not self._cycle_lock.acquire(blocking=False):
                    logger.warning("Heartbeat cycle skipped because another cycle is active.")
                    next_run = datetime.now(timezone.utc) + timedelta(seconds=interval)
                    continue

                lease_acquired = self.memory is None
                try:
                    if self.memory is not None:
                        try:
                            lease_acquired = self.memory.store.acquire_heartbeat_lease(
                                self._owner,
                                self.lease_seconds,
                            )
                        except Exception:
                            lease_acquired = False
                            logger.exception("Could not acquire the heartbeat worker lease.")
                    if not lease_acquired:
                        logger.warning("Heartbeat cycle skipped because another worker owns the lease.")
                        next_run = datetime.now(timezone.utc) + timedelta(seconds=interval)
                        continue
                    success, error = self._run_cycle()
                except Exception as exc:
                    success = False
                    error = f"{type(exc).__name__}: heartbeat infrastructure failed"
                    logger.exception("Heartbeat infrastructure failed; the worker will retry.")
                finally:
                    if lease_acquired and self.memory is not None:
                        try:
                            self.memory.store.release_heartbeat_lease(self._owner)
                        except Exception:
                            logger.warning("Could not release heartbeat worker lease.")
                    self._cycle_lock.release()

                failures = int(self._get_state("consecutive_failure_count", "0") or 0)
                if success:
                    delay = interval
                    failures = 0
                else:
                    failures += 1
                    delay = min(
                        max(interval, 60) * (2 ** min(failures - 1, 8)),
                        max(self.max_backoff_seconds, interval),
                    )
                self._set_state("consecutive_failure_count", str(failures))
                next_run = datetime.now(timezone.utc) + timedelta(seconds=delay)
                self._set_state("next_scheduled_cycle", next_run.isoformat())
                logger.info(
                    "Next heartbeat scheduled | delay_seconds=%s | consecutive_failures=%s | error=%s",
                    delay,
                    failures,
                    error,
                )
        finally:
            self.running = False
            self._restore_signal_handlers(old_handlers)
            logger.info("Penelope_360 heartbeat stopped.")

    def _run_cycle(self) -> tuple[bool, str | None]:
        if self.memory is not None:
            self._set_state("last_cycle_start", datetime.now(timezone.utc).isoformat())
            self._set_state("cycle_status", "running")
        try:
            result = self.callback()
            success = self._result_succeeded(result)
            error = getattr(result, "error", None)
            if not success and result is not None:
                errors = getattr(result, "unresolved_errors", None)
                if errors:
                    error = "; ".join(str(item) for item in errors)
                error = error or "Cycle completed with unresolved work."
            finished = datetime.now(timezone.utc).isoformat()
            self._set_state("cycle_status", "success" if success else "failed")
            self._set_state("last_cycle_completion", finished)
            if success:
                self._set_state("last_successful_cycle", finished)
                self._set_state("last_error", "")
            else:
                self._set_state("last_error", str(error or "Cycle failed.")[:1000])
            return success, None if success else str(error or "Cycle failed.")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Molt cycle failed; the worker will retry on its schedule.")
            finished = datetime.now(timezone.utc).isoformat()
            self._set_state("cycle_status", "failed")
            self._set_state("last_cycle_completion", finished)
            self._set_state("last_error", error[:1000])
            return False, error[:1000]

    @staticmethod
    def _result_succeeded(result) -> bool:
        if result is None:
            return True
        if hasattr(result, "success"):
            return bool(result.success)
        if hasattr(result, "completed"):
            return bool(result.completed and not getattr(result, "error", None))
        return True

    def _initial_schedule(self):
        if self.memory is None:
            return None
        next_scheduled = self._get_state("next_scheduled_cycle")
        last_start = self._get_state("last_cycle_start")
        if next_scheduled and last_start:
            try:
                parsed = datetime.fromisoformat(next_scheduled.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if parsed > datetime.now(timezone.utc):
                    logger.info("Resuming persisted heartbeat schedule.")
                    return parsed.astimezone(timezone.utc)
            except ValueError:
                logger.warning("Ignoring invalid persisted heartbeat schedule.")
        return None

    def _install_signal_handlers(self):
        handlers = {}
        if threading.current_thread() is not threading.main_thread():
            return handlers
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_args: self.stop())
        return handlers

    @staticmethod
    def _restore_signal_handlers(handlers):
        for sig, handler in handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _get_state(self, key: str, default: str | None = None):
        if self.memory is None:
            return default
        try:
            return self.memory.store.get_state(key) or default
        except Exception:
            logger.warning("Could not read heartbeat state | state_key=%s", key)
            return default

    def _set_state(self, key: str, value: str):
        if self.memory is None:
            return
        try:
            self.memory.store.set_state(key, value)
        except Exception:
            logger.warning("Could not persist heartbeat state | state_key=%s", key)
