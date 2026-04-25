#!/usr/bin/env python3
"""
poppycock — Ubiquant AI Inference Challenge submission client.

Design priorities (in order):

    1. Stability over throughput. The competition is a single-shot offline
       evaluation; a hang or crash costs the entire score. We therefore use
       hard wall-clock timeouts on every vLLM call and *never* await on a
       cancellation cleanup path that could itself hang.

    2. Always submit. Whatever the inference outcome (success, timeout,
       exception, broken weights), the task is submitted with default-filled
       answers so the platform sees us as alive and responsive.

    3. Bounded concurrency and bounded buffering. Both the worker pool and
       the local task buffer are bounded so we cannot accumulate ghost tasks.

The most important fix versus prior versions is the timeout path. Previously
``asyncio.wait_for(generate, timeout=T)`` would cancel the inner coroutine on
timeout and *await its cleanup* — but vLLM's async generator ``aclose()``
could hang under broken-weight / unhealthy-engine conditions. ``wait_for``
then blocked indefinitely, taking down all worker slots.  This version uses
``asyncio.create_task`` + ``asyncio.wait`` and on timeout abandons the inner
task without awaiting cleanup, fires an abort to the engine, and returns an
empty answer immediately.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Team identity (hardcoded per submission requirements)
# ---------------------------------------------------------------------------
TEAM_NAME = "poppycock"
TEAM_TOKEN = "d51ebe5cc6594d53ef2853fd8f9dadc1"


# ---------------------------------------------------------------------------
# Defaults — overridable via env / contest.json
# ---------------------------------------------------------------------------
DEFAULT_PLATFORM_URL = "http://127.0.0.1:8003"
DEFAULT_MAX_CONCURRENT_TASKS = 4
DEFAULT_MAX_BUFFERED_TASKS = 8
DEFAULT_POLL_INTERVAL = 0.15
DEFAULT_IDLE_SLEEP = 0.40
DEFAULT_HTTP_TIMEOUT = 15.0
DEFAULT_TASK_TIMEOUT = 600.0          # per-task wall clock; matches eval_timeout_s default
DEFAULT_MSG_TIMEOUT = 60.0            # per-message hard ceiling on a single vLLM call
DEFAULT_ABORT_TIMEOUT = 3.0           # how long we wait for vLLM.abort() before giving up
DEFAULT_HEARTBEAT_INTERVAL = 30.0     # watchdog log cadence

SLA_PRIORITY = {
    "Supreme": 8,
    "Glorious": 7,
    "Stellar": 6,
    "Diamond": 5,
    "Platinum": 4,
    "Gold": 3,
    "Silver": 2,
    "Bronze": 1,
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_runtime_config() -> Dict[str, Any]:
    """Resolve runtime configuration from env vars and optional contest.json."""
    config_path = os.environ.get("CONFIG_PATH", "")
    config_data: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception:
            config_data = {}

    platform_url = (
        os.environ.get("PLATFORM_URL")
        or config_data.get("platform_url")
        or DEFAULT_PLATFORM_URL
    )
    model_path = os.environ.get("MODEL_PATH") or config_data.get("model_path") or ""

    return {
        "platform_url": platform_url.rstrip("/"),
        "model_path": model_path,
        "config_path": config_path,
        "http_timeout": _env_float("HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT),
        "max_concurrent_tasks": max(1, _env_int("MAX_CONCURRENT_TASKS", DEFAULT_MAX_CONCURRENT_TASKS)),
        "max_buffered_tasks": max(1, _env_int("MAX_BUFFERED_TASKS", DEFAULT_MAX_BUFFERED_TASKS)),
        "poll_interval": max(0.05, _env_float("POLL_INTERVAL", DEFAULT_POLL_INTERVAL)),
        "idle_sleep": max(0.10, _env_float("IDLE_SLEEP", DEFAULT_IDLE_SLEEP)),
        "task_timeout": max(5.0, _env_float("TASK_TIMEOUT_S", DEFAULT_TASK_TIMEOUT)),
        "msg_timeout": max(3.0, _env_float("MSG_TIMEOUT_S", DEFAULT_MSG_TIMEOUT)),
        "abort_timeout": max(0.5, _env_float("ABORT_TIMEOUT_S", DEFAULT_ABORT_TIMEOUT)),
        "heartbeat_interval": max(5.0, _env_float("HEARTBEAT_INTERVAL_S", DEFAULT_HEARTBEAT_INTERVAL)),
    }


RUNTIME = load_runtime_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("poppycock")
# Quiet down httpx access logs (one INFO per HTTP call is too noisy).
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass(order=True)
class PrioritizedTask:
    priority: int
    created_at: float
    task_id: int = field(compare=False)
    overview: Dict[str, Any] = field(compare=False)


class TaskScheduler:
    """Priority queue with explicit states and bounded buffering."""

    def __init__(self) -> None:
        self._queue: List[PrioritizedTask] = []
        self._queued_ids: set[int] = set()
        self._processing_ids: set[int] = set()
        self._completed_ids: set[int] = set()

    def add_task(self, overview: Dict[str, Any]) -> bool:
        task_id = int(overview["task_id"])
        if (
            task_id in self._queued_ids
            or task_id in self._processing_ids
            or task_id in self._completed_ids
        ):
            return False

        sla = overview.get("target_sla", "Bronze")
        reward = float(overview.get("target_reward", 0.0))
        eval_timeout = float(overview.get("eval_timeout_s", 600.0))

        score = (
            SLA_PRIORITY.get(sla, 0) * 1_000_000
            + int(reward * 100)
            - int(eval_timeout * 10)
        )

        heapq.heappush(
            self._queue,
            PrioritizedTask(
                priority=-score,
                created_at=time.time(),
                task_id=task_id,
                overview=overview,
            ),
        )
        self._queued_ids.add(task_id)
        return True

    def can_buffer_more(self, limit: int) -> bool:
        return (len(self._queue) + len(self._processing_ids)) < limit

    def has_capacity(self, limit: int) -> bool:
        return len(self._processing_ids) < limit

    def dispatch_next(self, limit: int) -> Optional[PrioritizedTask]:
        if not self.has_capacity(limit):
            return None
        while self._queue:
            task = heapq.heappop(self._queue)
            self._queued_ids.discard(task.task_id)
            if task.task_id in self._completed_ids or task.task_id in self._processing_ids:
                continue
            self._processing_ids.add(task.task_id)
            return task
        return None

    def mark_completed(self, task_id: int) -> None:
        self._processing_ids.discard(task_id)
        self._queued_ids.discard(task_id)
        self._completed_ids.add(task_id)

    def mark_failed(self, task_id: int) -> None:
        self._processing_ids.discard(task_id)
        self._queued_ids.discard(task_id)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._processing_ids)

    @property
    def completed_count(self) -> int:
        return len(self._completed_ids)


# ---------------------------------------------------------------------------
# Inference engine — vLLM-backed with hard timeouts
# ---------------------------------------------------------------------------
class InferenceEngine:
    """vLLM AsyncLLMEngine wrapper; never blocks on cancellation cleanup."""

    def __init__(self, model_path: str, abort_timeout: float) -> None:
        self.model_path = model_path
        self.abort_timeout = abort_timeout
        self.use_vllm = False
        self.async_engine: Any = None
        self.tokenizer: Any = None
        self._sampling_cls: Any = None
        self._prefix_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._abort_tasks: set[asyncio.Task[Any]] = set()

        if not (model_path and os.path.isdir(model_path)):
            logger.warning(
                "MODEL_PATH '%s' is not a directory; falling back to mock inference",
                model_path,
            )
            return

        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
            from transformers import AutoTokenizer

            tensor_parallel_size = max(1, self._detect_gpu_count())
            logger.info(
                "Loading vLLM model from %s (TP=%s)", model_path, tensor_parallel_size
            )
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=4096,
                gpu_memory_utilization=0.90,
                enable_prefix_caching=True,
                max_num_seqs=128,
                disable_log_requests=True,
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._sampling_cls = SamplingParams
            self.use_vllm = True
            logger.info("vLLM ready (TP=%s)", tensor_parallel_size)
        except Exception as exc:
            logger.warning(
                "Failed to initialize vLLM (%s); falling back to mock inference", exc
            )

    @staticmethod
    def _detect_gpu_count() -> int:
        try:
            import torch
            return max(1, torch.cuda.device_count())
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # Core: run one vLLM request with a hard wall-clock timeout
    # ------------------------------------------------------------------
    async def _run_with_timeout(
        self,
        coro: Awaitable[Any],
        request_id: str,
        timeout_s: float,
    ) -> Any:
        """Run *coro* with a hard timeout.

        Returns the coroutine's result on success.
        Returns ``None`` on timeout, exception, or if the inner task fails.
        On timeout we *do not* await cleanup of the inner task — we cancel
        and abandon, and fire an abort to the engine in the background.
        This is the key to escaping vLLM async-generator hangs.
        """
        inner = asyncio.create_task(coro, name=f"vllm-{request_id}")
        try:
            done, _ = await asyncio.wait({inner}, timeout=timeout_s)
        except asyncio.CancelledError:
            self._fire_and_forget_abort(request_id)
            inner.cancel()
            raise

        if inner in done:
            try:
                return inner.result()
            except asyncio.CancelledError:
                self._fire_and_forget_abort(request_id)
                return None
            except Exception as exc:
                logger.warning("vLLM request %s failed: %s", request_id, exc)
                self._fire_and_forget_abort(request_id)
                return None

        # Hard timeout: abandon inner without awaiting its cleanup.
        logger.warning(
            "vLLM request %s exceeded %.1fs, abandoning", request_id, timeout_s
        )
        self._fire_and_forget_abort(request_id)
        inner.cancel()
        # Note: we deliberately do NOT await `inner` — its cleanup might hang.
        return None

    def _fire_and_forget_abort(self, request_id: str) -> None:
        """Schedule an engine abort but never await it on the hot path."""
        if not self.use_vllm or self.async_engine is None:
            return
        try:
            task = asyncio.create_task(
                self._safe_abort(request_id), name=f"abort-{request_id}"
            )
            self._abort_tasks.add(task)
            task.add_done_callback(self._abort_tasks.discard)
        except RuntimeError:
            # No running loop (shouldn't happen in normal flow)
            pass

    async def _safe_abort(self, request_id: str) -> None:
        try:
            res = self.async_engine.abort(request_id)
            if asyncio.iscoroutine(res):
                await asyncio.wait_for(res, timeout=self.abort_timeout)
            logger.info("Aborted vLLM request %s", request_id)
        except asyncio.TimeoutError:
            logger.warning("abort(%s) itself timed out after %.1fs",
                           request_id, self.abort_timeout)
        except Exception as exc:
            logger.debug("abort(%s) error: %s", request_id, exc)

    async def _consume_generation(
        self, prompt: str, sampling_params: Any, request_id: str
    ) -> Any:
        final_output: Any = None
        results_generator = self.async_engine.generate(
            prompt, sampling_params, request_id
        )
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    # ------------------------------------------------------------------
    # Public APIs used by TaskProcessor
    # ------------------------------------------------------------------
    async def generate_until(
        self,
        prompt: str,
        gen_kwargs: Dict[str, Any],
        timeout_s: float,
    ) -> str:
        if not self.use_vllm or self.async_engine is None:
            return self._mock_generate(prompt, gen_kwargs)

        gk = gen_kwargs or {}
        top_k = gk.get("top_k", -1)
        sampling_params = self._sampling_cls(
            temperature=gk.get("temperature", 0.0),
            top_p=gk.get("top_p", 1.0),
            top_k=top_k if isinstance(top_k, int) and top_k > 0 else -1,
            max_tokens=int(gk.get("max_gen_toks", 256)),
            stop=gk.get("until", []) or [],
            repetition_penalty=gk.get("repetition_penalty", 1.0),
            frequency_penalty=gk.get("frequency_penalty", 0.0),
            presence_penalty=gk.get("presence_penalty", 0.0),
        )
        request_id = f"gen-{uuid.uuid4().hex}"
        final_output = await self._run_with_timeout(
            self._consume_generation(prompt, sampling_params, request_id),
            request_id,
            timeout_s,
        )
        if final_output is None or not getattr(final_output, "outputs", None):
            return ""
        return final_output.outputs[0].text or ""

    async def compute_loglikelihood_one(
        self,
        prompt: str,
        continuation: str,
        timeout_s: float,
    ) -> float:
        cache_key = self._ll_cache_key(prompt, continuation)
        if cache_key in self._prefix_cache:
            self._cache_hits += 1
            return self._prefix_cache[cache_key]
        self._cache_misses += 1

        if not self.use_vllm or self.async_engine is None or self.tokenizer is None:
            score = self._mock_loglikelihood(continuation)
            self._prefix_cache[cache_key] = score
            return score

        sampling_params = self._sampling_cls(
            temperature=0.0, max_tokens=1, prompt_logprobs=1
        )
        full_text = prompt + continuation
        request_id = f"ll-{uuid.uuid4().hex}"

        final_output = await self._run_with_timeout(
            self._consume_generation(full_text, sampling_params, request_id),
            request_id,
            timeout_s,
        )
        if final_output is None:
            return -10.0

        prompt_logprobs = getattr(final_output, "prompt_logprobs", None)
        if not prompt_logprobs:
            return -10.0

        try:
            prompt_token_count = len(
                self.tokenizer.encode(prompt, add_special_tokens=False)
            )
        except Exception:
            return -10.0

        total = 0.0
        counted = 0
        for token_logprob_map in prompt_logprobs[prompt_token_count:]:
            if not token_logprob_map:
                continue
            top_entry = next(iter(token_logprob_map.values()))
            total += getattr(top_entry, "logprob", -10.0)
            counted += 1
        score = total if counted else -10.0
        self._prefix_cache[cache_key] = score
        return score

    async def compute_rolling_one(self, prompt: str, timeout_s: float) -> float:
        cache_key = self._rolling_cache_key(prompt)
        if cache_key in self._prefix_cache:
            self._cache_hits += 1
            return self._prefix_cache[cache_key]
        self._cache_misses += 1

        if not self.use_vllm or self.async_engine is None:
            score = self._mock_rolling(prompt)
            self._prefix_cache[cache_key] = score
            return score

        sampling_params = self._sampling_cls(
            temperature=0.0, max_tokens=1, prompt_logprobs=1
        )
        request_id = f"rolling-{uuid.uuid4().hex}"

        final_output = await self._run_with_timeout(
            self._consume_generation(prompt, sampling_params, request_id),
            request_id,
            timeout_s,
        )
        if final_output is None:
            return -100.0

        prompt_logprobs = getattr(final_output, "prompt_logprobs", None)
        if not prompt_logprobs:
            return -100.0
        total = 0.0
        counted = 0
        for token_logprob_map in prompt_logprobs:
            if not token_logprob_map:
                continue
            top_entry = next(iter(token_logprob_map.values()))
            total += getattr(top_entry, "logprob", -10.0)
            counted += 1
        score = total if counted else -100.0
        self._prefix_cache[cache_key] = score
        return score

    # ------------------------------------------------------------------
    # Mock fallback for environments without vLLM
    # ------------------------------------------------------------------
    @staticmethod
    def _mock_generate(prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        max_chars = min(120, max(16, int((gen_kwargs or {}).get("max_gen_toks", 64)) * 2))
        text = prompt[-max_chars:].strip()
        return text if text else "ok"

    @staticmethod
    def _mock_loglikelihood(continuation: str) -> float:
        return -0.1 * max(1, len(continuation))

    @staticmethod
    def _mock_rolling(prompt: str) -> float:
        return -0.01 * max(1, len(prompt))

    @staticmethod
    def _ll_cache_key(prompt: str, continuation: str) -> str:
        return f"ll::{hash((prompt, continuation))}"

    @staticmethod
    def _rolling_cache_key(prompt: str) -> str:
        return f"rolling::{hash(prompt)}"

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total) if total else 0.0,
            "cache_size": len(self._prefix_cache),
        }

    async def shutdown(self) -> None:
        if self._abort_tasks:
            for task in list(self._abort_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._abort_tasks, return_exceptions=True)
        if self.use_vllm and self.async_engine is not None:
            try:
                shutdown_fn = getattr(self.async_engine, "shutdown_background_loop", None)
                if shutdown_fn is not None:
                    shutdown_fn()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Platform HTTP client
# ---------------------------------------------------------------------------
class PlatformClient:
    def __init__(self, platform_url: str, token: str, http_timeout: float) -> None:
        self.platform_url = platform_url
        self.token = token
        limits = httpx.Limits(max_keepalive_connections=32, max_connections=64)
        self.client = httpx.AsyncClient(timeout=http_timeout, limits=limits)

    async def register(self, name: str) -> bool:
        try:
            response = await self.client.post(
                f"{self.platform_url}/register",
                json={"name": name, "token": self.token},
            )
            if response.status_code == 200:
                logger.info("Registered as %s", name)
                return True
            logger.error("Registration failed with HTTP %s", response.status_code)
            return False
        except Exception as exc:
            logger.error("Registration error: %s", exc)
            return False

    async def query_task(self) -> Optional[Dict[str, Any]]:
        try:
            response = await self.client.post(
                f"{self.platform_url}/query", json={"token": self.token}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def ask_task(self, task_id: int, sla: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self.client.post(
                f"{self.platform_url}/ask",
                json={"token": self.token, "task_id": task_id, "sla": sla},
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            if payload.get("status") == "accepted":
                return payload.get("task")
            return None
        except Exception:
            return None

    async def submit_task(self, task_data: Dict[str, Any]) -> bool:
        try:
            response = await self.client.post(
                f"{self.platform_url}/submit",
                json={"user": {"name": TEAM_NAME, "token": self.token}, "msg": task_data},
            )
            return response.status_code == 200
        except Exception as exc:
            logger.error(
                "Submit error for task %s: %s",
                task_data.get("overview", {}).get("task_id"),
                exc,
            )
            return False

    async def close(self) -> None:
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------
class TaskProcessor:
    """Per-message timeout, fill-default-on-failure, ALWAYS submit."""

    def __init__(self, engine: InferenceEngine, client: PlatformClient,
                 default_msg_timeout: float) -> None:
        self.engine = engine
        self.client = client
        self.default_msg_timeout = default_msg_timeout
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.msg_timeouts = 0

    async def process_task(self, task: Dict[str, Any], task_budget_s: float) -> bool:
        task_id = task["overview"]["task_id"]
        messages = task.get("messages", [])
        started_at = time.time()
        logger.info(
            "[Task %s] start (%s messages, budget=%.1fs)",
            task_id, len(messages), task_budget_s,
        )

        # Per-message timeout: spread the task budget across messages but cap
        # each message at default_msg_timeout to keep individual hangs bounded.
        per_msg = self.default_msg_timeout
        if messages:
            spread = max(5.0, task_budget_s / len(messages))
            per_msg = min(self.default_msg_timeout, spread)

        deadline = started_at + task_budget_s

        for msg in messages:
            if time.time() > deadline:
                # Out of budget — fill remaining with defaults and still submit.
                self._fill_default(msg)
                continue

            remaining = max(2.0, deadline - time.time())
            msg_timeout = min(per_msg, remaining)
            try:
                await self._process_one_message(task_id, msg, msg_timeout)
            except Exception as exc:
                logger.warning(
                    "[Task %s] message %s raised %s; using default",
                    task_id, msg.get("ID"), exc,
                )
                self._fill_default(msg)

        # Always submit, even if every message defaulted out.
        submit_ok = await self.client.submit_task(task)
        elapsed = time.time() - started_at
        if submit_ok:
            self.tasks_completed += 1
            logger.info("[Task %s] submitted in %.2fs", task_id, elapsed)
        else:
            self.tasks_failed += 1
            logger.error("[Task %s] submit failed after %.2fs", task_id, elapsed)
        return submit_ok

    async def _process_one_message(
        self, task_id: int, msg: Dict[str, Any], timeout_s: float
    ) -> None:
        rt = msg.get("eval_request_type")
        prompt = msg.get("prompt", "") or ""

        if rt == "generate_until":
            response = await self.engine.generate_until(
                prompt, msg.get("eval_gen_kwargs") or {}, timeout_s
            )
            msg["response"] = response
            if response == "":
                self.msg_timeouts += 1

        elif rt == "loglikelihood":
            continuation = msg.get("eval_continuation") or ""
            score = await self.engine.compute_loglikelihood_one(
                prompt, continuation, timeout_s
            )
            msg["accuracy"] = score

        elif rt == "loglikelihood_rolling":
            score = await self.engine.compute_rolling_one(prompt, timeout_s)
            msg["accuracy"] = score

        else:
            self._fill_default(msg)

    @staticmethod
    def _fill_default(msg: Dict[str, Any]) -> None:
        rt = msg.get("eval_request_type")
        if rt == "generate_until":
            msg.setdefault("response", "")
        elif rt == "loglikelihood":
            msg.setdefault("accuracy", -10.0)
        elif rt == "loglikelihood_rolling":
            msg.setdefault("accuracy", -100.0)
        else:
            msg.setdefault("response", "")


# ---------------------------------------------------------------------------
# Main competition loop
# ---------------------------------------------------------------------------
async def competition_loop() -> None:
    engine = InferenceEngine(RUNTIME["model_path"], RUNTIME["abort_timeout"])
    client = PlatformClient(RUNTIME["platform_url"], TEAM_TOKEN, RUNTIME["http_timeout"])
    scheduler = TaskScheduler()
    processor = TaskProcessor(engine, client, RUNTIME["msg_timeout"])
    active_workers: set[asyncio.Task[Any]] = set()
    shutdown_event = asyncio.Event()

    # Register first; retry a few times then fail loudly.
    for attempt in range(5):
        if await client.register(TEAM_NAME):
            break
        await asyncio.sleep(2.0)
    else:
        raise RuntimeError("Failed to register after 5 attempts")

    logger.info("=" * 72)
    logger.info("poppycock client up | platform=%s", RUNTIME["platform_url"])
    logger.info(
        "model_path=%s | concurrency=%s | buffer=%s | task_budget=%.0fs | msg_budget=%.0fs",
        RUNTIME["model_path"] or "<mock>",
        RUNTIME["max_concurrent_tasks"],
        RUNTIME["max_buffered_tasks"],
        RUNTIME["task_timeout"],
        RUNTIME["msg_timeout"],
    )
    logger.info("=" * 72)

    async def worker(task_item: PrioritizedTask) -> None:
        task_id = task_item.task_id
        overview = task_item.overview
        try:
            accepted_task = await client.ask_task(
                task_id, overview.get("target_sla", "Gold")
            )
            if accepted_task is None:
                scheduler.mark_failed(task_id)
                return

            eval_timeout = float(overview.get("eval_timeout_s", RUNTIME["task_timeout"]))
            budget = max(5.0, min(eval_timeout, RUNTIME["task_timeout"]))

            success = await processor.process_task(accepted_task, budget)
            if success:
                scheduler.mark_completed(task_id)
            else:
                scheduler.mark_failed(task_id)
        except Exception as exc:
            scheduler.mark_failed(task_id)
            logger.exception("Worker for task %s crashed: %s", task_id, exc)

    async def heartbeat() -> None:
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=RUNTIME["heartbeat_interval"]
                )
                break
            except asyncio.TimeoutError:
                pass
            stats = engine.get_cache_stats()
            logger.info(
                "[hb] queue=%d active=%d done=%d completed=%d failed=%d "
                "msg_timeouts=%d cache=%d/%d (%.0f%%)",
                scheduler.queue_size,
                scheduler.active_count,
                scheduler.completed_count,
                processor.tasks_completed,
                processor.tasks_failed,
                processor.msg_timeouts,
                stats["hits"],
                stats["hits"] + stats["misses"],
                stats["hit_rate"] * 100.0,
            )

    hb_task = asyncio.create_task(heartbeat(), name="heartbeat")

    try:
        while True:
            # Reap finished workers.
            finished = {t for t in active_workers if t.done()}
            if finished:
                active_workers -= finished
                for t in finished:
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.exception("Worker task exception: %s", exc)

            # Pull new tasks if buffer has room.
            if scheduler.can_buffer_more(RUNTIME["max_buffered_tasks"]):
                overview = await client.query_task()
                if overview is not None and scheduler.add_task(overview):
                    logger.info(
                        "Discovered task %s (SLA=%s, reward=%s)",
                        overview.get("task_id"),
                        overview.get("target_sla"),
                        overview.get("target_reward"),
                    )
                else:
                    await asyncio.sleep(RUNTIME["idle_sleep"])
            else:
                await asyncio.sleep(RUNTIME["poll_interval"])

            # Dispatch as many workers as concurrency allows.
            while scheduler.has_capacity(RUNTIME["max_concurrent_tasks"]):
                next_task = scheduler.dispatch_next(RUNTIME["max_concurrent_tasks"])
                if next_task is None:
                    break
                w = asyncio.create_task(
                    worker(next_task), name=f"worker-{next_task.task_id}"
                )
                active_workers.add(w)
    finally:
        shutdown_event.set()
        if not hb_task.done():
            hb_task.cancel()
        if active_workers:
            await asyncio.gather(*active_workers, return_exceptions=True)
        await client.close()
        await engine.shutdown()
        logger.info("Cache stats: %s", engine.get_cache_stats())
        logger.info(
            "Final: completed=%s failed=%s msg_timeouts=%s",
            processor.tasks_completed,
            processor.tasks_failed,
            processor.msg_timeouts,
        )


def main() -> None:
    try:
        asyncio.run(competition_loop())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
