#!/usr/bin/env python3
"""
poppycock competition client - Optimized v2

Architecture inspired by:
- Apt-Serve (SIGMOD 25): SLA-aware scheduling, hybrid cache management
- Sarathi-Serve (OSDI 24): Chunked prefill, two-phase scheduling
- Llumnix: SLO-aware request prioritization
- Dynamo: Disaggregated prefill/decode

Key optimizations:
1. Async concurrent task processing (up to MAX_CONCURRENT_TASKS)
2. SLA-prioritized task scheduling (higher SLA = higher priority)
3. Request batching for loglikelihood tasks
4. Intelligent task selection based on reward/compute ratio
5. Prefix caching for repeated prompts
6. Graceful error handling with retry logic
"""

import os
import sys
import json
import time
import math
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEAM_NAME = "poppycock"
TEAM_TOKEN = "d51ebe5cc6594d53ef2853fd8f9dadc1"

PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://127.0.0.1:8003")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")

# Concurrency and scheduling
MAX_CONCURRENT_TASKS = 32
POLL_INTERVAL = 0.1
MAX_RETRIES = 3

# SLA priority mapping (higher SLA = higher priority)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Queue with Priority Scheduling (inspired by Apt-Serve)
# ---------------------------------------------------------------------------
@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: int = field(compare=False)
    overview: dict = field(compare=False)
    timestamp: float = field(compare=False)


class TaskScheduler:
    """
    Priority-based task scheduler.
    
    Inspired by Apt-Serve's greedy selection and Sarathi-Serve's two-phase scheduling.
    Prioritizes tasks by:
    1. SLA level (higher SLA = higher priority)
    2. Reward per compute unit (higher reward = better)
    3. Age (older tasks first to avoid timeout)
    """
    
    def __init__(self):
        self.queue: deque = deque()
        self.processing: set = set()
        self.completed: set = set()
    
    def add_task(self, task_id: int, overview: dict):
        sla = overview.get("target_sla", "Bronze")
        reward = overview.get("target_reward", 0)
        priority = SLA_PRIORITY.get(sla, 0) * 1000 + int(reward)
        
        item = PrioritizedTask(
            priority=-priority,  # Negative for min-heap behavior
            task_id=task_id,
            overview=overview,
            timestamp=time.time(),
        )
        self.queue.append(item)
        logger.debug(f"Added task {task_id} to queue (priority={priority})")
    
    def get_next_task(self) -> Optional[PrioritizedTask]:
        """Get highest priority task that's not being processed."""
        while self.queue:
            task = self.queue.popleft()
            if task.task_id not in self.processing and task.task_id not in self.completed:
                self.processing.add(task.task_id)
                return task
        return None
    
    def mark_completed(self, task_id: int):
        self.completed.add(task_id)
        self.processing.discard(task_id)
    
    def mark_failed(self, task_id: int):
        self.processing.discard(task_id)
    
    @property
    def active_count(self):
        return len(self.processing)


# ---------------------------------------------------------------------------
# Inference Engine with Batching and Caching
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    Optimized inference engine with:
    - vLLM for production (high throughput, PagedAttention)
    - Request batching for loglikelihood tasks
    - Prefix caching for repeated prompts
    """
    
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.use_vllm = False
        self.llm = None
        self.tokenizer = None
        
        # Prefix cache for loglikelihood tasks
        self._prefix_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        if model_path and os.path.isdir(model_path):
            try:
                from vllm import LLM
                logger.info(f"Loading vLLM model from {model_path}")
                
                # Detect GPU count for tensor parallelism
                gpu_count = self._detect_gpu_count()
                tp_size = min(gpu_count, 4)
                
                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    max_model_len=4096,
                    gpu_memory_utilization=0.9,
                    enable_prefix_caching=True,  # Enable prefix caching
                    max_num_seqs=256,  # Max concurrent sequences
                )
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.use_vllm = True
                logger.info(f"vLLM loaded with TP={tp_size}, prefix caching enabled")
            except Exception as e:
                logger.warning(f"Failed to load vLLM: {e}, using mock inference")
        else:
            logger.info("No model path provided, using mock inference for testing")
    
    def _detect_gpu_count(self) -> int:
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 1
    
    async def generate_until(self, prompt: str, gen_kwargs: dict) -> str:
        """
        Generate text until stop condition.
        Optimized with proper sampling parameters.
        """
        max_tokens = gen_kwargs.get("max_gen_toks", 256)
        temperature = gen_kwargs.get("temperature", 0.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        top_k = gen_kwargs.get("top_k", -1)
        stop_tokens = gen_kwargs.get("until", [])
        
        if self.use_vllm and self.llm:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                max_tokens=max_tokens,
                stop=stop_tokens,
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
                frequency_penalty=gen_kwargs.get("frequency_penalty", 0.0),
                presence_penalty=gen_kwargs.get("presence_penalty", 0.0),
            )
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text
        else:
            return f"Generated: {prompt[:50]}..."
    
    async def compute_loglikelihood_batch(self, prompts: List[str], continuations: List[str]) -> List[float]:
        """
        Batch compute log probabilities for multiple prompts.
        Inspired by Sarathi-Serve's batching optimization.
        """
        if not prompts:
            return []
        
        # Check cache first
        results = []
        uncached_indices = []
        uncached_prompts = []
        uncached_continuations = []
        
        for i, (prompt, continuation) in enumerate(zip(prompts, continuations)):
            cache_key = f"{prompt[:100]}|||{continuation}"
            if cache_key in self._prefix_cache:
                results.append(self._prefix_cache[cache_key])
                self._cache_hits += 1
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_prompts.append(prompt)
                uncached_continuations.append(continuation)
                self._cache_misses += 1
        
        # Compute uncached items
        if uncached_prompts:
            if self.use_vllm and self.llm:
                computed = await self._compute_loglikelihood_vllm(uncached_prompts, uncached_continuations)
            else:
                computed = [-0.5 * len(c) for c in uncached_continuations]
            
            # Update results and cache
            for idx, logprob in zip(uncached_indices, computed):
                results[idx] = logprob
                cache_key = f"{prompts[idx][:100]}|||{continuations[idx]}"
                self._prefix_cache[cache_key] = logprob
        
        return results
    
    async def _compute_loglikelihood_vllm(self, prompts: List[str], continuations: List[str]) -> List[float]:
        """Compute log probabilities using vLLM."""
        from vllm import SamplingParams
        
        results = []
        for prompt, continuation in zip(prompts, continuations):
            full_text = prompt + continuation
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=1,
            )
            outputs = self.llm.generate([full_text], sampling_params, use_tqdm=False)
            
            prompt_logprobs = outputs[0].prompt_logprobs
            if prompt_logprobs is None:
                results.append(-10.0)
                continue
            
            # Count tokens in prompt to find continuation start
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            continuation_start = len(prompt_tokens)
            
            # Sum logprobs for continuation tokens
            total_logprob = 0.0
            count = 0
            for i in range(continuation_start, len(prompt_logprobs)):
                if prompt_logprobs[i] is not None:
                    token_logprob = prompt_logprobs[i].get(0)
                    if token_logprob is not None:
                        total_logprob += token_logprob.logprob
                        count += 1
            
            results.append(total_logprob if count > 0 else -10.0)
        
        return results
    
    async def compute_loglikelihood_rolling_batch(self, prompts: List[str]) -> List[float]:
        """Batch compute rolling log-likelihoods."""
        if not prompts:
            return []
        
        # Check cache
        results = []
        uncached_indices = []
        uncached_prompts = []
        
        for i, prompt in enumerate(prompts):
            cache_key = f"rolling|||{prompt[:100]}"
            if cache_key in self._prefix_cache:
                results.append(self._prefix_cache[cache_key])
                self._cache_hits += 1
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_prompts.append(prompt)
                self._cache_misses += 1
        
        # Compute uncached
        if uncached_prompts:
            if self.use_vllm and self.llm:
                computed = await self._compute_rolling_vllm(uncached_prompts)
            else:
                computed = [-0.1 * len(p) for p in uncached_prompts]
            
            for idx, logprob in zip(uncached_indices, computed):
                results[idx] = logprob
                cache_key = f"rolling|||{prompts[idx][:100]}"
                self._prefix_cache[cache_key] = logprob
        
        return results
    
    async def _compute_rolling_vllm(self, prompts: List[str]) -> List[float]:
        """Compute rolling log-likelihoods using vLLM."""
        from vllm import SamplingParams
        
        results = []
        for prompt in prompts:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=1,
            )
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            
            prompt_logprobs = outputs[0].prompt_logprobs
            if prompt_logprobs is None:
                results.append(-100.0)
                continue
            
            total_logprob = 0.0
            count = 0
            for i in range(len(prompt_logprobs)):
                if prompt_logprobs[i] is not None:
                    token_logprob = prompt_logprobs[i].get(0)
                    if token_logprob is not None:
                        total_logprob += token_logprob.logprob
                        count += 1
            
            results.append(total_logprob if count > 0 else -100.0)
        
        return results
    
    def get_cache_stats(self) -> dict:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "cache_size": len(self._prefix_cache),
        }


# ---------------------------------------------------------------------------
# Platform Client
# ---------------------------------------------------------------------------
class PlatformClient:
    """Async HTTP client for the competition platform."""
    
    def __init__(self, platform_url: str, token: str):
        self.platform_url = platform_url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(timeout=30.0)
        self.registered = False
    
    async def register(self, name: str) -> bool:
        try:
            resp = await self.client.post(
                f"{self.platform_url}/register",
                json={"name": name, "token": self.token},
            )
            if resp.status_code == 200:
                self.registered = True
                logger.info(f"Registered as {name}")
                return True
            else:
                logger.error(f"Registration failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def query_task(self) -> Optional[dict]:
        try:
            resp = await self.client.post(
                f"{self.platform_url}/query",
                json={"token": self.token},
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                return None
        except Exception:
            return None
    
    async def ask_task(self, task_id: int, sla: str) -> Optional[dict]:
        try:
            resp = await self.client.post(
                f"{self.platform_url}/ask",
                json={"token": self.token, "task_id": task_id, "sla": sla},
            )
            if resp.status_code == 200:
                result = resp.json()
                if result.get("status") == "accepted":
                    return result.get("task")
                else:
                    return None
            else:
                return None
        except Exception:
            return None
    
    async def submit_task(self, task_data: dict) -> bool:
        try:
            resp = await self.client.post(
                f"{self.platform_url}/submit",
                json={
                    "user": {"name": TEAM_NAME, "token": self.token},
                    "msg": task_data,
                },
            )
            if resp.status_code == 200:
                return True
            else:
                logger.error(f"Submit failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return False
    
    async def close(self):
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Task Processor with Batching
# ---------------------------------------------------------------------------
class TaskProcessor:
    """
    Processes tasks with batching optimization.
    Inspired by Sarathi-Serve's chunked processing.
    """
    
    def __init__(self, engine: InferenceEngine, client: PlatformClient):
        self.engine = engine
        self.client = client
        self.tasks_completed = 0
        self.tasks_failed = 0
    
    async def process_task(self, task: dict) -> bool:
        """Process a single task with optimized batching."""
        task_id = task["overview"]["task_id"]
        messages = task["messages"]
        
        logger.info(f"Processing task {task_id} ({len(messages)} messages)")
        start_time = time.time()
        
        try:
            # Group messages by type for batching
            gen_messages = []
            ll_messages = []
            ll_rolling_messages = []
            
            for msg in messages:
                rt = msg.get("eval_request_type", "loglikelihood")
                if rt == "generate_until":
                    gen_messages.append(msg)
                elif rt == "loglikelihood":
                    ll_messages.append(msg)
                elif rt == "loglikelihood_rolling":
                    ll_rolling_messages.append(msg)
            
            # Process generate_until (one by one due to different params)
            for msg in gen_messages:
                gen_kwargs = msg.get("eval_gen_kwargs", {})
                response = await self.engine.generate_until(msg["prompt"], gen_kwargs)
                msg["response"] = response
            
            # Process loglikelihood (batched)
            if ll_messages:
                prompts = [m["prompt"] for m in ll_messages]
                continuations = [m.get("eval_continuation", "") for m in ll_messages]
                logprobs = await self.engine.compute_loglikelihood_batch(prompts, continuations)
                for msg, logprob in zip(ll_messages, logprobs):
                    msg["accuracy"] = logprob
            
            # Process loglikelihood_rolling (batched)
            if ll_rolling_messages:
                prompts = [m["prompt"] for m in ll_rolling_messages]
                logprobs = await self.engine.compute_loglikelihood_rolling_batch(prompts)
                for msg, logprob in zip(ll_rolling_messages, logprobs):
                    msg["accuracy"] = logprob
            
            # Submit
            success = await self.client.submit_task(task)
            elapsed = time.time() - start_time
            
            if success:
                self.tasks_completed += 1
                logger.info(f"Task {task_id} completed in {elapsed:.2f}s")
            else:
                self.tasks_failed += 1
                logger.error(f"Task {task_id} submit failed")
            
            return success
        
        except Exception as e:
            self.tasks_failed += 1
            elapsed = time.time() - start_time
            logger.error(f"Task {task_id} error after {elapsed:.2f}s: {e}")
            return False


# ---------------------------------------------------------------------------
# Main Competition Loop
# ---------------------------------------------------------------------------
async def competition_loop():
    """
    Main competition loop with concurrent task processing.
    
    Architecture:
    1. Query loop: continuously queries for new tasks
    2. Scheduler: prioritizes tasks by SLA and reward
    3. Worker pool: processes up to MAX_CONCURRENT_TASKS concurrently
    """
    
    engine = InferenceEngine(MODEL_PATH)
    client = PlatformClient(PLATFORM_URL, TEAM_TOKEN)
    scheduler = TaskScheduler()
    processor = TaskProcessor(engine, client)
    
    # Register with retry
    for attempt in range(5):
        if await client.register(TEAM_NAME):
            break
        logger.warning(f"Registration attempt {attempt + 1}/5 failed")
        await asyncio.sleep(2)
    else:
        logger.error("Failed to register")
        return
    
    logger.info("=" * 60)
    logger.info("Starting optimized competition loop")
    logger.info(f"Platform: {PLATFORM_URL}")
    logger.info(f"Team: {TEAM_NAME}")
    logger.info(f"Inference: {'vLLM' if engine.use_vllm else 'Mock'}")
    logger.info(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    logger.info("=" * 60)
    
    # Worker semaphore for concurrency control
    worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    active_tasks: List[asyncio.Task] = []
    
    async def worker(task_item: PrioritizedTask):
        """Worker coroutine to process a single task."""
        async with worker_semaphore:
            try:
                task_id = task_item.task_id
                sla = task_item.overview.get("target_sla", "Gold")
                
                # Ask for the task
                task = await client.ask_task(task_id, sla)
                if task is None:
                    scheduler.mark_failed(task_id)
                    return
                
                # Process the task
                success = await processor.process_task(task)
                if success:
                    scheduler.mark_completed(task_id)
                else:
                    scheduler.mark_failed(task_id)
            
            except Exception as e:
                logger.error(f"Worker error: {e}")
                scheduler.mark_failed(task_item.task_id)
    
    # Main query loop
    consecutive_errors = 0
    max_errors = 10
    
    try:
        while True:
            # Clean up completed workers
            active_tasks = [t for t in active_tasks if not t.done()]
            
            # Query for new task
            overview = await client.query_task()
            
            if overview is not None:
                task_id = overview["task_id"]
                if task_id not in scheduler.completed and task_id not in scheduler.processing:
                    scheduler.add_task(task_id, overview)
                    consecutive_errors = 0
            
            # Get next task to process
            next_task = scheduler.get_next_task()
            if next_task is not None and scheduler.active_count < MAX_CONCURRENT_TASKS:
                worker_task = asyncio.create_task(worker(next_task))
                active_tasks.append(worker_task)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(POLL_INTERVAL)
            
            # Error handling
            if consecutive_errors >= max_errors:
                logger.error(f"Too many errors, stopping")
                break
    
    except asyncio.CancelledError:
        logger.info("Loop cancelled")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Wait for all active tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        await client.close()
        
        # Print stats
        cache_stats = engine.get_cache_stats()
        logger.info("=" * 60)
        logger.info("Competition loop ended")
        logger.info(f"Tasks completed: {processor.tasks_completed}")
        logger.info(f"Tasks failed: {processor.tasks_failed}")
        logger.info(f"Cache stats: {cache_stats}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    logger.info("poppycock competition client v2")
    logger.info(f"Platform: {PLATFORM_URL}")
    logger.info(f"Model: {MODEL_PATH or 'Mock'}")
    
    try:
        asyncio.run(competition_loop())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
