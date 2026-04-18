"""
TaskBuilder: builds tasks from messages by bucketing prompts by length.

Generates three request types matching the eval path:
  - generate_until: prompt + eval_gen_kwargs (with sampling params)
  - loglikelihood: prompt + eval_continuation
  - loglikelihood_rolling: prompt only

Variance is assigned randomly per task and determines eval_gen_kwargs
for generate_until messages.
"""

import bisect
import random
import time
from collections import deque
from typing import List, Optional

from defination import TaskOverview, Task, Message


class Bucket:
    def __init__(self, name):
        self.name = name
        self.queue = deque()
        self.last_push_time = time.time()

    def push(self, msg):
        self.queue.append(msg)
        self.last_push_time = time.time()

    def pop(self):
        return self.queue.popleft()

    def __len__(self):
        return len(self.queue)


_VARIANCE_LEVELS = ["Deterministic", "Normal", "HighEntropy", "ExtremePenalty"]
_VARIANCE_WEIGHTS = [20, 40, 30, 10]

# Request type distribution within a task
_REQUEST_TYPES = ["generate_until", "loglikelihood", "loglikelihood", "loglikelihood_rolling"]

# Fake continuations for loglikelihood tasks
_FAKE_CONTINUATIONS = [
    " Yes", " No", " True", " False",
    " A", " B", " C", " D",
    " is correct", " is incorrect",
    " positive", " negative",
]


class TaskBuilder:
    """
    Message -> Bucket -> Task

    Messages are bucketed by prompt length.
    Tasks are formed by sampling across buckets (one per bucket).
    Each message gets a random request type and appropriate metadata.
    """

    def __init__(self, rank, registry, max_wait_time: float = 0.5):
        self.registry = registry
        self._bounds, self._labels = self._build_index(registry._lenspec)
        self.buckets = [Bucket(name) for name in self._labels]
        self.task_id = rank * 1_000_000_000
        self.max_wait_time = max_wait_time

        # Build sampling param dicts from registry for eval_gen_kwargs
        self._sampling_params = {}
        for name in registry.list_sampling():
            sp = registry.get_sampling(name)
            self._sampling_params[name] = {
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "repetition_penalty": sp.penalty_repetition,
                "frequency_penalty": sp.penalty_frequency,
                "presence_penalty": sp.penalty_presence,
            }

    def _build_index(self, spec_dict):
        items = sorted(spec_dict.items(), key=lambda x: x[1].prompt_min)
        bounds = []
        labels = []
        prev_max = None
        for label, spec in items:
            if spec.prompt_min >= spec.prompt_max:
                raise ValueError(f"Invalid interval for {label}")
            if prev_max is not None and spec.prompt_min < prev_max:
                raise ValueError(f"Overlapping intervals")
            bounds.append(spec.prompt_min)
            labels.append(label)
            prev_max = spec.prompt_max
        return bounds, labels

    def put(self, msg):
        prompt_len = len(msg.prompt) if msg.prompt else 0
        idx = bisect.bisect_right(self._bounds, prompt_len) - 1
        idx = max(0, min(idx, len(self.buckets) - 1))
        self.buckets[idx].push(msg)

    def maybe_build(self) -> Optional[Task]:
        if self._ready_strict():
            return self._build_task(allow_partial=False)
        if self._ready_timeout():
            return self._build_task(allow_partial=True)
        return None

    def _ready_strict(self):
        return all(len(b) > 0 for b in self.buckets)

    def _ready_timeout(self):
        now = time.time()
        return any(len(b) > 0 and (now - b.last_push_time > self.max_wait_time)
                   for b in self.buckets)

    def _build_task(self, allow_partial=False) -> Optional[Task]:
        selected: List[Message] = []
        for bucket in self.buckets:
            if len(bucket) > 0:
                selected.append(bucket.pop())
            elif not allow_partial:
                return None

        if not selected:
            return None

        self.task_id += 1

        # Pick variance randomly
        variance = random.choices(_VARIANCE_LEVELS, weights=_VARIANCE_WEIGHTS, k=1)[0]
        sp_dict = self._sampling_params.get(variance, {})

        # Assign request types and metadata to each message
        for i, msg in enumerate(selected):
            rt = _REQUEST_TYPES[i % len(_REQUEST_TYPES)]
            msg.eval_request_type = rt

            if rt == "generate_until":
                # Provide sampling params + generation config via eval_gen_kwargs
                msg.eval_gen_kwargs = {
                    **sp_dict,
                    "max_gen_toks": random.choice([64, 128, 256]),
                    "until": random.choice([
                        ["\n"], ["\n\n"], ["</s>"], ["."],
                    ]),
                }
            elif rt == "loglikelihood":
                # Provide continuation text for likelihood scoring
                msg.eval_continuation = random.choice(_FAKE_CONTINUATIONS)
            # loglikelihood_rolling: only prompt needed, no extra fields

        reward = self._select_reward(selected, variance)
        overview = TaskOverview(
            task_id=self.task_id,
            target_sla="Gold",
            target_reward=reward,
            max_winners=1,
        )
        return Task(messages=selected, overview=overview)

    def _select_reward(self, msgs, variance):
        base = sum(len(m.prompt) for m in msgs if m.prompt)
        coef = {"Deterministic": 0.7, "Normal": 1.0, "HighEntropy": 1.3, "ExtremePenalty": 1.6}
        return int(base * coef.get(variance, 1.0))
