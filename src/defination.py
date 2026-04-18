import json
from enum import Enum, auto
from typing import List, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from typing import Optional

@dataclass(frozen=True)
class SLA:
    """
    Service Level Agreement specification for inference latency.

    All values are defined in seconds unless otherwise specified.

    Attributes:
        ttft_avg (float):
            Average Time-To-First-Token (TTFT).
            Measures how long it takes to produce the first output token
            after receiving a request.

        tpot_p50 (float):
            50th percentile (median) Time-Per-Output-Token (TPOT).
            Represents the typical per-token decoding latency.

        tpot_p75 (float):
            75th percentile Time-Per-Output-Token (TPOT).
            Captures tail latency under moderate load.
    """
    ttft_avg: float = 0.0
    tpot_p50: float = 0.0
    tpot_p75: float = 0.0


@dataclass(frozen=True)
class SamplingParam:
    """
    Sampling configuration for autoregressive decoding.

    Defines stochasticity, truncation, and repetition penalties
    used during token generation.

    Attributes:
        temperature (float):
            Softmax temperature. Higher values increase randomness.
            0.0 implies deterministic decoding (argmax).

        top_p (float):
            Nucleus sampling threshold. Tokens are sampled from the
            smallest set whose cumulative probability >= top_p.

        top_k (int):
            Top-k sampling limit. Only the k most probable tokens
            are considered at each decoding step.

        penalty_repetition (float):
            Repetition penalty applied to previously generated tokens.
            Values > 1.0 discourage repeated tokens.

        penalty_frequency (float):
            Frequency-based penalty proportional to token occurrence count.

        penalty_presence (float):
            Presence penalty applied once a token has appeared,
            regardless of its frequency.
    """
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    penalty_repetition: float = 0.0
    penalty_frequency: float = 0.0
    penalty_presence: float = 0.0


@dataclass(frozen=True)
class LenSpec:
    """
    Length specification used for workload classification and
    output constraint enforcement.

    This spec serves two different purposes:

    1. Prompt length range [prompt_min, prompt_max):
       Defines the token count interval used to classify incoming
       requests into a specific workload tier. This is a bucketization
       rule and does NOT enforce strict rejection.

    2. Output length range [output_min, output_max):
       Defines the allowed generation length range.
       The model must generate tokens within this interval.
       This is a hard constraint enforced at decoding time.

    Attributes:
        prompt_min (int):
            Inclusive lower bound for prompt token length classification.

        prompt_max (int):
            Exclusive upper bound for prompt token length classification.

        output_min (int):
            Inclusive lower bound for generated token length.

        output_max (int):
            Exclusive upper bound for generated token length.
    """
    prompt_min: int
    prompt_max: int
    output_min: int
    output_max: int

class ConfigRegistry:
    """
    A registry/factory that loads configuration from a JSON file
    and materializes SLA, SamplingParam, and LenSpec instances.

    Instances can be retrieved by key.
    """

    def __init__(self, json_path: str):
        """
        Initialize registry from a JSON configuration file.

        :param json_path: Path to configuration JSON file.
        """
        with open(json_path, "r") as f:
            config = json.load(f)

        self._sla: Dict[str, SLA] = {}
        self._sampling: Dict[str, SamplingParam] = {}
        self._lenspec: Dict[str, LenSpec] = {}

        self._load_sla(config.get("SLA", {}))
        self._load_sampling(config.get("SamplingParam", {}))
        self._load_lenspec(config.get("LenSpec", {}))

    # ------------------------------------------------------------------
    # Private loading helpers
    # ------------------------------------------------------------------

    def _load_sla(self, sla_config: Dict):
        for name, values in sla_config.items():
            self._sla[name] = SLA(**values)

    def _load_sampling(self, sampling_config: Dict):
        for name, values in sampling_config.items():
            self._sampling[name] = SamplingParam(**values)

    def _load_lenspec(self, lenspec_config: Dict):
        for name, bounds in lenspec_config.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"LenSpec '{name}' must be [min, max].")

            min_len, max_len = bounds

            self._lenspec[name] = LenSpec(
                prompt_min=min_len,
                prompt_max=max_len,
                output_min=min_len,
                output_max=max_len,
            )

    # ------------------------------------------------------------------
    # Public access APIs
    # ------------------------------------------------------------------

    def get_sla(self, name: str = None) -> SLA:
        """Return SLA instance by name."""
        return self._sla[name] if name is not None else self._sla

    def get_sampling(self, name: str = None) -> SamplingParam:
        """Return SamplingParam instance by name."""
        return self._sampling[name] if name is not None else self._sampling

    def get_lenspec(self, name: str = None) -> LenSpec:
        """Return LenSpec instance by name."""
        return self._lenspec[name] if name is not None else self._lenspec

    def list_sla(self):
        """List available SLA keys."""
        return list(self._sla.keys())

    def list_sampling(self):
        """List available SamplingParam keys."""
        return list(self._sampling.keys())

    def list_lenspec(self):
        """List available LenSpec keys."""
        return list(self._lenspec.keys())

sizes = ["Small", "Medium", "Large", "XL", "XXL"]
levels = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Stellar", "Glorious", "Supreme"]

# ---------------------------------------------------------------------
# Message schema
# ---------------------------------------------------------------------
class Message(BaseModel):
    """
    Pydantic model describing incoming JSON payload.

    Fields:
    - ID: message index within the task
    - prompt: input text (set by platform)
    - response: generated text (filled by contestant for generate_until)
    - accuracy: log-probability (filled by contestant for loglikelihood)
    - eval_gen_kwargs: generation parameters (temperature, stop tokens, etc.)
    - eval_continuation: continuation text (for loglikelihood tasks)
    - eval_request_type: "generate_until" | "loglikelihood" | "loglikelihood_rolling"
    """
    ID: int

    # Exactly one of the following three must be present
    prompt: Optional[str] = None
    response: Optional[str] = None
    accuracy: Optional[float] = None

    # ── Task metadata (set by platform) ──
    eval_gen_kwargs: Optional[dict] = None
    eval_continuation: Optional[str] = None
    eval_request_type: Optional[str] = None

    @model_validator(mode="after")
    def check_one_of_prompt_response_accuracy(self):
        fields = [
            self.prompt is not None,
            self.response is not None,
            self.accuracy is not None,
        ]
        if sum(fields) == 0:
            raise ValueError(
                "At least one of 'prompt', 'response', or 'accuracy' must be provided"
            )
        return self

# ---------------------------------------------------------------------
# User & request schemas
# ---------------------------------------------------------------------
class User(BaseModel):
    """Client identity."""
    name: str
    token: str

class ParamQuery(BaseModel):
    """Query request payload."""
    token: str

class ParamAsk(BaseModel):
    """Task claim (bid) request."""
    token: str
    task_id: int
    sla: str

# SLALevel represents a relative quality-of-service tier.
#
# The actual numeric constraints (e.g., P99 latency threshold,
# success rate requirement) are defined externally and may
# change across competition stages.
#
# This allows dynamic remapping without breaking bids.

class TaskOverview(BaseModel):
    """
    High-level specification of a competition task.
    """
    task_id: int

    # Minimum SLA level required to bid
    target_sla: str = "Gold"

    # Maximum reward for first successful completion
    target_reward: float = 0.0

    # Maximum number of teams that can win this task
    max_winners: int = 5

    # ── Task metadata (set by platform) ──
    eval_batch_id: Optional[str] = None
    eval_task_name: Optional[str] = None
    eval_request_type: Optional[str] = None
    eval_sampling_param: Optional[str] = None
    eval_timeout_s: Optional[int] = None

class Task(BaseModel):
    """
    Full task payload (returned after successful ask).

    Contains:
    - Concrete messages to process
    - Associated overview metadata
    """
    messages: List[Message]
    overview: TaskOverview