"""Data types, named and shaped as in Thinking Machines' Tinker SDK, so a training
loop written for one reads the same on the other."""
from dataclasses import dataclass, field


def _flat(x):
    if x is None:
        return None
    if isinstance(x, TensorData):
        return list(x.data)
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for v in x:
        out.extend(_flat(v) if isinstance(v, (list, tuple)) else [v])
    return out


@dataclass
class TensorData:
    data: list
    dtype: str = "float32"
    shape: list | None = None

    @classmethod
    def from_numpy(cls, a):
        return cls(a.reshape(-1).tolist(), str(a.dtype), list(a.shape))

    @classmethod
    def from_torch(cls, t):
        t = t.detach().cpu()
        return cls(t.reshape(-1).tolist(), str(t.dtype).replace("torch.", ""), list(t.shape))

    def tolist(self):
        return list(self.data)

    def to_torch(self):
        import torch

        t = torch.tensor(self.data)
        return t.reshape(self.shape) if self.shape else t

    def to_numpy(self):
        import numpy as np

        a = np.array(self.data)
        return a.reshape(self.shape) if self.shape else a


@dataclass
class EncodedTextChunk:
    tokens: list
    type: str = "encoded_text"

    @property
    def length(self):
        return len(self.tokens)


@dataclass
class ModelInput:
    chunks: list = field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens):
        return cls([EncodedTextChunk(list(map(int, _flat(tokens))))])

    @classmethod
    def empty(cls):
        return cls([])

    def to_ints(self):
        out = []
        for c in self.chunks:
            out.extend(c.tokens)
        return out

    def append_int(self, t):
        return ModelInput(self.chunks + [EncodedTextChunk([int(t)])])

    def append(self, chunk):
        return ModelInput(self.chunks + [chunk])

    @property
    def length(self):
        return sum(len(c.tokens) for c in self.chunks)

    def to_json(self):
        return {"chunks": [{"type": "encoded_text", "tokens": list(map(int, c.tokens))} for c in self.chunks]}


@dataclass
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict = field(default_factory=dict)

    def to_json(self):
        mi = self.model_input if isinstance(self.model_input, ModelInput) else ModelInput.from_ints(self.model_input)
        return {"model_input": mi.to_json(), "loss_fn_inputs": {k: _flat(v) for k, v in self.loss_fn_inputs.items()}}


@dataclass
class AdamParams:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


@dataclass
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: list = field(default_factory=list)
    seed: int | None = None


@dataclass
class ForwardBackwardOutput:
    metrics: dict
    loss_fn_outputs: list


@dataclass
class OptimStepResponse:
    metrics: dict


@dataclass
class SampledSequence:
    tokens: list
    logprobs: list
    stop_reason: str


@dataclass
class SampleResponse:
    sequences: list
    prompt_logprobs: list | None = None


@dataclass
class SaveWeightsResponse:
    path: str
    name: str = ""


@dataclass
class SaveStateResponse:
    path: str
    name: str = ""


@dataclass
class LoraConfig:
    rank: int = 32
    alpha: float = 32.0
    seed: int = 0
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False

