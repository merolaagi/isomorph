"""isotinker: a Tinker-style client for Isomorph's local training service.

    import isotinker as tinker
    from isotinker import types

    sc = tinker.ServiceClient()                                   # finds the Isomorph server on this Mac
    tc = sc.create_lora_training_client(base_model="microopus:words", rank=8)
    tok = tc.get_tokenizer()
    ids = tok.encode("the cat sat on the mat .")
    d = types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                    loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)})
    for step in range(20):
        fb = tc.forward_backward([d], "cross_entropy")
        tc.optim_step(types.AdamParams(learning_rate=1e-3))
        print(step, fb.result().metrics["loss:sum"])
    sampler = tc.save_weights_and_get_sampling_client(name="my-first-adapter")
    out = sampler.sample(types.ModelInput.from_ints(tok.encode("the")), num_samples=4,
                         sampling_params=types.SamplingParams(max_tokens=8)).result()

Calls return futures (.result()); calls on one client run in the order made, so
forward_backward and optim_step can be issued back to back as in Tinker. Each
also has an _async twin to await. Where Tinker runs on its cluster, this runs
on the Isomorph server on this Mac (its GPU), so models must be small.

The server is found from ISOMORPH_URL, else data/server.json (written by the
server when it starts), else http://127.0.0.1:47821. The password, if one is
set, from ISOMORPH_PASSWORD or data/settings.json.
"""
import asyncio
import base64
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import types

__all__ = ["ServiceClient", "TrainingClient", "SamplingClient", "types", "IsomorphError"]
ROOT = Path(__file__).resolve().parents[2]


class IsomorphError(RuntimeError):
    pass


def _url():
    u = os.environ.get("ISOMORPH_URL")
    if u:
        return u.rstrip("/")
    f = ROOT / "data" / "server.json"
    if f.exists():
        try:
            return json.loads(f.read_text())["url"].rstrip("/")
        except Exception:
            pass
    return f"http://127.0.0.1:{os.environ.get('ISOMORPH_PORT', '47821')}"


def _password():
    pw = os.environ.get("ISOMORPH_PASSWORD")
    if pw:
        return pw
    f = ROOT / "data" / "settings.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("password")
        except Exception:
            return None
    return None


class _Http:
    def __init__(self, url=None, password=None, timeout=3600):
        self.url, self.timeout = (url or _url()).rstrip("/"), timeout
        pw = password if password is not None else _password()
        self.auth = "Basic " + base64.b64encode(f"isomorph:{pw}".encode()).decode() if pw else None

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        if self.auth:
            req.add_header("Authorization", self.auth)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("detail")
            except Exception:
                msg = str(e)
            raise IsomorphError(f"{e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise IsomorphError(f"Cannot reach Isomorph at {self.url} ({e.reason}). Is it running? Start it with ./run.sh") from None


class _Future:
    """What Tinker calls an APIFuture: .result() waits; await works too."""

    def __init__(self, fut):
        self._f = fut

    def result(self, timeout=None):
        return self._f.result(timeout)

    def done(self):
        return self._f.done()

    def __await__(self):
        return asyncio.wrap_future(self._f).__await__()


def _ser(data):
    out = []
    for d in data:
        if isinstance(d, types.Datum):
            out.append(d.to_json())
        elif isinstance(d, dict):
            mi = d.get("model_input")
            if isinstance(mi, types.ModelInput):
                d = {**d, "model_input": mi.to_json()}
            d = {**d, "loss_fn_inputs": {k: types._flat(v) for k, v in (d.get("loss_fn_inputs") or {}).items()}}
            out.append(d)
        else:
            raise TypeError("data must be a list of types.Datum (or dicts of the same shape).")
    return out


def _ints(prompt):
    if isinstance(prompt, types.ModelInput):
        return prompt.to_ints()
    return [int(t) for t in types._flat(prompt)]


class Tokenizer:
    """The base model's tokenizer, run by the server (so Micro Opus's word tokenizer works too)."""

    def __init__(self, http, base_model):
        self._h, self.base_model = http, base_model
        info = http.call("POST", "/api/tune/v1/tokenizer", {"base_model": base_model, "text": ""})
        self.vocab_size = info["vocab_size"]
        self.eos_token_id = info.get("eos_token_id")
        self.bos_token_id = info.get("bos_token_id")

    def encode(self, text, add_special_tokens=True):
        return self._h.call("POST", "/api/tune/v1/tokenizer", {"base_model": self.base_model, "text": text,
                                                                "add_special_tokens": add_special_tokens})["tokens"]

    def decode(self, tokens, **_):
        return self._h.call("POST", "/api/tune/v1/tokenizer", {"base_model": self.base_model, "tokens": [int(t) for t in tokens]})["text"]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **_):
        ids = self._h.call("POST", "/api/tune/v1/tokenizer", {"base_model": self.base_model, "messages": messages,
                                                               "add_generation_prompt": add_generation_prompt})["tokens"]
        return ids if tokenize else self.decode(ids)

    def __len__(self):
        return self.vocab_size


class SamplingClient:
    def __init__(self, http, base_model=None, model_path=None, session=None, pool=None, tokenizer_model=None):
        self._h = http
        self.base_model, self.model_path, self.session = base_model, model_path, session
        self._pool = pool or ThreadPoolExecutor(1)
        self._tok_model = tokenizer_model or base_model or (f"tune:{model_path.rstrip('/').rsplit('/', 1)[-1]}" if model_path else None)

    def _ref(self):
        return {"base_model": self.base_model, "model_path": self.model_path, "session": self.session}

    def sample(self, prompt, num_samples=1, sampling_params=None, include_prompt_logprobs=False, **_):
        sp = sampling_params or types.SamplingParams()
        body = {**self._ref(), "prompt": _ints(prompt), "num_samples": int(num_samples), "max_tokens": sp.max_tokens,
                "temperature": sp.temperature, "top_p": sp.top_p, "top_k": sp.top_k, "stop": list(sp.stop or []),
                "seed": sp.seed, "include_prompt_logprobs": include_prompt_logprobs}

        def run():
            r = self._h.call("POST", "/api/tune/v1/sample", body)
            return types.SampleResponse([types.SampledSequence(s["tokens"], s["logprobs"], s["stop_reason"]) for s in r["sequences"]],
                                        r.get("prompt_logprobs"))
        return _Future(self._pool.submit(run))

    async def sample_async(self, *a, **k):
        return self.sample(*a, **k)

    def compute_logprobs(self, prompt):
        body = {**self._ref(), "prompt": _ints(prompt)}
        return _Future(self._pool.submit(lambda: self._h.call("POST", "/api/tune/v1/compute_logprobs", body)["logprobs"]))

    def get_tokenizer(self):
        if not self._tok_model:
            raise IsomorphError("This sampler does not know its base model; ask the training client for the tokenizer.")
        return Tokenizer(self._h, self._tok_model)


class TrainingClient:
    def __init__(self, http, info):
        self._h, self.info = http, info
        self.id, self.base_model = info["id"], info["spec"]
        self._pool = ThreadPoolExecutor(1)       # keeps calls in the order they were made
        self._tok = None

    def _p(self, path):
        return f"/api/tune/v1/sessions/{self.id}{path}"

    def get_info(self):
        return self._h.call("GET", self._p(""))

    def get_tokenizer(self):
        if self._tok is None:
            self._tok = Tokenizer(self._h, self.base_model)
        return self._tok

    # ------------------------------------------------------------ the primitives
    def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        body = {"data": _ser(data), "loss_fn": loss_fn, "loss_fn_config": loss_fn_config}

        def run():
            r = self._h.call("POST", self._p("/forward_backward"), body)
            return types.ForwardBackwardOutput(r["metrics"], [{"logprobs": types.TensorData(o["logprobs"])} for o in r["loss_fn_outputs"]])
        return _Future(self._pool.submit(run))

    def forward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        body = {"data": _ser(data), "loss_fn": loss_fn, "loss_fn_config": loss_fn_config}

        def run():
            r = self._h.call("POST", self._p("/forward"), body)
            return types.ForwardBackwardOutput(r["metrics"], [{"logprobs": types.TensorData(o["logprobs"])} for o in r["loss_fn_outputs"]])
        return _Future(self._pool.submit(run))

    def forward_backward_custom(self, data, loss_fn):
        """loss_fn(data, logprobs) -> (loss tensor, metrics dict), where logprobs is one torch tensor per datum.
        The gradient of your loss with respect to each log-probability is sent back and applied on the server."""
        ser = _ser(data)

        def run():
            import torch

            r = self._h.call("POST", self._p("/forward"), {"data": ser, "loss_fn": "cross_entropy"})
            lps = [torch.tensor(o["logprobs"], dtype=torch.float32, requires_grad=True) for o in r["loss_fn_outputs"]]
            loss, metrics = loss_fn(data, lps)
            loss.backward()
            gdata = [{"model_input": d["model_input"], "loss_fn_inputs": {"target_tokens": d["loss_fn_inputs"]["target_tokens"],
                                                                         "grads": lp.grad.tolist()}} for d, lp in zip(ser, lps)]
            self._h.call("POST", self._p("/forward_backward"), {"data": gdata, "loss_fn": "linear"})
            return types.ForwardBackwardOutput({**(metrics or {}), "loss:sum": float(loss.detach())},
                                               [{"logprobs": types.TensorData(o["logprobs"])} for o in r["loss_fn_outputs"]])
        return _Future(self._pool.submit(run))

    def optim_step(self, adam_params=None):
        a = adam_params or types.AdamParams()
        body = {k: getattr(a, k) for k in ("learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")}
        return _Future(self._pool.submit(lambda: types.OptimStepResponse(self._h.call("POST", self._p("/optim_step"), body)["metrics"])))

    def log(self, metrics):
        """Isomorph only: put your own numbers (reward, accuracy...) on the Tune tab's chart for this step."""
        return _Future(self._pool.submit(lambda: self._h.call("POST", self._p("/report"), dict(metrics))))

    # ------------------------------------------------------------ saving
    def save_weights_for_sampler(self, name=""):
        return _Future(self._pool.submit(lambda: types.SaveWeightsResponse(
            **{k: v for k, v in self._h.call("POST", self._p("/save_weights"), {"name": name}).items() if k in ("path", "name")})))

    def save_weights_and_get_sampling_client(self, name=""):
        path = self.save_weights_for_sampler(name).result().path
        return SamplingClient(self._h, model_path=path, tokenizer_model=self.base_model)

    def create_sampling_client(self, model_path):
        return SamplingClient(self._h, model_path=model_path, tokenizer_model=self.base_model)

    def live_sampling_client(self):
        """Isomorph only: sample the current weights without saving them first."""
        return SamplingClient(self._h, session=self.id, pool=self._pool, tokenizer_model=self.base_model)

    def save_state(self, name="", with_optimizer=True):
        return _Future(self._pool.submit(lambda: types.SaveStateResponse(
            **{k: v for k, v in self._h.call("POST", self._p("/save_state"), {"name": name, "with_optimizer": with_optimizer}).items()
               if k in ("path", "name")})))

    def close(self):
        self._h.call("DELETE", self._p(""))

    # ------------------------------------------------------------ async twins
    async def forward_backward_async(self, *a, **k):
        return self.forward_backward(*a, **k)

    async def forward_async(self, *a, **k):
        return self.forward(*a, **k)

    async def optim_step_async(self, *a, **k):
        return self.optim_step(*a, **k)

    async def save_state_async(self, *a, **k):
        return self.save_state(*a, **k)

    async def save_weights_for_sampler_async(self, *a, **k):
        return self.save_weights_for_sampler(*a, **k)

    async def save_weights_and_get_sampling_client_async(self, *a, **k):
        return await asyncio.to_thread(self.save_weights_and_get_sampling_client, *a, **k)


class ServiceClient:
    def __init__(self, base_url=None, password=None, api_key=None, **_):
        self._h = _Http(base_url, password)
        self._h.call("GET", "/api/tune/v1/health")

    def get_server_capabilities(self):
        return self._h.call("GET", "/api/tune/v1/models")

    def create_lora_training_client(self, base_model, rank=32, alpha=32.0, seed=0, train_attn=True, train_mlp=True,
                                    train_unembed=False, name=""):
        info = self._h.call("POST", "/api/tune/v1/sessions", {"base_model": base_model, "rank": rank, "alpha": alpha, "seed": seed,
                                                               "train_attn": train_attn, "train_mlp": train_mlp,
                                                               "train_unembed": train_unembed, "name": name})
        return TrainingClient(self._h, info)

    def create_training_client_from_state(self, path):
        return TrainingClient(self._h, self._h.call("POST", "/api/tune/v1/sessions/from_state", {"path": path, "with_optimizer": False}))

    def create_training_client_from_state_with_optimizer(self, path):
        return TrainingClient(self._h, self._h.call("POST", "/api/tune/v1/sessions/from_state", {"path": path, "with_optimizer": True}))

    def create_sampling_client(self, base_model=None, model_path=None):
        if not base_model and not model_path:
            raise IsomorphError("Give base_model or model_path.")
        return SamplingClient(self._h, base_model=base_model, model_path=model_path, tokenizer_model=base_model)

    async def create_lora_training_client_async(self, *a, **k):
        return await asyncio.to_thread(self.create_lora_training_client, *a, **k)

    async def create_sampling_client_async(self, *a, **k):
        return self.create_sampling_client(*a, **k)
