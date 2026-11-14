# Designing an LLM Inference Platform for 10,000 RPS

> A system-design walkthrough, plus real measurements: how to serve 10,000 requests per second with p99 latency under 2 seconds, and what actually breaks first. Tested on free Kaggle T4 GPUs with vLLM.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![vLLM](https://img.shields.io/badge/serving-vLLM-purple)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green)

**Read the full write-up:** [Medium post](https://medium.com/@your-handle/your-post-link) (replace with your link)

---

## The question

> Design an LLM inference platform that has to serve 10,000 requests per second with p99 latency under 2 seconds. How would you handle batching, KV cache, autoscaling, and cost? What breaks first as traffic grows?

This repo breaks the question into parts: the sizing arithmetic, the architecture, the cost on AWS, and a hands-on experiment that checks which claims in a typical answer hold up on real hardware.

---

## Architecture

![Full architecture](images/fig1_architecture.png)

| Layer | Components | Role |
|---|---|---|
| Request path | Clients, API gateway, cache-aware router | Auth, rate limits, admission control, token streaming, prefix-affinity routing |
| GPU data plane | Replica pool (vLLM / SGLang), weight store | Run the model; replicas can be split into prefill and decode pools |
| Control plane | Metrics, autoscaler, warm pool | Scale on queue depth and TTFT (not CPU); shed load when queues grow |

Inside each replica:

![Inside one replica](images/fig2_replica.png)

- **Continuous batching:** requests join and leave the batch at every decode step
- **Chunked prefill:** long prompts do not block other users' token generation
- **PagedAttention:** KV cache in small blocks, so less memory is wasted
- **Prefix caching:** shared system prompts are computed once

---

## Sizing arithmetic

Scenario: 8B-class model, 500 input and 200 output tokens per request, streamed responses.

| Quantity | Value |
|---|---|
| Requests in flight (Little's law, 10,000 RPS x about 1.5 s) | about 15,000 |
| Prefill load | about 5 million input tokens per second |
| Decode load | about 2 million output tokens per second |
| Per-token budget (0.3 s TTFT, 200 tokens in the remaining 1.7 s) | about 8.5 ms |
| KV cache per token, Llama-3-8B-style, FP16 | about 128 KB |
| KV cache for 15,000 requests at 1,000 tokens | about 2 TB across the fleet |

Fleet formula used throughout:

```
replicas = target RPS / (RPS one replica sustains within the SLO) x headroom
```

---

## The experiment

One replica, then two, on Kaggle T4 GPUs: `Qwen/Qwen2.5-0.5B-Instruct` on vLLM, about 200-token unique prompts, 128 output tokens, `--max-num-seqs 64`. A streaming asyncio load generator measures TTFT and end-to-end latency while a sampler reads vLLM's `/metrics` for queue depth and KV cache usage.

### Single replica (unique prompts)

| Concurrency | RPS | p99 TTFT | p99 latency | Within 2 s p99? |
|---|---|---|---|---|
| 16 | 11.4 | 0.36 s | 1.55 s | yes |
| 24 | 13.9 | 0.40 s | 1.84 s | yes (thin margin) |
| 32 | 15.7 | 0.54 s | 2.22 s | no |
| 40 | 16.6 | 0.66 s | 2.72 s | no |
| 48 | 17.4 | 0.82 s | 3.22 s | no |
| 64 | 19.5 | 1.07 s | 3.93 s | no |

### Two replicas (round-robin, warmed up)

| Total concurrency | RPS | p99 TTFT | p99 latency | Within 2 s p99? |
|---|---|---|---|---|
| 32 | 22.1 | 0.30 s | 1.57 s | yes |
| 48 | 27.5 | 0.44 s | 1.94 s | yes |
| 64 | 31.6 | 0.71 s | 2.34 s | no |
| 96 | 36.4 | 0.78 s | 2.99 s | no |
| 128 | 40.3 | 1.07 s | 3.63 s | no |
| 160 | 40.0 | 3.48 s | 6.52 s | no |

![Capacity curves](images/fig3_capacity_curves.png)

### What the data showed

1. **Batching is a roughly 20x throughput lever.** About 168 tokens/s with one stream versus about 3,500 with 64. The price: time per token grew from about 6 ms to about 19 ms.
2. **Identical prompts flatter the numbers.** The prefix cache hid prefill cost. With unique prompts, throughput fell 19-28% and p99 TTFT rose about five-fold.
3. **The SLO breaks in stages.** Per-token slowdown first, then prefill queueing (requests waited before the scheduler cap was reached), then the cap itself.
4. **KV cache never mattered here.** It stayed under 3% for a 0.5B model with short prompts. It would come into play with long contexts or bigger models.
5. **The GPU was busy, not starved.** 100% utilization, about 66 W of a 70 W cap, and the busiest vLLM process used 43-61% of one CPU core. That ruled out a host bottleneck, though the card still ran far below its compute and bandwidth peaks.
6. **Two replicas scaled almost linearly,** 98-105% of double the throughput at equal load per replica, until the roughly 40 RPS ceiling.

![What breaks first](images/fig4_what_breaks_first.png)

### Measurement traps worth knowing

| Trap | Symptom | Fix |
|---|---|---|
| Cold replica | Absurd TTFT on the first run against a new server | Send warm-up requests to every replica |
| `aiohttp` default connection limit of 100 | TTFT inflated above 100 concurrent requests; server shows 64 running + 36 waiting, not 128 | `aiohttp.TCPConnector(limit=0)` |
| Two GPUs in one `nvidia-smi` query | Meaningless 0% median utilization | Query one device: `nvidia-smi -i 0` |

---

## Cost on AWS (planning estimates, not quotes)

Prices from public trackers, US East on-demand, October 2026: `p5.48xlarge` (8 x H100) about $55.04/hr; `g6e.xlarge` (1 x L40S) about $1.86/hr; `g4dn.xlarge` (1 x T4) about $0.53/hr. AWS GPU prices have changed several times in 2025-2026, so verify in the AWS Pricing Calculator.

![Cost levers](images/fig5_cost_levers.png)

| Scenario (8B-class, 10,000 RPS) | Nodes | Estimated per month |
|---|---|---|
| Baseline: FP16, no prefix caching | about 60 `p5.48xlarge` | about $2.4M |
| Prefix caching (80% of prompt shared) | about 34 | about $1.4M |
| Plus FP8 | about 17 | about $0.7M |

A 1-year commitment would take roughly 30-50% off. For the toy 0.5B model on T4s, the measured safe point of about 11 RPS per replica implies about 1,180 replicas, roughly $0.45M per month. That demonstrates the method and should not be compared with the 8B scenario.

---

## Limitations

- A 0.5B model on T4s is not an 8B model on H100s; the bottleneck mix differs.
- Requests had similar lengths. I did not test mixed lengths or load-aware versus round-robin routing.
- Only one and two replicas were tested. The router, autoscaler and warm pool rest on reasoning and published practice, not measurements.
- Cost figures are estimates built on assumed throughput per H100.

---

## Run it yourself

1. Create a [Kaggle](https://www.kaggle.com) notebook, set the accelerator to a GPU (T4) and turn **Internet** on.
2. Install and start the server:

```python
!pip install -q vllm
```

```python
import subprocess
server = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--dtype", "half", "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.85", "--max-num-seqs", "64",
    "--enable-prefix-caching", "--port", "8000",
])
```

3. Run the load generator in `src/load_test.py` (or the notebook) at rising concurrency, and compare p99 with your own SLO.
4. For a second replica, start another server with `CUDA_VISIBLE_DEVICES=1` on port 8001 and round-robin between them.

vLLM versions and attention backends change; if the install or launch fails on a T4, check `vllm.log` first.

---

## Repository structure

```
llm-inference-platform-10k-rps/
├── README.md
├── notebooks/
│   └── capacity_experiment.ipynb
├── src/
│   ├── load_test.py
│   └── metrics_sampler.py
├── images/
│   ├── fig1_architecture.png
│   ├── fig2_replica.png
│   ├── fig3_capacity_curves.png
│   ├── fig4_what_breaks_first.png
│   └── fig5_cost_levers.png
└── docs/
    └── blog-post.md
```

---

## Glossary

| Term | Meaning |
|---|---|
| TTFT | Time to first token |
| p99 | The latency 99% of requests beat |
| Prefill | Processing the prompt (compute-bound) |
| Decode | Generating tokens one at a time (memory-bound) |
| Continuous batching | Adding and removing requests at every decode step |
| KV cache | Stored attention keys and values for each token in a request |
| PagedAttention | Storing the KV cache in small blocks to avoid waste |
| Warm pool | Pre-loaded spare replicas that absorb bursts |

---

## Further reading

- vLLM documentation (continuous batching, PagedAttention, prefix caching, chunked prefill)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Related repo: GPU profiling and the roofline model (compute-bound vs memory-bound vs launch-bound)

## License

MIT. See `LICENSE`.
