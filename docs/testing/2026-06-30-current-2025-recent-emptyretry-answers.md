# eval run — 2026-06-30-current-2025-recent-emptyretry-answers

1 cases, 1 passed all applicable checks. Each section is one case (structural header + answer body).

---

## current-2025-recent

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:12.1s  total:411.8s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

You asked about the most significant developments in open-weight large language models during 2025, specifically what happened and when. Based on the available information, here is a chronological overview of the key releases and shifts that shaped the landscape that year.

**DeepSeek-R1 and the Foundation (January & Late 2024)**
DeepSeek-R1 was reportedly released on January 20, 2025. It is described as a reasoning-focused model that reportedly uses chain-of-thought to work through complex problems step-by-step before producing an answer. The model was made fully open-weight under the MIT license, and it reportedly demonstrated reasoning capabilities competitive with OpenAI’s proprietary o1 model. DeepSeek also reported building it at a fraction of the cost of comparable models. This release built upon DeepSeek-V3, which is said to have been released on December 26, 2024. That base model reportedly featured an efficient Mixture-of-Experts architecture and, according to DeepSeek, had an extremely low training cost relative to comparable models. DeepSeek reported a training cost of roughly $5.6 million, though this figure has not been independently verified. The base model is also said to have had 671B total parameters with approximately 37B active per token. Following the R1 release, there were reports that it triggered a significant stock-market reaction and widespread discussion about the viability of open-weight models versus closed, capital-intensive alternatives, including a notable drop in NVIDIA's stock price, though multiple factors may have contributed to market movements at the time. DeepSeek also reportedly released distilled versions of R1 ranging from 1.5B to 70B parameters, which, according to DeepSeek’s technical report, were built on base models from Qwen and Llama.

**Spring 2025 Releases: Gemma 3, Qwen3, and Llama 4**
In March, Google released Gemma 3 on March 12, 2025, according to reports. This update reportedly introduced multimodal capabilities including vision support and came in sizes from 1B to 27B parameters. It was designed to run efficiently on consumer hardware, including single-GPU and mobile inference configurations, and reportedly supported 128K-token context windows.

Later that spring, Alibaba's Qwen team reportedly released Qwen3 on April 29, 2025. The release reportedly included models ranging from 0.6B to 235B parameters, including both dense and Mixture-of-Experts architectures. A notable feature was the introduction of "hybrid thinking" models that could switch between a standard response mode and an extended reasoning mode within a single model. The flagship Qwen3-235B-A22B (235B total parameters, 22B active) was reportedly competitive with top-tier proprietary models on several benchmarks. The entire family is said to have been released under Apache 2.0.

Around the same time, Meta released Llama 4 on April 5, 2025, according to reports. This marked the first time Llama models reportedly adopted a Mixture-of-Experts architecture. The initial releases included two variants: Llama 4 Scout and Llama 4 Maverick. Scout is said to have had approximately 17B active parameters out of approximately 109B total, was natively multimodal (text + vision), and Meta claimed a 10M-token context window for it. It is also described as capable of running on a single GPU. Maverick reportedly had approximately 17B active parameters out of approximately 400B total and was also multimodal, targeting stronger performance for more demanding tasks. Meta claimed Maverick was competitive with GPT-4o and Claude 3.5 Sonnet on several benchmarks, though third-party evaluations and community testing reportedly suggested the models underperformed those claims. There were also allegations that Meta's LMArena benchmark submissions used a differently-configured version of Maverick than the released open-weight model. A larger variant, Llama 4 Behemoth, was mentioned as still in training and not released.

**Mid-to-Late 2025 Updates and Other Notable Releases**
DeepSeek continued iterating through the year. DeepSeek-V3.1 was reportedly released in September 2025, incorporating reasoning capabilities distilled from R1 into the base V3 model. Additionally, DeepSeek-R1-0528 was reportedly released on May 28, 2025, as an updated version of the R1 reasoning model with improved reasoning and coding performance.

Mistral also released open-weight models, including what it called Mistral Small 3, though its impact relative to other releases is less clear. It was reportedly released in 2025 under the Apache 2.0 license. It is described as designed for efficiency and low-latency deployment, offering performance competitive with much larger models while being small enough to run locally on laptops and edge devices.

**Broader Trends and Context**
Throughout 2025, it was reported that the performance gap between the best open-weight models and closed models like GPT-4o and Claude 3.5 narrowed significantly, particularly in reasoning and coding benchmarks. DeepSeek R1's January release catalyzed a wave of reasoning-capable open models, and reasoning capabilities became increasingly common across 2025 releases rather than serving as a unique differentiator. Additionally, Mixture-of-Experts became more common in major open-weight releases, with both Llama 4 and Qwen3 adopting the architecture. The trend toward permissive licensing (MIT, Apache 2.0) reportedly accelerated in 2025, lowering barriers for commercial adoption and fine-tuning.

Please note that while these developments are widely discussed in tech coverage, specific dates, benchmark figures, and internal claims should be verified against official announcements from the respective labs. Let me know if you would like a deeper dive into any of these releases or their technical architectures!

## Sources
- [DeepSeek-R1 Official Announcement and Technical Report](https://deepseek.com)
- [Meta Llama 4 Announcement](https://ai.meta.com/blog/llama-4)
- [Mistral Small 3 Release](https://mistral.ai)


---
> _Tools used:_  _(✅ = calls succeeded, ❌ = calls failed)_
> - **deepseek-v4-pro:cloud** — `web_search` ✅13
> - **qwen3.6:35b** — `web_search` ✅11 ❌3
> - **glm-5.2:cloud** — `web_search` ✅15
