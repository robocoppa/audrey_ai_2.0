# eval run — 2026-06-30-current-2025-recent-rerun-answers

1 cases, 1 passed all applicable checks. Each section is one case (structural header + answer body).

---

## current-2025-recent

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:12.8s  total:252.1s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

I couldn’t fully verify these details against retrieved sources, as multiple researchers found no usable primary documentation for this period and the available snippets were internally contradictory. That said, I can share what is well-established in the field, followed by a careful breakdown of the specific claims from the research notes.

**(1) Established Context**
Open-weight large language models have consistently prioritized advanced reasoning capabilities and efficient inference architectures. The industry widely adopted Mixture of Experts (MoE) designs to balance scale with computational efficiency, and open-weight systems have steadily narrowed the performance gap with proprietary counterparts on standard benchmarks. These shifts represent well-documented directions in the field rather than isolated events.

**(2) Specific Claims from the Research Notes**
Because the underlying search results were unverified and primary sources like official release blogs or benchmark papers could not be accessed, I cannot confirm the exact dates, license terms, architectural details, or performance claims for any of these models. Based on the available reports, here is what was noted:

- **DeepSeek-R1**: Widely reported to have been released in January 2025 under a permissive license (often described as MIT). It was seen by many as evidence that open-weight models could compete at the frontier of reasoning capabilities.
- **Meta Llama 4**: Reportedly announced or released around April 2025, featuring a Mixture of Experts architecture and native multimodal processing.
- **Mistral Large 2**: Mistral AI reportedly released this updated model in 2025, though exact specifications remain unconfirmed.
- **Qwen 2.5**: The series may have first appeared in late 2024, with expanded releases and variants rolling out through 2025.
- **DeepSeek-V3**: DeepSeek also reportedly released this large MoE model in 2025.

I cannot verify the precise release windows, licensing details, or benchmark performance for any of these models from the available materials. To get a reliable timeline, you would need to consult verified vendor announcements, official model cards, or technical documentation directly.

---
> _Tools used:_  _(✅ = calls succeeded, ❌ = calls failed)_
> - **deepseek-v4-pro:cloud** — `web_search` ✅11 ❌2
> - **qwen3.6:35b** — `web_search` ✅13
> - **glm-5.2:cloud** — `web_search` ✅13
