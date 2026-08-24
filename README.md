# Black Hole Sun

`black-hole-sun` is an extension of the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" (WAT) orchestration system which conducts blackbox co-optimization of distributed artificial intelligence agents.

It runs on a [custom engine](https://github.com/nicksenger/paramecia) supporting GGUF quantizations of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc), and implements the forward-only optimization strategies described in these publications:

- [Gradients without Backpropagation](https://arxiv.org/abs/2202.08587) (Baydin et al., 2022)
- [QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models](https://arxiv.org/abs/2502.12346) (Zhou et al., 2026)
- [Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost](https://arxiv.org/abs/2602.03120) (Xu et al., 2026)

It includes a GUI monitoring tool with optional piano:

https://github.com/user-attachments/assets/9c399800-04bb-4bc7-93fc-d2fa1a400b53

A `BlackHole::Sun` graph is a hierarchical, static-topology DAG where each node is an agent implemented in terms of `jungle::Flow` that progresses through 3 phases:

1. **Propagation 1**: a perturbation is applied (to weights or activations) and samples are collected
2. **Propagation 2**: the inverse perturbation is applied and samples collected again
3. **Potentiation**: gradients are approximated and any unfrozen weights are updated

The process then starts over again from propagation 1.

The role of `black-hole-sun` is to orchestrate this cycle efficiently and reliably over arbitrarily large networks of cells.

`black-hole-sun` is an art project and research tool. Documentation is sparse and the interface should be considered highly unstable.

The software is provided 'as-is', without warranty of any kind, express or implied. In no event shall the author be held liable for any claim, damages or other liability, whether in an action or contract, tort or otherwise, arising from, out of or in connection with the software or the use or dealings in the software. 

