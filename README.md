# Black Hole Sun

`black-hole-sun` is an extension of the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" (WAT) orchestration system that conducts blackbox co-optimization of networked AI agents.

It runs on a custom inference engine which supports several forward-only optimization methods for GGUF quantizations of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc). It also has a UI with a piano that the [Man in the Box](https://github.com/nicksenger/Man-in-the-Box) ported over from one of my [first github projects](https://github.com/nicksenger/NanoMoog):

https://github.com/user-attachments/assets/9c399800-04bb-4bc7-93fc-d2fa1a400b53

Nodes in a `BlackHole::Sun` graph (`Cell`s) are agents implemented in terms of `jungle::Flow` which combine some neural network with arbitrary symbolic pre/post processing steps and support progression through the following 3 phases:

1. **Propagation 1**: perturbation is applied and samples are collected
2. **Propagation 2**: opposite perturbation is applied and samples collected
3. **Potentiation**: the gradient is approximated and the weights optimized

The process then starts over again from propagation 1.

The role of `black-hole-sun` is to orchestrate this process efficiently and reliably over arbitrarily large networks of cells.

