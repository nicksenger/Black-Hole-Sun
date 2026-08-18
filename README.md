# Black Hole Sun

`black-hole-sun` facilitates blackbox co-optimization of multi-agent systems. It uses the [QuZO](https://arxiv.org/html/2502.12346v1) zeroth-order optimization method through a custom engine which supports GGUF quantized models of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc).

`black-hole-sun` is an extension of the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" (WAT) event-replay orchestration system. It is primarily a research tool designed to explore some questions I have about the behavior of multi-agent systems. I'm sharing it here in case it proves helpful for others working in this area.

The appeal of this distributed training approach is that it requires very little memory beyond simple inference, making it suitable for consumer/edge devices. The tradeoff is that it is an approximation, so convergence may take longer (sometimes even of cosmic proportions) than when training by first-order methods.

To compensate for this I've added a piano to the UI as shown in this video:

[vid]

Nodes in a `BlackHole::Sun` graph (`Cell`s) are agents implemented in terms of `jungle::Flow` that is expected to combine some neural network with arbitrary symbolic pre/post processing steps and support progression through the following 3 phases:

1. **Propagation 1**: weights are perturbed and samples are collected
2. **Propagation 2**: weights are perturbed in the opposite direction and samples collected
3. **Potentiation**: the gradient is approximated and the weights optimized

The process then starts over again from propagation 1.

The role of `black-hole-sun` is to orchestrate this process efficiently and reliably over arbitrarily large networks of cells.

