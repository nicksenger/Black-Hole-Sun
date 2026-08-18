# Black Hole Sun

`black-hole-sun` is designed to perform blackbox co-optimization of multi-agent systems. It uses the [QuZO](https://arxiv.org/html/2502.12346v1) zeroth-order optimization method via [paramecia](https://github.com/nicksenger/paramecia). This means that only GGUF quantized models of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc) are supported.

`black-hole-sun` is an extension of the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" (WAT) event-replay orchestration system. It is being developed mainly to answer some questions I have about the behavior of multi-agent systems, but I'm sharing it here in case anyone else is interested.

The appeal this approach to training models is that it requires very little memory, making it suitable for consumer or edge devices. It can take a long time though, so I've added a piano to the UI for entertainment:

[vid]

Each node in a `BlackHole::Sun` graph proceeds through 3 phases:

1. Propagation 1: models' weights are perturbed with random noise and samples are collected
2. Propagation 2: models' weights are perturbed in the opposite direction and samples collected
3. Potentiation: the direction of the gradient is predicted from the collected samples and used to optimize the weights

The process then starts over again from propagation 1. `black-hole-sun` orchestrates this across N models, so they can work together to achieve their goals.

