# Black Hole Sun

`black-hole-sun` is designed to perform blackbox co-optimization of multi-agent systems. It uses the [QuZO](https://arxiv.org/html/2502.12346v1) zeroth-order optimization method via [paramecia](https://github.com/nicksenger/paramecia), which means that at this time only GGUF quantized models of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc) are supported.

`black-hole-sun` is heavily reliant on the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" (WAT) event-replay orchestration system, and could be considered an extension of that work. I am developing it to answer some questions I have about the behavior of multi-agent systems under training/selection pressure, but will share it here for anyone else interested.

Here's a video of training a network of Qwens using `black-hole-sun`: 

[vid]

Each node proceeds through 3 phases:

1. Propagation 1: (unfrozen) models' weights are perturbed with random noise
2. Propagation 2: (unfrozen) models' weights are again perturbed in the opposite direction
3. Potentiation: the direction of the gradient predicted from the prop1 and prop2 predictions is used to optimize the weights

Then the process starts over again from propagation 1.

The appeal is that this allows for training quantized weights with very little memory overhead beyond what is required for normal inference situations. The downside is that it will generally take _much_ longer to converge compared to first-order methods.

For this reason I've added a piano to the UI to keep folks entertained while the weights cook.

## Commands

## Test

NVIDIA:

```bash
cargo test -p black-hole-probe --features cuda,qwen35_0p8b --release -- --nocapture
```

Apple Silicon:

```bash
cargo test -p black-hole-probe --features metal,qwen35_0p8b --release -- --nocapture
```

```bash
cargo test -p black-hole-probe beam_test --features metal,qwen35_0p8b --release -- --nocapture --ignored
```

