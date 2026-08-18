# Black Hole Sun

`black-hole-sun` is a collection of tools designed to facilitate blackbox co-optimization of agents in a distributed system. Training is based on the [QuZO](https://arxiv.org/html/2502.12346v1) zeroth-order optimization method, and currently supports _only_ GGUF quantized models of the Qwen3.* architecture (e.g. Qwen3.8 27b, Qwen3-Next, etc). `black-hole-sun` relies heavily on the [Jungle](https://github.com/nicksenger/Jungle) "workflow-as-type" orchestration system.

[vid]

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

