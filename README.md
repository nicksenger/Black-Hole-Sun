# Black Hole Probe

## Sanity

```bash
cargo test -p black-hole-probe --features cuda,qwen35_0p8b --release -- --nocapture --ignored
```

Full QuZO optimization flow (PerturbUp -> Infer -> PerturbDown -> Infer -> Optimize -> Infer):

```bash
BLACK_HOLE_PROBE_MODEL_PATH=/path/to/model.gguf cargo test -p black-hole-probe --features cuda,qwen35_0p8b --release -- --nocapture --ignored optimization
```
