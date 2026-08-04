# Black Hole Probe

## Test

NVIDIA:

```bash
cargo test -p black-hole-probe --features cuda,qwen35_0p8b --release -- --nocapture
```

Apple Silicon:

```bash
cargo test -p black-hole-probe --features metal,qwen35_0p8b --release -- --nocapture
```

