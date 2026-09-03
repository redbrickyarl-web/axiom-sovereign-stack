# Aethelarch integration (turnkey)

Native sources are vendored under `native/aethelarch/`.

## Build

```bash
cargo build --features aethelarch
```

No environment variables required. The `cc` crate compiles:

- `quantize.c` + `kernel_scalar.c` (always)
- `kernel_neon.c` on `aarch64`
- `kernel_avx512.c` on `x86_64` (when toolchain accepts AVX-512 VPOPCNTDQ flags)

## API

See `src/ffi/aethelarch.rs`:

- `AethelarchMatrix::from_ternary` / `encode_dense`
- `quantize_activation` / `gemv`
- `AethelarchError` structured errors

Standalone C library: https://github.com/redbrickyarl-web/aethelarch
