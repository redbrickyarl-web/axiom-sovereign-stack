# Integrating Aethelarch into Axiom Sovereign Stack

## Overview

[Aethelarch](https://github.com/redbrickyarl-web/aethelarch) is the dual-bitplane ternary GEMV microkernel.
This repo consumes it via FFI under `src/ffi/`.

## Quick link steps

```bash
# 1. Build Aethelarch
git clone https://github.com/redbrickyarl-web/aethelarch.git
cd aethelarch && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

# 2. Build this crate with the feature
export AETHELARCH_LIB=$(pwd)          # dir with libaethelarch.a
export AETHELARCH_INCLUDE=../include
cd /path/to/axiom-sovereign-stack
cargo build --features aethelarch
```

## Module map

| Rust path | Role |
|-----------|------|
| `src/ffi/mod.rs` | FFI module root |
| `src/ffi/aethelarch.rs` | Safe wrapper + extern C decls |
| `build.rs` | Links `libaethelarch` when feature enabled |

## Feature flag

```toml
[features]
default = []
aethelarch = []
```

Without the feature, the wrapper compiles as stubs (quantize falls back to pure Rust bit packing; gemv returns false).
