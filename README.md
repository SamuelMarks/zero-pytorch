# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-pytorch

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-pytorch/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-94.2%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
[![Version](https://img.shields.io/badge/version-2.10.0-blue.svg)](#)

Zero-dependency pure Python implementation of the PyTorch API surface.

*Current snapshot version: 2.10.0*

---

## Why this project exists

`zero-pytorch` is a core component of the **Abstract ML Machine Ecosystem**, designed to solve the $N \times M$ translation problem in Machine Learning. Instead of writing bespoke translators for every framework (JAX, PyTorch, Keras) to every target (WASM, WebGPU, TensorRT), the ecosystem traces $N$ frontends into a strictly defined Intermediate Representation (IR), which is then consumed by $M$ backends.

`zero-pytorch` specifically serves as a "Tier 4" **Frontend** in this architecture. Its goal is to mimic the eager, object-oriented, and stateful semantics of the PyTorch API. However, underneath the familiar PyTorch-like API surface, it performs no heavy computation itself. 

Instead, it relies on the computational heart of the ecosystem, `ml-switcheroo-compiler`. When a user executes code in `zero-pytorch`, the operations are captured using `ProxyTensors` that overload Python math dunders. It dynamically lifts mutable states (such as `nn.Parameter` or buffers) into purely functional graph inputs and outputs via the compiler's internal `lift_state` pass. 

### Key Features:
- **Strictly zero external dependencies:** Relies entirely on the Python Standard Library and `numpy` (for eager evaluations and broadcast rules).
- **Source-to-browser compilation ready:** Directly interoperates with `ml-switcheroo-compiler`'s tracer tape, enabling compilation of PyTorch code into executable WASM or WebGPU binary code without pulling in the massive binaries of the real PyTorch framework.
- **Identical API Surface:** Maintains strict signature compliance with PyTorch version 2.10.0, automatically validated against canonical JSON snapshots to ensure drop-in compatibility.

By bridging the familiar stateful PyTorch API with a purely functional compiler backend, `zero-pytorch` enables headless CI pipelines and embedded execution without the bloat of traditional ML frameworks.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
