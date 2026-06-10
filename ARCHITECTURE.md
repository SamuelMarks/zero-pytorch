**Current Repository Context:** You are viewing the unified architecture documentation from within the `zero-pytorch` repository.

# Abstract ML Machine Ecosystem Architecture

*Note: This architecture document is shared across all repositories in the `zero-*` and `ml-switcheroo-*` ecosystem to provide comprehensive technical context on how the frameworks interoperate.*

## The N-to-M Translation Problem

The Abstract ML Compiler ecosystem is designed to solve the $N \times M$ translation problem in Machine Learning. Instead of writing bespoke translators for every framework (JAX, PyTorch, Keras) to every target (WASM, WebGPU, TensorRT), we trace $N$ frontends into a strictly defined Intermediate Representation (IR), which is then consumed by $M$ backends.

This achieves a source-to-source and source-to-browser compilation pipeline utilizing **strictly zero external dependencies** (relying solely on the Python Standard Library and `numpy` for eager evaluations).

## Ecosystem Repository Taxonomy

The ecosystem is strictly hierarchical. Circular dependencies are forbidden. The repositories are organized into tiers:

```mermaid
graph TD
    subgraph Tier 1: Core Definitions
        IR[ml-switcheroo-ir]
    end

    subgraph Tier 2: Tracing & AD Engine
        COMP[ml-switcheroo-compiler]
    end

    subgraph Tier 3: Functional Foundation
        ZJ[zero-jax]
    end

    subgraph Tier 4: Neural Networks & Frontends
        ZF[zero-flax]
        ZP[zero-pytorch]
        ZK[zero-keras]
        ZT[zero-tensorflow]
        ZM[zero-mlx]
        ZPX[zero-pax]
    end

    subgraph Tier 5: Verification
        ZZ[zero-zoo]
    end

    COMP -->|Depends On| IR
    
    ZJ -->|Depends On| COMP
    
    ZF -->|Depends On| ZJ
    ZP -->|Depends On| COMP
    ZK -->|Depends On| COMP
    ZT -->|Depends On| COMP
    ZM -->|Depends On| COMP
    ZPX -->|Depends On| COMP
    
    ZZ -.->|Tests| ZF
    ZZ -.->|Tests| ZP
    ZZ -.->|Tests| ZK
    ZZ -.->|Tests| ZJ
```

### 1. `ml-switcheroo-ir` (Tier 1)
The universal, canonical dialect. Defines `LogicalNode` and `LogicalGraph`. Contains the schema validator enforcing ONNX spec compliance without requiring the heavy `onnx` pip package.

### 2. `ml-switcheroo-compiler` (Tier 2)
The computational heart. Implements:
* **TracerTape**: Uses `threading.local` for concurrent, thread-safe AOT tracing.
* **ProxyTensors**: Overloads Python math dunders to capture eager operations.
* **AD Engine (`compiler.grad`)**: Reverse-mode automatic differentiation, topological sorting, gradient accumulation, and exact mathematical VJPs.
* **Optimizations**: Static Shape Inference (matching Numpy broadcast logic), Dead Code Elimination (DCE), and Common Subexpression Elimination (CSE).

### 3. Frontends (`zero-*`) (Tiers 3 & 4)
* **`zero-jax`**: Mimics the JAX API (`jnp`, `lax`, `jit`, `grad`, `vmap`). Uses Pytree flattening to route state safely into the compiler tape.
* **`zero-flax`**: Builds upon `zero-jax` to provide Neural Network layers (`Dense`, `Conv`, `Attention`) and `nnx` state functionalization.
* **`zero-pytorch`, `zero-keras`, `zero-tensorflow`, `zero-mlx`**: Mimic eager, object-oriented, and stateful semantics. They dynamically lift mutable states (like `nn.Parameter` or `tf.Variable`) into purely functional graph inputs/outputs via the compiler's internal `lift_state` pass.

### 4. `zero-zoo` (Tier 5)
The proving grounds. Contains identical architectural definitions (MLP, CNN, Micro-Transformer/NanoGPT) written across all frontends. Headless CI pipelines train these deterministically for 10 steps to assert `.allclose()` float-for-float equivalence ("Golden Seed" testing) across all simulated frameworks and final backend compilations.

---

## Compilation Pipeline & Data Flow

When a user executes code in any `zero-*` frontend, the framework delegates the logic to the backend pipeline, mapping high-level API calls down to executable WASM/WebGPU binary code.

```mermaid
sequenceDiagram
    participant User as zero-* Frontend API
    participant Compiler as ml-switcheroo-compiler
    participant IR as ml-switcheroo-ir
    participant Backend as onnx9000 (Export)

    User->>Compiler: Execute math (e.g., zero_torch.add)
    activate Compiler
    Compiler->>Compiler: Intercept via ProxyTensor
    Compiler->>Compiler: Calculate broadcast shapes (Numpy Rules)
    Compiler->>Compiler: Record to TracerTape
    Compiler-->>User: Return new ProxyTensor
    deactivate Compiler

    User->>Compiler: Trigger Compilation (.backward() / @jit)
    activate Compiler
    Compiler->>Compiler: compiler.grad() (Topological Sort & VJPs)
    Compiler->>Compiler: Optimizations (DCE, CSE, Constant Fold)
    Compiler->>Compiler: lift_state (Functionalize mutations)
    Compiler->>IR: Construct LogicalGraph & LogicalNodes
    deactivate Compiler
    
    IR->>Backend: Provide strict JSON Graph
    activate Backend
    Backend->>Backend: Static Arena Allocation (Memory Offsets)
    Backend->>Backend: Generate LEB128 WASM / WGSL Shaders
    Backend-->>User: Executable Browser Payload
    deactivate Backend
```

### Trace-to-AST Linking
To provide clear error messages and allow for framework-specific syntactic rewrites, the compiler dynamically links trace operations to the original Python syntax trees. Leveraging `inspect.currentframe()`, every `LogicalNode` emitted into the IR captures a `source_ast_ref` binding it back to the exact file path, line number, and AST ID in the user's source code.

## Testing Architecture
The `zero-pytorch` test suite is heavily optimized to guarantee complete behavioral and numerical parity with the official PyTorch distribution.

### `test_all_apis.py`
To avoid regressions and ensure complete coverage, `test_all_apis.py` utilizes dynamic code inspection to automatically discover and parameterize over all 170+ public APIs exported by the library.
It injects standardized tensor shapes and asserts that `zero_torch` output matrices strictly equal (to within `1e-4` precision) the native C++ distributions generated by `import torch`.

### Ported Official PyTorch Tests (`tests/pytorch_official/`)
Because PyTorch's native `test_*.py` files are densely coupled to private C++ metadata wrappers (e.g. `torch.testing._internal`), we've implemented a functional extraction pattern.
We port critical creation `test_ported_creation_ops.py`, network structures `test_ported_nn.py`, gradient calculations `test_ported_autograd.py`, and optimization routines `test_ported_optim.py` from the official repository into native `pytest` fixtures ensuring they execute transparently upon `zero_torch`.

### Adding new ported tests
To fetch new upstream tests:
1. Copy the targeted `test_*.py` file from the PyTorch repository.
2. Isolate the target test methods from the proprietary `TestCase` subclass.
3. Migrate assertions into standard Pytest (`assert`, `np.testing.assert_allclose`) and substitute `torch` calls for `zero_torch` or rely on our built-in EagerMode tensor evaluation.
