# LLaMA Model Quantization

## Overview

This notebook demonstrates quantization techniques for the LLaMA-7B language model. I implemented both basic Round-to-Nearest (RTN) and the more sophisticated GPTQ algorithm to reduce model memory requirements from 16-bit to 4-bit precision while maintaining reasonable performance.

The work shows how to make large language models more accessible by cutting memory usage by roughly 75%.

## Table of Contents

- [Motivation](#motivation)
- [Implementation](#implementation) 
- [Results](#results)
- [Memory Optimizations](#memory-optimizations)
- [Limitations](#limitations)
- [Usage](#usage)

## Motivation

LLaMA-7B normally needs 15GB+ of VRAM. Quantization offers a path to run these models on regular GPUs by representing weights with fewer bits. The challenge is doing this without ruining model quality.

Rather than just implementing naive quantization, I focused on GPTQ - a more advanced technique that considers actual input patterns when deciding how to quantize each layer. This should preserve more of the original model's capabilities.

## Implementation

The notebook covers three main approaches:

**Basic Quantization Framework**
- Row-wise quantization with scale/zero parameters
- Efficient conversion between FP16 and quantized representations
- Custom QuantizedLinear layer that works as a drop-in replacement

**RTN (Round-to-Nearest) Quantization**  
- Straightforward baseline approach
- Quantizes weights independently without considering layer interactions
- Simple but effective baseline

**GPTQ Algorithm**
- Input-aware quantization using Hessian information
- Processes weights in blocks with error compensation
- Sequential layer-by-layer optimization maintaining model dependencies

The GPTQ implementation required managing GPU memory when processing such large models.

## Results

Performance on WikiText-2:

| Method | Perplexity | Memory Usage |
|--------|------------|--------------|
| Original FP16 | ~5.8 | 15GB |
| RTN 4-bit | 6.43 | ~4GB |
| GPTQ 4-bit | 5.96 | ~4GB |

The GPTQ results are pretty encouraging - we get almost the same as original performance while using 4x less memory. RTN is simpler and deteriorates model quality.

Sample outputs from the quantized model show it can handle basic questions reasonably well:

```
Q: What is the capital of France?
A: Paris.

Q: Can you explain the Pythagorean theorem?  
A: The Pythagorean theorem states that the sum of the squares of the sides of a right triangle is equal to the square of the hypotenuse.
```

## Memory Optimizations

Beyond the main quantization work, I implemented dense packing to achieve additional memory saving. Since PyTorch doesn't have native 4-bit support, the naive approach wastes 50% of storage. The dense packing functions solve this by having two 4-bit values within one 8-bit container.


## Limitations


- **Calibration dependency**: GPTQ needs good representative data during quantization.
- **Memory constraints**: Even with optimizations, you still need substantial VRAM (12GB+) during the quantization process itself.
- **Hardware requirements**: Some optimizations are GPU-specific and won't work everywhere.

Future work could explore mixed-precision approaches (different bits for different layers).

## Usage

**Requirements:**
- GPU with 15GB+ VRAM (T4 or better)
- 12GB+ system RAM
- PyTorch 2.0+

The notebook handles downloading the LLaMA weights automatically. Just run the cells in order in Colab - the quantization process takes about 30 minutes on a T4.

For evaluation, it uses WikiText-2. The memory-efficient evaluation approach processes the model layer-by-layer to avoid running out of VRAM.
