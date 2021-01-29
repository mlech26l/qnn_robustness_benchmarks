# Benchmark networks of the AAAI 2021 paper Scalable Verification of Quantized Neural Networks

[Link to the technical report (arXiv)](https://arxiv.org/pdf/2012.08185.pdf)

The benchmark consists of robustness verification queries.

Benchmark 1 | MNIST 
--- | --- 
Network size | 784-64-32-10 
Input quantization | 8-bit 
Network quanization (weights and activations) | 6-bit
Samples | 400 (100 per epsilon value)
Epsilons (L-infinity norm) | 1,2,3,4
Weights | ```weights/mnist_mlp.h5```


Benchmark 2 | Fashion-MNIST 
--- | --- 
Network size | 784-64-32-10 
Input quantization | 8-bit 
Network quanization (weights and activations) | 6-bit
Samples | 300 (50/100 per epsilon value)
Epsilons (L-infinity norm) | 1,2,3,4
Weights | ```weights/fashion-mnist_mlp.h5```


The file ```iterate_samples.py``` shows how to load and use the networks.
It also iterates over the exact samples of the benchmark and output the corresponding epsilons to use.

```bash
python3 iterate_samples.py --dataset fashion-mnist
```

When implementing some verification method of the networks of these benchmarks, 
make sure they exactly match the bit-exact quantized rounding semantics in each layer. 
The files ```quantization_layers.py``` and ```quantization_utils.py``` should serve as a good starting point.

## Quantitative results of the bit-vector SMT-based verification

### MNIST samples 0-99

Type | Test sample ID
--- | --- 
Misclassified (1) | 18
Timeout (0) | -
Vulnerable (0) | -
Robust (99) | Remaining

### MNIST samples 100-199

Type | Test sample ID
--- | --- 
Misclassified (1) | 149
Timeout (5) | 104, 119, 175, 193, 195
Vulnerable (3) |  115, 151, 158
Robust (91) | Remaining

### MNIST samples 200-299

Type | Test sample ID
--- | --- 
Misclassified (4) | 217, 241, 247, 259
Timeout (25) | 204, 210, 211, 218, 221, 224, 227, 232, 233, 234, 235, 243, 244, 250, 251, 255, 257, 264, 266, 273, 274, 275, 289, 290, 299
Vulnerable (1) | 282
Robust (70) | Remaining

### MNIST samples 300-399

Type | Test sample ID
--- | --- 
Misclassified (3) | 321, 340, 381 
Timeout (43) | 300, 301, 303, 307, 308, 322, 324, 325, 326, 328, 329, 335, 336, 337, 339, 341, 344, 345, 349, 350, 352, 354, 357, 358, 359, 362, 366, 368, 370, 372, 373, 377, 376, 379, 383, 385, 386, 388, 389, 391, 393, 394, 397
Vulnerable (1) | 320
Robust (53) | Remaining

### Fashion-MNIST samples 0-99

Type | Test sample ID
--- | --- 
Misclassified (13) | 12, 17, 23, 25, 26, 29, 40, 48, 51, 66, 68, 89, 98
Timeout (11) | 4, 21, 28, 42, 43, 44, 49, 57, 67, 73, 91
Vulnerable (0) | -
Robust (76) | Remaining

### Fashion-MNIST samples 100-199

Type | Test sample ID
--- | --- 
Misclassified (10) | 107, 127, 145, 147, 150, 153, 163, 183, 192, 193
Timeout (17) | 101, 117, 122, 129, 136, 149, 151, 157, 166, 170, 172, 181, 188, 191, 195, 197, 198
Vulnerable (1) | 135
Robust (72) | Remaining

### Fashion-MNIST samples 200-249

Type | Test sample ID
--- | --- 
Misclassified (7) | 222, 226, 239, 241, 244, 247, 249
Timeout (16) | 203, 205, 202, 213, 217, 219, 221, 224, 228, 230, 235, 238, 243, 245, 246, 248
Vulnerable (1) | 227
Robust (26) | Remaining

### Fashion-MNIST samples 300-349

Type | Test sample ID
--- | --- 
Misclassified (6) | 316, 322, 324, 325, 332, 344
Timeout (26) | 300, 301, 302, 304, 312, 313, 309, 308, 311, 315, 318, 319, 320, 321, 323, 326, 329, 331, 333, 334, 336, 337, 338, 341, 342, 348
Vulnerable (0) | -
Robust (18) | Remaining
