# FHE-PPML: Fully Homomorphic Encryption for Privacy-Preserving Machine Learning

**FHE-PPML** is a project dedicated to the exploration and implementation of neural network components using Fully Homomorphic Encryption (FHE), ensuring sensitive data remains encrypted throughout the entire inference process. The repository leverages the **OpenFHE** library and the **CKKS** scheme to perform mathematical operations directly on ciphertexts.

---

## 📂 Project Structure

The repository is organized into modules for testing single-neuron logic, simple dense layers, and multi-layer architectures (with and without bootstrapping for noise budget refresh):

```text
fhe-ppml/
├── single_neuron/ (or silgle_neuron/)
│   └── test_ckks.py              # Validation of homomorphic operations for a single neuron.
├── multi_layer/
│   ├── 1layer-fhe.py             # Test of a homomorphic dense layer with SIMD vector packing.
│   ├── 2layer-fhe.py             # 2-layer homomorphic dense neural network without bootstrapping.
│   ├── 2layer-fhe-bootstrap.py   # 2-layer neural network using FHE Bootstrapping.
│   └── 4layer-bootstrap.py       # Deep 4-layer neural network using FHE Bootstrapping.
└── README.md                     # Repository documentation.
```

---

## 🧪 Modules and Features

### 1. Single Neuron (`single_neuron/test_ckks.py`)
This module validates the core homomorphic operations for an individual neuron.

* **Cryptographic Configuration:** Configures `CCParamsCKKSRNS` with a multiplicative depth of 5, scaling mod size of 50 bits, and batch size of 8.
* **Key Generation:** Generates public, secret, evaluation multiplication keys (`EvalMultKeyGen`), and rotation keys (`EvalAtIndexKeyGen`) for index shifts.
* **Ciphertext-Plaintext Multiplication:** Encrypts input data (`ctxt_data`) and homomorphically multiplies it with plaintext weights (`ptxt_weights`) via `EvalMult`.
* **Logarithmic Sum Reduction:** Computes the vector dot product using homomorphic rotations (`EvalRotate`) and additions (`EvalAdd`) at index shifts.
* **Polynomial Activation:** Evaluates a quadratic polynomial activation function $f(x) = 0.1 + 0.5x + 1.0x^2$ over the encrypted sum using `EvalPoly` with coefficients `[0.1, 0.5, 1.0]`.
* **Cleartext Verification:** Decrypts the final ciphertext (`ctxt_activation`) and compares results against expected values computed in cleartext.

---

### 2. Single Homomorphic Dense Layer (`multi_layer/1layer-fhe.py`)
This module evaluates a dense neural network layer with multiple neurons operating in parallel using SIMD vector packing.

* **Cryptographic Configuration:** Configures `CCParamsCKKSRNS` with a multiplicative depth of 4, scaling mod size of 50 bits, and batch size of 32.
* **Key Generation:** Enables features such as `KeySwitching`, `LeveledSHE`, `PKE`, and `AdvancedSHE`, generating multiplication and rotation keys for indices `[1, 2, 4, 8, -1, -2, -3]`.
* **SIMD Data Packing (`pack_data_for_layer`):** Flattens weight matrices and duplicates input vectors into 1D arrays to process multiple neurons simultaneously in a single encrypted batch.
* **Homomorphic Layer Evaluation (`eval_layer`):**
  * **Homomorphic Multiplication:** Multiplies encrypted packed data and weights via `EvalMult`.
  * **Sum-Tree Reduction (`eval_sum_tree`):** Executes logarithmic $O(\log N)$ reduction via `EvalAtIndex` and `EvalAdd` to sum vector blocks.
  * **Polynomial Sigmoid Activation:** Approximates the Sigmoid function using `EvalPoly` with coefficients `[0.5, 0.197, 0.0, -0.004]` ($f(x) = -0.004x^3 + 0.197x + 0.5$).
* **Noise Metrics & Decoding:** Decrypts layer output, extracts valid neuron output indices, and compares homomorphic results with cleartext computations (`calculate_cleartext_expected`) to track absolute noise error.

---

### 3. 2-Layer Homomorphic Neural Network (`multi_layer/2layer-fhe.py`)
Sequential evaluation module for 2-layer neural networks without requiring noise refresh via bootstrapping.

* **Layer Chaining:** Connects the encrypted output of the first dense layer directly as input to the second dense layer.
* **Depth Management:** Adjusts multiplicative depth parameters to accommodate two layer multiplications and two polynomial activation evaluations.
* **Data Reorganization:** Handles repacking or alignment of intermediate encrypted vectors between Layer 1 and Layer 2.

---

### 4. 2-Layer Neural Network with Bootstrapping (`multi_layer/2layer-fhe-bootstrap.py`)
Module introducing noise budget refresh in a 2-layer architecture.

* **Bootstrapping Integration:** Enables the `BOOTSTRAPPING` feature in OpenFHE to reset ciphertext levels between Layer 1 and Layer 2.
* **Precision Preservation:** Ensures that precision loss due to accumulated noise from first-layer multiplications does not degrade second-layer accuracy.
* **Bootstrapping Keys:** Configures and generates advanced bootstrapping keys within the CKKS cryptographic context.

---

### 5. 4-Layer Neural Network with Bootstrapping (`multi_layer/4layer-bootstrap.py`)
Advanced module extending private inference capability to deep **4-layer** neural networks.

* **Multi-stage Bootstrapping Support:** Runs sequential bootstrapping steps to refresh ciphertext noise across all 4 dense layers.
* **Deep Propagation:** Evaluates inference across 4 sequential layers, reapplying SIMD packing, homomorphic matrix multiplication, sum-tree reduction, and polynomial activations.
* **Dynamic Level Management:** Switches cryptographic key levels (`KeySwitching` and `LeveledSHE`) to maintain high numerical precision across the entire network architecture.

---

## 🛠 Function Reference Table

| Function | Description |
| :--- | :--- |
| `create_crypto_context(depth, batch_size)` | Configures the CKKS cryptographic context in OpenFHE. |
| `gen_keys(context)` | Generates public, secret, multiplication, and rotation keys. |
| `get_data_weight()` | Provides input data vectors and weight matrices for multiple neurons. |
| `pack_data_for_layer(data, weights_matrix, num_neurons)` | Packs data and weights into 1D arrays for parallel execution via SIMD. |
| `enc(message, context, keys)` | Encodes cleartext into CKKS packed format and encrypts into ciphertext. |
| `dec_and_print(ctxt, keys, openFheContext)` | Decrypts ciphertext and prints cleartext results. |
| `eval_sum_tree(context, ctxt, vector_size)` | Executes logarithmic $O(\log N)$ sum-tree reduction via rotations and additions. |
| `eval_layer(context, ctxt_packed_data, ctxt_packed_weights, block_size)` | Executes dense layer multiplication, sum-tree reduction, and polynomial activation. |
| `calculate_cleartext_expected(data, weights_matrix)` | Computes expected dot products and activation results in cleartext. |

---

## 💻 How to Run

### Prerequisites
1. **Python 3+**.
2. **OpenFHE** (C++) library and its **Python bindings** installed.

### Module Execution

```bash
# 1. Single Neuron Validation
python3 single_neuron/test_ckks.py

# 2. Single Homomorphic Dense Layer (1 Layer)
python3 multi_layer/1layer-fhe.py

# 3. 2-Layer Homomorphic Neural Network (without Bootstrapping)
python3 multi_layer/2layer-fhe.py

# 4. 2-Layer Neural Network with Bootstrapping
python3 multi_layer/2layer-fhe-bootstrap.py

# 5. 4-Layer Neural Network with Bootstrapping
python3 multi_layer/4layer-bootstrap.py
```