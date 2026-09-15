# FHE-PPML: Fully Homomorphic Encryption for Privacy-Preserving Machine Learning

**FHE-PPML** is a project dedicated to exploring and implementing neural network components using Fully Homomorphic Encryption (FHE), ensuring that sensitive data remains encrypted during inference. This repository utilizes the **OpenFHE** library and the **CKKS** scheme to perform mathematical operations directly on ciphertexts.

---

## 📂 Project Structure

The repository is organized into distinct modules for testing single-neuron logic and multi-neuron dense layer operations:

* **`single_neuron/test_ckks.py`**: Implementation and testing of a single neuron using the CKKS scheme.
* **`multi_layer/1layer-fhe.py`**: Implementation and testing of a single dense neural network layer using SIMD vector packing.

---

## 🧪 1. Single Neuron (`single_neuron/test_ckks.py`)

This module validates foundational homomorphic operations for an individual neuron.

### Features & Workflow
* **Cryptographic Setup:** Configures `CCParamsCKKSRNS` with a multiplicative depth of 5, scaling modulus size of 50 bits, and batch size of 8.
* **Key Generation:** Generates public and secret keys, evaluation multiplication keys (`EvalMultKeyGen`), and rotation keys (`EvalAtIndexKeyGen`) for shifts `[1, 2, 4]`.
* **Ciphertext-Plaintext Multiplication:** Encrypts input data (`ctxt_data`) and multiplies it homomorphically with plaintext weights (`ptxt_weights`) via `EvalMult`.
* **Logarithmic Sum Reduction:** Computes the vector dot product using homomorphic rotations (`EvalRotate`) and additions (`EvalAdd`) across index shifts of 1, 2, and 4.
* **Polynomial Activation:** Evaluates a quadratic polynomial activation $f(x) = 0.1 + 0.5x + 1.0x^2$ on the encrypted sum using `EvalPoly` with coefficients `[0.1, 0.5, 1.0]`.
* **Plaintext Verification:** Decrypts the output ciphertext (`ctxt_activation`) and compares it against explicit cleartext expected results.

---

## 🧬 2. Homomorphic Dense Layer (`multi_layer/1layer-fhe.py`)

This module evaluates a dense neural network layer with multiple neurons in parallel using SIMD vector operations.

### Features & Workflow
* **Cryptographic Setup:** Configures `CCParamsCKKSRNS` with a multiplicative depth of 4, scaling modulus size of 50 bits, and batch size of 32.
* **Key Generation:** Enables `KeySwitching`, `LeveledSHE`, `PKE`, and `AdvancedSHE`, generating evaluation multiplication keys and rotation keys for indices `[1, 2, 4, 8, -1, -2, -3]`.
* **SIMD Data Packing (`pack_data_for_layer`):** Flattens weight matrices and repeats input data vectors into single 1D arrays to process multiple neurons simultaneously in one ciphertext batch.
* **Homomorphic Layer Evaluation (`eval_layer`):**
  * **Homomorphic Multiplication:** Multiplies packed encrypted data and packed encrypted weights using `EvalMult`.
  * **Sum Tree Reduction (`eval_sum_tree`):** Executes logarithmic reduction using `EvalAtIndex` and `EvalAdd` to sum vector blocks.
  * **Polynomial Sigmoid Activation:** Approximates the Sigmoid activation function homomorphically using `EvalPoly` with coefficients `[0.5, 0.197, 0.0, -0.004]` ($f(x) = -0.004x^3 + 0.197x + 0.5$).
* **Decryption & Noise Metrics:** Decrypts the layer output, extracts valid neuron output indices (`i * len(data)`), and compares homomorphic results against cleartext calculations (`calculate_cleartext_expected`) to track absolute noise error.

---

## 🛠️ Function Reference

| Function | Description |
| :--- | :--- |
| `create_crypto_context(depth, batch_size)` | Configures the CKKS cryptographic context in OpenFHE. |
| `gen_keys(context)` | Generates public, secret, evaluation multiplication, and rotation keys. |
| `get_data_weight()` | Supplies sample input data vectors and multi-neuron weight matrices. |
| `pack_data_for_layer(data, weights_matrix, num_neurons)` | Packs data and weights into 1D arrays for parallel SIMD execution. |
| `enc(message, context, keys)` | Encodes plaintext into CKKS packed plaintext and encrypts into ciphertext. |
| `dec_and_print(ctxt, keys, openFheContext)` | Decrypts ciphertext and prints plaintext results. |
| `eval_sum_tree(context, ctxt, vector_size)` | Performs $O(\log N)$ logarithmic sum tree reduction via rotations and additions. |
| `eval_layer(context, ctxt_packed_data, ctxt_packed_weights, block_size)` | Executes dense layer multiplication, sum tree reduction, and polynomial activation. |
| `calculate_cleartext_expected(data, weights_matrix)` | Computes expected cleartext dot products and Sigmoid activation results. |

---

## 💻 How to Run

### Prerequisites
1. **Python 3.8+**.
2. **OpenFHE** C++ library and Python bindings installed.

### Running Single Neuron Validation
```bash
python3 single_neuron/test_ckks.py
```

### Running Homomorphic Dense Layer
```bash
python3 multi_layer/1layer-fhe.py
```