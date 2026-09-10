# FHE-PPML 

**FHE-PPML** (Fully Homomorphic Encryption for Privacy-Preserving Machine Learning) is a project dedicated to exploring and implementing neural network components using Homomorphic Encryption, ensuring that data remains encrypted during inference.

This repository uses the **OpenFHE** library to perform operations on encrypted data.

---

## 📂 Project Structure

As the project evolves, it will be divided into different modules to test various neural network architectures, from a single neuron to multi-layer deep learning models.

- `/single_neuron/` - Implementation and testing of a single neuron using the CKKS scheme.
- `/multi_layer/` - *(Coming Soon)* Tests and implementations for neural networks with multiple layers.

---

## 🧪 Single Neuron Test (`test_ckks.py`)

This section of the project focuses on validating the foundational operations of a neural network under encryption.

### Overview
The code `test_ckks.py` simulates a single neural network neuron using FHE (CKKS scheme) by computing a homomorphic dot product between inputs and weights, followed by a polynomial activation function. It also includes plaintext calculations to validate the decrypted results.

### Features
* **Encrypted Dot Product:** Computes the weighted sum of inputs using FHE rotations and additions (`EvalRotate` and `EvalAdd`).
* **Polynomial Activation:** Applies a polynomial approximation (e.g., $f(x) = c_0 + c_1x + c_2x^2$) to the encrypted result using `EvalPoly`.
* **Plaintext Validation:** Runs the exact same mathematical operations in classical plaintext before encryption, allowing for direct comparison and accuracy checking of the CKKS approximation.

### How to Run

1. Ensure you have [OpenFHE](https://github.com/openfheorg/openfhe-development) installed and properly configured with your Python environment.
2. Execute the script:
   ```bash
   python3 test_ckks.py