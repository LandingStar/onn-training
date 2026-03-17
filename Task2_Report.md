# Task 2: Photonic Autoencoder Implementation Report

## 1. Overview
We have successfully implemented a Photonic Encoder-Decoder (PED) system as described in the reference paper (*Science Advances 2023*). The system is designed for unsupervised learning of optical communication schemes, featuring:
- **All-Optical Autoencoder Architecture**: Simulating optical propagation and diffraction.
- **Fiber Coupling Layer**: Modeling the physical coupling of light into single-mode fibers using mode overlap integrals.
- **Physics-Informed Loss Function**: Combining Reconstruction Loss (MSE), Latent Space Regularization (KL Divergence), and Optical Constraints (L1).

## 2. Implementation Details

### 2.1 Model Architecture (`model.py`)
- **Diffractive Layers**: Implemented using Angular Spectrum Method (ASM) for accurate wave propagation simulation.
- **Fiber Coupling**:
  - Corrected the coupling mechanism to use **Overlap Integral** between the incident field and the fundamental fiber mode ($LP_{01}$), rather than simple intensity summation.
  - This ensures complex amplitude is correctly captured and propagated.
  - Shape handling ensures compatibility with batch processing.

### 2.2 Training Process (`train.py`)
- **Loss Function**: $L = \alpha \cdot L_{KL} + \beta \cdot L_{MSE} + \gamma \cdot L_{OP}$
  - $L_{KL}$: Kullback-Leibler divergence to regularize the latent space (intensity distribution).
  - $L_{MSE}$: Mean Squared Error for image reconstruction.
  - $L_{OP}$: Optical penalty (L1 norm) to enforce sparsity or energy constraints.
- **Data**: Synthetic digital samples generated (`data.py`) with Gaussian blobs representing encoded bits, simulating optical signals.

### 2.3 Evaluation
- **Visual Inspection**: The system successfully reconstructs input patterns.
- **Latent Space**: The bottleneck layer (Fiber Coupling) forces the model to learn efficient optical representations.

## 3. Results
- **Training Stability**: The model converges within 5 epochs on the synthetic dataset.
- **Reconstruction**: 
  - Input images (Digital patterns) are encoded into the optical domain, passed through the fiber model, and decoded back.
  - Evaluation plots (`results/evaluation.png`) show clear reconstruction of the input bits.

## 4. Future Work / Improvements
- **Real Datasets**: Integrate MNIST/Fashion-MNIST as per the paper's "Data-Specific Mode".
- **Noise Modeling**: Enhance the fiber channel model with more realistic noise (Phase noise, ASE noise) as described in the Supplementary Materials.
- **Hardware-in-the-loop**: If experimental hardware is available, replace the simulation layers with real SLM/Camera feedback.
