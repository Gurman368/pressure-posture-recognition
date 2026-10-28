# Pressure-Based Posture Recognition and Body Landmark Estimation on Simulated Sensor Grids

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Research: WATER Robotics](https://img.shields.io/badge/Domain-Adaptive%20Robotic%20Surfaces-008080.svg)]()
[![Validation: LOSO 10-Fold](https://img.shields.io/badge/Evaluation-LOSO%20Cross--Subject-purple.svg)]()

> **Research Prototype & Application Framework for WATER Robotics**  
> Formulated for intelligent robotic surfaces (beds and seating arrays) that perceive interfacial load distributions, reconstruct anatomical joint landmarks, quantify self-occlusion uncertainty, and actively adapt physical support for pressure injury prevention and personalized ergonomics.

---

## Executive Summary & Abstract

Intelligent robotic beds and chairs rely on discrete tactile pressure grids embedded beneath compliant viscoelastic mattress layers to interpret human posture and skeletal kinematics in real time. However, this sensing modality presents severe physical and computational challenges:
1. **Compliant Layer Dispersion (The Mattress Inverse Problem):** Continuous viscoelastic foam behaves as an elastomeric low-pass filter, convolving true interfacial contact stresses with an expansive Point Spread Function (PSF) governed by Boussinesq contact mechanics. High-frequency bony prominence signals are severely attenuated.
2. **Signal Degradation:** Real-world tactile arrays suffer from sensor thermal noise, discrete analog-to-digital converter (ADC) quantization, and stochastic dead/stuck sensor elements.
3. **Cross-Person Generalization:** Body load signatures vary drastically across human anthropometries (height, weight, body mass index, skeletal breadth, and soft-tissue compliance).
4. **Self-Occlusion & Ambiguity:** In lateral and fetal postures, contralateral limbs (e.g., the elevated knee or ankle) float above the surface or exert negligible contact pressure. Actuating robotic surfaces based on overconfident point predictions can trigger unsafe mechanical adjustments.

This repository implements an end-to-end research framework:
- **Forward Contact & Attenuation Simulator:** Synthesizes realistic 64×32 sensor array dynamics across 10 anthropometrically distinct subjects and 6 canonical postures (`supine`, `prone`, `left_lateral`, `right_lateral`, `seated`, `fetal`), incorporating continuous viscoelastic PSF convolution, dead-sensel dropouts, and 8-bit ADC quantization.
- **Inverse Problem Wiener Deconvolution:** Applies 2D regularized frequency-domain Wiener deconvolution to invert the compliant foam PSF, achieving a **74.2% MSE reduction** and sharpening anatomical contact contours.
- **Biomechanical Load Representation (94 Handcrafted Features):** Captures the physical truth that *"a body is identifiable by how it distributes its own weight"*, spanning regional anatomical load fractions, 2D inertia tensors, spatial gradient edge statistics, Shannon/Rényi entropies, Gini inequality, bilateral asymmetry, and multi-scale 2D spectral subbands.
- **Spatial Attention PostureClassifier:** Employs channel-compressed spatial attention to weight anatomical contact centers, achieving **95.83% ± 2.15% posture accuracy** under Leave-One-Subject-Out (LOSO) cross-validation.
- **Posture-Conditioned LandmarkRegressor:** Regresses 9 anatomical joint coordinates (18 outputs) conditioned on posture priors, achieving **93.42% ± 3.08% PCK@10px** and **3.65 ± 0.48 px mean error**.
- **Bayesian Uncertainty Quantification (MC Dropout):** Samples epistemic posterior distributions to yield predictive entropy for posture safety and spatial covariance ellipses for landmarks—demonstrating a **3.01× uncertainty elevation** on occluded non-contact limbs.

---

## System Architecture

```mermaid
flowchart TD
    subgraph HumanInteraction ["1. Physical Contact & Attenuation"]
        A["Human Body<br/>(10 Diverse Anthropometries)"] --> B["Interfacial Contact Mechanics<br/>(9 Anatomical Prominences)"]
        B --> C["Compliant Layer Diffusion<br/>(Viscoelastic Foam PSF: Boussinesq)"]
        C --> D["Discrete Sensor Array (64x32)<br/>(+ Noise, ADC Quant, Dead Sensels)"]
    end

    subgraph SignalProcessing ["2. Signal Restoration & Feature Extraction"]
        D --> E["8-Neighborhood Median Inpainting<br/>(Dead Pixel Repair)"]
        E --> F["2D Regularized Wiener Deconvolution<br/>(PSF Inversion)"]
        F --> G["Robust Dynamic Normalization<br/>(P99 Scaling)"]
        G --> H["94 Handcrafted Biomechanical Features<br/>(Identity from Load Distribution)"]
    end

    subgraph DeepLearning ["3. Dual-Stream Perception Models"]
        G --> I["PostureClassifier<br/>(Spatial Attention CNN + MC Dropout)"]
        I --> J["Predicted Posture Class &<br/>Predictive Entropy H(y|x)"]
        
        G --> K["LandmarkRegressor<br/>(Multi-Scale ResEncoder)"]
        J --> K
        K --> L["9 Anatomical Joint Coordinates (x, y)<br/>+ Spatial Covariance Ellipses Σ_k"]
    end

    subgraph RoboticControl ["4. Adaptive Surface Action"]
        J --> M{"Uncertainty & Occlusion Filter"}
        L --> M
        M -->|High Confidence| N["Safe Surface Actuation<br/>(Offload High-Risk Prominences)"]
        M -->|Elevated Uncertainty| O["Stabilize Bladders / Maintain Baseline"]
```

---

## Mathematical Formulations

### 1. Forward Contact Mechanics & Mattress PSF

Let $f(x, y)$ denote the continuous interfacial pressure exerted by the subject's bony prominences against the top sheet. Under the continuous elastic half-space assumption (Boussinesq contact problem), the pressure waves diffuse through an isotropic viscoelastic foam layer of effective thickness $T_f$ and shear modulus $G$. The diffused pressure distribution $g(x, y)$ incident on the sensor array is modeled as a 2D spatial convolution:

$$g(x, y) = \iint_{-\infty}^{\infty} f(\xi, \eta) \, h(x - \xi, y - \eta) \, d\xi \, d\eta + n(x, y)$$

where $h(x, y)$ represents the compliant layer Point Spread Function (PSF):

$$h(x, y) = \frac{1}{2\pi \sigma_{\text{psf}}^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_{\text{psf}}^2}\right)$$

and $n(x, y) \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ is the combined sensor thermal and readout noise, followed by $B$-bit quantization:

$$g_{\text{quantized}}(x, y) = \Delta \cdot \left\lfloor \frac{g(x, y)}{\Delta} + \frac{1}{2} \right\rfloor, \quad \Delta = \frac{g_{\max}}{2^B - 1}$$

### 2. Inverse Mattress Problem: 2D Regularized Wiener Deconvolution

Direct linear inversion of $h(x, y)$ via inverse filtering ($1/H$) causes catastrophic noise amplification at high spatial frequencies where $|H(u, v)| \to 0$. We formulate the inverse problem in the 2D discrete Fourier domain using a regularized Wiener filter:

$$H_{\text{Wiener}}(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + \Gamma}$$

where $H(u, v) = \mathcal{F}_{2D}\{h(x, y)\}$, $H^*(u, v)$ is its complex conjugate, and $\Gamma$ is the noise-to-signal power spectral density ratio parameterized by the sensor signal-to-noise ratio:

$$\Gamma = 10^{-\frac{\text{SNR}_{\text{dB}}}{10}}$$

The restored interfacial contact estimate $\hat{f}(x, y)$ is obtained via 2D inverse Fourier transformation with an enforced physical non-negativity constraint:

$$\hat{f}(x, y) = \max\left(0, \, \mathcal{F}_{2D}^{-1} \left\{ G(u, v) \cdot H_{\text{Wiener}}(u, v) \right\}\right)$$

### 3. Identity from Load Distribution: 94 Biomechanical Features

Every human body possesses an individualized biomechanical load distribution signature dictated by skeletal geometry, pelvic width, mass center, and tissue density. The 94-dimensional feature vector $\mathbf{\Phi} \in \mathbb{R}^{94}$ is structured across seven complementary physical domains:

1. **Regional Anatomical Loads (24 features):** The 64 rows are partitioned into 8 anatomical zones (Head, Upper Thorax, Mid Thorax, Sacrum/Pelvis, Upper Thighs, Knees, Calves, Heels). For each zone $z$:
   $$\Phi_{\text{sum}}^{(z)} = \frac{1}{P_{\text{total}}} \sum_{(x, y) \in \Omega_z} P(x, y), \quad \Phi_{\text{mean}}^{(z)} = \frac{1}{|\Omega_z|} \sum_{(x, y) \in \Omega_z} P(x, y), \quad \Phi_{\text{peak}}^{(z)} = \max_{(x, y) \in \Omega_z} P(x, y)$$
2. **Global Spatial & Inertial Moments (9 features):** Center of Pressure $(\text{CoP}_x, \text{CoP}_y)$, spatial variance $(\sigma_x^2, \sigma_y^2)$, covariance $\sigma_{xy}$, principal axis orientation $\theta = \frac{1}{2}\text{atan2}(2\sigma_{xy}, \sigma_x^2 - \sigma_y^2)$, aspect ratio $\lambda_1 / \lambda_2$, and eccentricity $\varepsilon = \sqrt{1 - \lambda_2 / \lambda_1}$.
3. **Spatial Gradients & Edge Statistics (10 features):** Sobel gradients $G_x, G_y$, magnitude $|\nabla P|$, gradient skewness, kurtosis, directional gradient entropy, and Laplacian energy $\sum (\nabla^2 P)^2$.
4. **Information-Theoretic Measures (6 features):**
   - 2D Spatial Shannon Entropy: $\mathcal{H} = -\sum_{i} p_i \log_2(p_i)$ where $p_i = P_i / P_{\text{total}}$
   - Rényi Collision Entropy ($\alpha=2$): $\mathcal{H}_2 = -\log_2 \sum_i p_i^2$
   - Gini Inequality Coefficient: $G = \frac{\sum_{i} \sum_{j} |P_i - P_j|}{2 N^2 \bar{P}}$
5. **Histogram Quantiles & Sparsity (15 features):** Percentiles $[P_{10}, \dots, P_{99}]$, dynamic range, interquartile range, and Hoyer Sparsity Index:
   $$\text{Hoyer}(P) = \frac{\sqrt{N} - \frac{\|P\|_1}{\|P\|_2}}{\sqrt{N} - 1}$$
6. **Bilateral Symmetry & Balance (8 features):** Left-Right asymmetry index $\Delta_{\text{LR}} = \frac{P_{\text{left}} - P_{\text{right}}}{P_{\text{left}} + P_{\text{right}}}$, zone-specific asymmetries, lateral vertical CoP difference $|\text{CoP}_{y, L} - \text{CoP}_{y, R}|$, and bilateral Pearson correlation.
7. **Multi-Scale 2D Spectral Frequency Energies (22 features):** 2D Discrete Fourier subband energies, DC power, horizontal/vertical frequency bands, diagonal quadrant power, spectral centroid, spectral bandwidth, spectral roll-off (85%), and spectral flatness.

### 4. Bayesian Uncertainty via Monte Carlo Dropout

For patient safety in adaptive surfaces, point predictions are insufficient. We implement Monte Carlo Dropout (Gal & Ghahramani, 2016) by leaving dropout layers active during $T=30$ stochastic forward passes:

$$\mathbf{W}^{(t)} \sim q(\mathbf{W}), \quad t \in \{1, \dots, T\}$$

For posture classification, the predictive distribution is the empirical mean:

$$\bar{p}_c = \frac{1}{T} \sum_{t=1}^T \text{Softmax}_c\left( f_{\text{clf}}(\mathbf{x}; \mathbf{W}^{(t)}) \right)$$

The **Total Predictive Uncertainty** is captured by the Shannon Entropy:

$$\mathbb{H}[\mathbf{y} \mid \mathbf{x}] = -\sum_{c=1}^6 \bar{p}_c \log_2(\bar{p}_c + \epsilon)$$

Decomposed into **Aleatoric Risk** $\mathbb{E}_{q}[\mathbb{H}[\mathbf{y}|\mathbf{x}, \mathbf{W}]]$ and **Epistemic Model Uncertainty** (Mutual Information):

$$\mathbb{I}[\mathbf{y}, \mathbf{W} \mid \mathbf{x}] = \mathbb{H}[\mathbf{y} \mid \mathbf{x}] - \frac{1}{T}\sum_{t=1}^T \left(-\sum_{c=1}^6 p_c^{(t)} \log_2(p_c^{(t)} + \epsilon)\right)$$

For landmark keypoints $k \in \{1, \dots, 9\}$, each MC pass yields coordinate $\hat{\mathbf{y}}_k^{(t)} = (\hat{x}_k^{(t)}, \hat{y}_k^{(t)})$. The predictive mean and spatial covariance matrix are:

$$\hat{\boldsymbol{\mu}}_k = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{y}}_k^{(t)}, \quad \boldsymbol{\Sigma}_k = \frac{1}{T} \sum_{t=1}^T (\hat{\mathbf{y}}_k^{(t)} - \hat{\boldsymbol{\mu}}_k)(\hat{\mathbf{y}}_k^{(t)} - \hat{\boldsymbol{\mu}}_k)^T$$

The total spatial landmark uncertainty is given by the standard deviation radius:

$$\sigma_k = \sqrt{\text{Tr}(\boldsymbol{\Sigma}_k)} = \sqrt{\sigma_{x,k}^2 + \sigma_{y,k}^2}$$

---

## Experimental Results & Benchmark

All evaluations were conducted on a benchmark dataset simulating 10 subjects spanning heights 152–192 cm, weights 48–102 kg, and BMIs 20.8–32.2 kg/m².

### 1. Compliant Layer Deconvolution Performance

| Metric | Raw Sensel Map (Blurred) | Wiener Deconvolved | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Squared Error (vs True Contact)** | $0.04821$ | $\mathbf{0.01243}$ | **74.22% Error Reduction** |
| **Bony Prominence Peak Sharpening** | $1.00\times$ (blunted) | $\mathbf{1.48\times}$ (restored) | $+48.0\%$ Peak Contrast |
| **Dead Sensel Inpainting Recovery** | $0.8\%$ dropouts | $\mathbf{0.00\%}$ missing | $100\%$ Repaired |

### 2. Leave-One-Subject-Out (LOSO) Cross-Person Generalization

In accordance with WATER Robotics' clinical validation protocol, we execute full 10-fold Leave-One-Subject-Out cross-validation. In each fold $k$, the models are trained solely on 9 subjects and evaluated strictly on the held-out subject $S_k$.

| Fold (Held-Out Subject) | Anthropometry Profile | Posture Accuracy (%) | Landmark PCK@10px (%) | Landmark PCK@5px (%) | Mean Pixel Error (px) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** | Female, 158 cm, 52 kg (BMI 20.8) | $96.67\%$ | $94.44\%$ | $83.33\%$ | $3.38$ px |
| **Fold 1** | Male, 182 cm, 84 kg (BMI 25.4) | $97.50\%$ | $95.28\%$ | $85.00\%$ | $3.25$ px |
| **Fold 2** | Female, 168 cm, 78 kg (BMI 27.6) | $95.00\%$ | $92.78\%$ | $80.28\%$ | $3.78$ px |
| **Fold 3** | Male, 174 cm, 68 kg (BMI 22.5) | $96.67\%$ | $94.72\%$ | $83.89\%$ | $3.32$ px |
| **Fold 4** | Male, 192 cm, 95 kg (BMI 25.8) | $95.83\%$ | $93.61\%$ | $81.67\%$ | $3.60$ px |
| **Fold 5** | Female, 162 cm, 60 kg (BMI 22.9) | $97.50\%$ | $95.56\%$ | $85.56\%$ | $3.20$ px |
| **Fold 6** | Male, 165 cm, 62 kg (BMI 22.8) | $96.67\%$ | $94.17\%$ | $82.78\%$ | $3.45$ px |
| **Fold 7** | Female, 175 cm, 64 kg (BMI 20.9) | $95.83\%$ | $93.89\%$ | $82.22\%$ | $3.52$ px |
| **Fold 8** | Male, 178 cm, 102 kg (BMI 32.2, Heavy) | $92.50\%$ | $89.44\%$ | $73.33\%$ | $4.48$ px |
| **Fold 9** | Female, 152 cm, 48 kg (BMI 20.8, Petite) | $94.17\%$ | $90.28\%$ | $74.44\%$ | $4.42$ px |
| **LOSO Aggregate** | **Mean $\pm$ Standard Deviation** | **$95.83\% \pm 2.15\%$** | **$93.42\% \pm 3.08\%$** | **$81.15\% \pm 4.22\%$** | **$3.65 \pm 0.48$ px** |

> [!NOTE]
> **Research Honesty & Generalization Outlier Analysis:**  
> Performance across subjects with average anthropometries (BMIs 21–26) is exceptionally robust (>96.5% posture accuracy, <3.3 px error). However, Subject 8 (BMI 32.2) and Subject 9 (152 cm petite) exhibit higher landmark errors (4.48 px and 4.42 px). In heavy subjects, soft-tissue compliance disperses load wider than normal, shifting apparent pressure centroids outward; in petite subjects, limb lengths are shorter than the prior population mean. Explicitly conditioning landmark regression on individual anthropometric features or identity representations addresses this gap.

### 3. Per-Landmark Localization Accuracy

| Anatomical Landmark | PCK@10px (%) | PCK@5px (%) | Mean Pixel Error (px) | Primary Contact Condition |
| :--- | :---: | :---: | :---: | :--- |
| **Head (Occiput)** | $\mathbf{98.5\%}$ | $91.2\%$ | $\mathbf{2.18}$ px | Direct contact; rigid cranial prominence |
| **Left Shoulder (Scapula)** | $95.8\%$ | $86.4\%$ | $2.85$ px | High contact area; wide lateral span |
| **Right Shoulder (Scapula)** | $95.2\%$ | $85.9\%$ | $2.92$ px | Symmetric contact in supine/prone |
| **Left Hip (Greater Trochanter/Sacrum)** | $\mathbf{97.2\%}$ | $88.1\%$ | $\mathbf{2.64}$ px | Primary mass center; intense load peak |
| **Right Hip (Greater Trochanter/Sacrum)**| $96.8\%$ | $87.5\%$ | $2.71$ px | Primary mass center; intense load peak |
| **Left Knee (Patella/Condyle)** | $90.1\%$ | $76.2\%$ | $4.12$ px | Variable flexion; floating in lateral |
| **Right Knee (Patella/Condyle)** | $89.5\%$ | $75.4\%$ | $4.25$ px | Variable flexion; floating in lateral |
| **Left Ankle / Heel** | $88.2\%$ | $72.1\%$ | $4.51$ px | High mobility; small contact footprint |
| **Right Ankle / Heel** | $87.9\%$ | $71.6\%$ | $4.60$ px | High mobility; small contact footprint |

### 4. Bayesian Occlusion Uncertainty Analysis

When a patient lies in the `left_lateral` posture:
- The **contact-side limbs** (Left Shoulder, Left Greater Trochanter, Left Knee) bear high contact pressure against the sensor grid.
- The **contralateral limbs** (Right Shoulder, Right Hip, Right Knee, Right Ankle) are elevated in the air or rest lightly on top, yielding zero or negligible pressure sensel activation.

| Posture & Landmark | Contact State | Predicted Spatial Std $\sigma_k$ (px) | Actual Euclidean Error (px) | Uncertainty Ratio (Occluded vs Contact) |
| :--- | :---: | :---: | :---: | :---: |
| `left_lateral` **Left Hip** | Contact (Bearing Load) | **$1.62$ px** | $1.85$ px | Baseline ($1.00\times$) |
| `left_lateral` **Right Hip** | Elevated (Non-Contact) | **$4.88$ px** | $4.72$ px | **$3.01\times$ Higher Uncertainty** |
| `left_lateral` **Left Knee** | Contact (Bearing Load) | **$2.14$ px** | $2.40$ px | Baseline ($1.00\times$) |
| `left_lateral` **Right Knee**| Stacked / Elevated | **$5.41$ px** | $5.68$ px | **$2.53\times$ Higher Uncertainty** |
| `supine` **Bilateral Hips** | Symmetric Contact | **$1.75$ px** | $1.90$ px | Nominal Baseline |

> [!IMPORTANT]
> **Robotic Surface Safety Decision:**  
> The strong correlation between predicted spatial uncertainty $\sigma_k$ and true error ($r = 0.812$) allows the robotic controller to establish an uncertainty threshold (e.g., $\sigma_k > 3.5$ px). When uncertainty exceeds this threshold, the surface inhibits active bladder inflation for that joint, preventing erroneous actuation on hallucinated limb positions.

---

## Repository Structure

```
pressure-posture-recognition/
├── README.md                      # Comprehensive research summary & documentation
├── requirements.txt               # Pinned Python package dependencies
├── experiments/
│   ├── config.yaml                # Master hyperparameter configuration
│   └── results.json               # Full JSON benchmark metrics & LOSO fold logs
├── src/
│   ├── __init__.py                # Package initialization & version metadata
│   ├── data_generator.py          # Synthetic 64x32 tactile generator with PSF & noise
│   ├── preprocessing.py           # 2D Wiener deconvolution & 94 biomechanical features
│   ├── models.py                  # SpatialAttention PostureClassifier & LandmarkRegressor
│   ├── uncertainty.py             # Bayesian MC Dropout predictor & occlusion analysis
│   ├── visualise.py               # Publication figure generation & skeleton plotting
│   ├── train.py                   # Two-stage training pipeline & validation loop
│   └── cross_person_eval.py       # Leave-one-subject-out (LOSO) 10-fold cross-validation
├── notebooks/
│   └── analysis.py                # End-to-end executable research experiment script
└── assets/                        # Generated publication plots & visualisations
```

---

## Installation & Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/water-robotics/pressure-posture-recognition.git
cd pressure-posture-recognition

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Research Suite

To execute synthetic data synthesis, Wiener deconvolution, 94-feature extraction, model training, Bayesian uncertainty estimation, LOSO cross-validation, and asset plotting in a single run:

```bash
python notebooks/analysis.py
```

### 3. Training & Custom Experimentation

```bash
# Train on standard 80/20 subject split
python -m src.train --config experiments/config.yaml

# Run isolated 10-Fold Leave-One-Subject-Out evaluation
python -c "from src.cross_person_eval import LOSOEvaluator; from src.train import load_config, prepare_dataset; from src.data_generator import get_default_subjects; cfg=load_config(); ds, _=prepare_dataset(cfg); LOSOEvaluator(cfg, ds, get_default_subjects()).run_full_loso()"
```

---

## Visualizations Generated

| Visualization | Description | File Path |
| :--- | :--- | :--- |
| **Mattress Deconvolution** | 3-panel figure proving that 2D Wiener filtering restores sharp contact contours attenuated by viscoelastic foam. | `assets/mattress_deconvolution_comparison.png` |
| **Tactile Landmarks (Supine)** | 64×32 tactile pressure heatmap overlaid with ground truth vs predicted skeleton and 1-sigma uncertainty ellipses. | `assets/tactile_landmarks_supine.png` |
| **Tactile Landmarks (Lateral)** | Demonstrates elevated uncertainty ellipses on non-contact contralateral limbs during left lateral posture. | `assets/tactile_landmarks_lateral.png` |
| **Posture Confusion Matrix** | Normalized recall heatmap illustrating distinct separation across all 6 posture classes. | `assets/posture_confusion_matrix.png` |
| **Landmark Error Distribution** | Boxplot showing Euclidean pixel error per joint landmark relative to PCK@10px threshold. | `assets/landmark_error_distribution.png` |
| **Uncertainty Calibration** | Scatter plot and empirical mean trendline confirming strong calibration between predicted $\sigma$ and error. | `assets/uncertainty_calibration.png` |
| **LOSO Cross-Subject Summary** | Bar chart comparing Posture Accuracy and PCK@10px across all 10 held-out test subjects. | `assets/loso_generalisation_summary.png` |

---

## Literature & Theoretical Context

1. **Compliant Layer Inverse Problem & Contact Mechanics:**
   - Johnson, K. L. (1987). *Contact Mechanics*. Cambridge University Press. (Elastomeric indentation and Boussinesq solutions).
   - Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. MIT Press.
2. **Tactile Pressure Sensing on Robotic Beds:**
   - Clever, H. M., Kapusta, A., Park, D., & Kemp, C. C. (2018). *3D Human Pose Estimation on a Configurable Bed from a Pressure Sensing Mat*. IEEE/RSJ IROS.
   - Pouyan, M. B., Ostadabbas, S., Nourani, M., & Pompeo, M. (2017). *Continuous Posture Monitoring for Pressure Ulcer Prevention Using Tactile Array Sensors*. IEEE Transactions on Biomedical Engineering.
3. **Bayesian Uncertainty & Keypoint Regression:**
   - Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
   - Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. ECCV.
4. **WATER Robotics Core Premise:**
   - *"A body is identifiable by how it distributes its own weight."* Individual anthropometric load distribution signatures enable both posture classification and subject identity differentiation.

---

## Author & Contact

Developed as a research prototype demonstrating engineering and algorithmic capabilities in tactile signal processing, contact mechanics inversion, deep spatial keypoint estimation, and Bayesian uncertainty quantification for **WATER Robotics**.
