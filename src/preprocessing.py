"""
Signal Processing and Preprocessing Pipeline for Tactile Pressure Arrays.

Key Modules:
1. Inverse Mattress Problem: 2D Regularized Wiener Deconvolution to invert
   the compliant viscoelastic foam Point Spread Function (PSF).
2. Sensor Fault Inpainting: 8-neighborhood median/interpolation for dead/stuck sensels.
3. Robust Normalisation: Percentile-based dynamic range scaling and per-subject standardisation.
4. Biomechanical Feature Extraction: 94 handcrafted physical features representing
   load distributions, anatomical moments, spatial gradients, bilateral asymmetry,
   information-theoretic entropy, and multi-scale spectral energies.
   (Directly captures biomechanical identity-from-load-distribution principles).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import sobel, laplace, generic_filter
from scipy.stats import skew, kurtosis
from scipy.signal import convolve2d


class PressurePreprocessor:
    """
    Production-grade signal processing pipeline for raw tactile mattress arrays.
    """

    def __init__(
        self,
        psf_sigma: float = 1.8,
        psf_kernel_size: int = 9,
        wiener_snr_db: float = 22.0,
        normalization_method: str = "minmax_robust"
    ):
        self.psf_sigma = psf_sigma
        self.psf_kernel_size = psf_kernel_size
        self.wiener_snr_db = wiener_snr_db
        self.normalization_method = normalization_method

        # Build 2D Gaussian PSF kernel for deconvolution
        self.psf_kernel = self._create_gaussian_psf(psf_kernel_size, psf_sigma)

    @staticmethod
    def _create_gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
        """Constructs a normalized 2D Gaussian Point Spread Function."""
        radius = kernel_size // 2
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        psf = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        psf_sum = psf.sum()
        if psf_sum > 0:
            psf /= psf_sum
        return psf.astype(np.float32)

    def inpaint_dead_pixels(
        self,
        pressure_map: np.ndarray,
        dead_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inpaints missing/faulty sensor pixels using 8-connected local median filtering.
        
        Args:
            pressure_map: (H, W) array of raw sensor readings
            dead_mask: (H, W) boolean mask where True = dead pixel. If None, detected by zero-outliers.
            
        Returns:
            inpainted: (H, W) array with repaired sensor elements
        """
        inpainted = pressure_map.copy()
        H, W = pressure_map.shape

        if dead_mask is None:
            # Heuristic detection: pixel is 0.0 while median of 8 neighbors is > 5% of map peak
            threshold = 0.05 * np.max(pressure_map) if np.max(pressure_map) > 0 else 0.01
            dead_mask = np.zeros((H, W), dtype=bool)
            for r in range(1, H - 1):
                for c in range(1, W - 1):
                    if pressure_map[r, c] < 1e-4:
                        nbrs = pressure_map[r - 1:r + 2, c - 1:c + 2]
                        if np.median(nbrs) > threshold:
                            dead_mask[r, c] = True

        dead_indices = np.argwhere(dead_mask)
        for r, c in dead_indices:
            r_min = max(0, r - 1)
            r_max = min(H, r + 2)
            c_min = max(0, c - 1)
            c_max = min(W, c + 2)

            nbrs = pressure_map[r_min:r_max, c_min:c_max].flatten()
            valid_nbrs = nbrs[nbrs > 1e-5]
            if len(valid_nbrs) > 0:
                inpainted[r, c] = np.median(valid_nbrs)
            else:
                inpainted[r, c] = 0.0

        return inpainted

    def wiener_deconvolution(
        self,
        pressure_map: np.ndarray,
        snr_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Performs 2D Frequency-Domain Regularized Wiener Deconvolution to invert
        mattress compliant layer diffusion:
            W(u, v) = H*(u, v) / (|H(u, v)|^2 + 10^(-SNR_dB / 10))
            
        Args:
            pressure_map: (H, W) pressure array
            snr_db: Signal-to-Noise Ratio in decibels
            
        Returns:
            deconvolved: (H, W) deconvolved contact pressure array
        """
        if snr_db is None:
            snr_db = self.wiener_snr_db

        H, W = pressure_map.shape
        # Pad PSF to grid dimensions
        kh, kw = self.psf_kernel.shape
        padded_psf = np.zeros((H, W), dtype=np.float32)
        r_start = (H - kh) // 2
        c_start = (W - kw) // 2
        padded_psf[r_start:r_start + kh, c_start:c_start + kw] = self.psf_kernel

        # Roll PSF so center is at (0, 0) for FFT alignment
        padded_psf = np.roll(padded_psf, -H // 2, axis=0)
        padded_psf = np.roll(padded_psf, -W // 2, axis=1)

        # 2D FFT
        G = np.fft.fft2(pressure_map)
        H_freq = np.fft.fft2(padded_psf)

        # Regularization parameter gamma = 1 / SNR
        gamma = 10.0 ** (-snr_db / 10.0)

        # Wiener transfer function
        H_conj = np.conj(H_freq)
        H_mag_sq = np.real(H_freq * H_conj)
        W_filter = H_conj / (H_mag_sq + gamma)

        # Inverse FFT
        F_est = G * W_filter
        deconvolved = np.real(np.fft.ifft2(F_est))

        # Enforce non-negativity constraint (pressure cannot be negative)
        deconvolved = np.maximum(deconvolved, 0.0)
        return deconvolved.astype(np.float32)

    def normalize(
        self,
        pressure_map: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Normalizes pressure map for stable neural network feature representation.
        
        Methods:
            - 'minmax_robust': Scaled by 99th percentile to suppress transient spike noise.
            - 'zscore': Zero-mean unit-variance on active contact regions.
            - 'peak': Scaled strictly by maximum value.
        """
        if method is None:
            method = self.normalization_method

        arr = pressure_map.astype(np.float32)
        if method == "minmax_robust":
            p99 = np.percentile(arr, 99.0)
            if p99 > 1e-5:
                normed = np.clip(arr / p99, 0.0, 1.0)
            else:
                normed = np.zeros_like(arr)
            return normed

        elif method == "zscore":
            active = arr[arr > 0.01]
            if len(active) > 10:
                mean = np.mean(active)
                std = np.std(active) + 1e-6
                normed = (arr - mean) / std
                normed[arr <= 0.01] = 0.0
            else:
                normed = arr
            return normed

        elif method == "peak":
            p_max = np.max(arr)
            if p_max > 1e-5:
                normed = arr / p_max
            else:
                normed = np.zeros_like(arr)
            return normed

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def preprocess(
        self,
        raw_sensor_map: np.ndarray,
        dead_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        End-to-end preprocessing pipeline:
        Raw Sensor Map -> Inpainting -> Wiener Deconvolution -> Robust Normalisation.
        """
        inpainted = self.inpaint_dead_pixels(raw_sensor_map, dead_mask)
        deconvolved = self.wiener_deconvolution(inpainted)
        normalized = self.normalize(deconvolved)
        return normalized

    @staticmethod
    def extract_94_biomechanical_features(pressure_map: np.ndarray) -> np.ndarray:
        """
        Extracts 94 handcrafted biomechanical features characterising posture and identity:
        
        - 24 Regional load features (8 anatomical zones x {sum, mean, peak})
        - 9  Spatial & inertial moments (CoP, spatial variance, aspect ratio, eccentricity, orientation)
        - 10 Spatial gradient & edge statistics (Sobel magnitude, skewness, kurtosis, Laplacian)
        - 6  Information-theoretic entropy measures (Shannon, Rényi, Gini, CV)
        - 15 Histogram quantiles & sparsity indices (percentiles, dynamic range, Hoyer index)
        - 8  Bilateral symmetry & left-right imbalance indices
        - 22 Multi-scale 2D spectral & spatial frequency energy metrics
        
        Total: exactly 94 features.
        """
        H, W = pressure_map.shape
        features: List[float] = []

        total_load = np.sum(pressure_map) + 1e-6
        active_mask = pressure_map > 0.02
        active_count = np.sum(active_mask)
        active_pressures = pressure_map[active_mask] if active_count > 0 else np.array([0.0], dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Regional Pressure Distribution (8 Anatomical Zones) = 24 features
        # -------------------------------------------------------------
        # Zones along cranio-caudal axis (64 rows):
        # 0: Head/Neck (0-11)
        # 1: Upper Thorax/Scapula (12-21)
        # 2: Mid/Lower Thorax (22-31)
        # 3: Pelvis/Sacrum (32-43)
        # 4: Upper Thighs (44-51)
        # 5: Lower Thighs/Knees (52-57)
        # 6: Calves (58-61)
        # 7: Heels/Feet (62-63)
        zones = [
            (0, 12), (12, 22), (22, 32), (32, 44),
            (44, 52), (52, 58), (58, 62), (62, 64)
        ]
        for r_start, r_end in zones:
            sub = pressure_map[r_start:r_end, :]
            features.append(float(np.sum(sub) / total_load))     # Zone sum fraction
            features.append(float(np.mean(sub)))                 # Zone mean
            features.append(float(np.max(sub)))                  # Zone peak

        # -------------------------------------------------------------
        # 2. Global Spatial & Inertial Moments = 9 features
        # -------------------------------------------------------------
        # Active contact area ratio
        contact_area_ratio = active_count / float(H * W)
        features.append(float(contact_area_ratio))

        # Center of Pressure (CoP)
        ys, xs = np.indices((H, W))
        cop_x = np.sum(xs * pressure_map) / total_load
        cop_y = np.sum(ys * pressure_map) / total_load
        features.append(float(cop_x / W))
        features.append(float(cop_y / H))

        # Spatial second moments (inertia tensor)
        var_x = np.sum(((xs - cop_x) ** 2) * pressure_map) / total_load
        var_y = np.sum(((ys - cop_y) ** 2) * pressure_map) / total_load
        cov_xy = np.sum((xs - cop_x) * (ys - cop_y) * pressure_map) / total_load
        features.append(float(var_x / (W * W)))
        features.append(float(var_y / (H * H)))
        features.append(float(cov_xy / (W * H)))

        # Orientation angle & Eigenvalues
        diff = var_x - var_y
        orientation = 0.5 * np.arctan2(2.0 * cov_xy, diff + 1e-6)
        features.append(float(orientation))

        # Principal axes eigenvalues
        disc = np.sqrt(diff * diff + 4.0 * cov_xy * cov_xy)
        lambda1 = 0.5 * (var_x + var_y + disc) + 1e-6
        lambda2 = 0.5 * (var_x + var_y - disc) + 1e-6
        aspect_ratio = float(lambda1 / lambda2) if lambda2 > 1e-6 else 1.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (lambda2 / lambda1))))
        features.append(float(np.clip(aspect_ratio, 0.0, 50.0)))
        features.append(float(eccentricity))

        # -------------------------------------------------------------
        # 3. Spatial Gradient & Structural Texture Statistics = 10 features
        # -------------------------------------------------------------
        gx = sobel(pressure_map, axis=1)
        gy = sobel(pressure_map, axis=0)
        g_mag = np.sqrt(gx * gx + gy * gy)
        
        features.append(float(np.mean(g_mag)))
        features.append(float(np.std(g_mag)))
        features.append(float(skew(g_mag.flatten())))
        features.append(float(kurtosis(g_mag.flatten())))
        features.append(float(np.max(g_mag)))
        features.append(float(np.mean(gx)))
        features.append(float(np.mean(gy)))
        
        # Gradient direction entropy
        g_dir = np.arctan2(gy, gx + 1e-6)
        dir_hist, _ = np.histogram(g_dir, bins=8, density=True)
        dir_hist = dir_hist[dir_hist > 1e-6]
        dir_entropy = -np.sum(dir_hist * np.log2(dir_hist)) if len(dir_hist) > 0 else 0.0
        features.append(float(dir_entropy))

        # Laplacian energy
        lap = laplace(pressure_map)
        features.append(float(np.mean(lap * lap)))
        features.append(float(np.mean(g_mag > 0.15)))

        # -------------------------------------------------------------
        # 4. Information-Theoretic & Distributional Statistics = 6 features
        # -------------------------------------------------------------
        p_dist = pressure_map / total_load
        p_nonzero = p_dist[p_dist > 1e-7]
        shannon_entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        features.append(float(shannon_entropy))

        # Rényi collision entropy (alpha = 2)
        renyi_entropy = -np.log2(np.sum(p_nonzero ** 2) + 1e-8)
        features.append(float(renyi_entropy))

        # Gini Coefficient
        sorted_p = np.sort(pressure_map.flatten())
        n = len(sorted_p)
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * sorted_p)) / (n * np.sum(sorted_p) + 1e-6) - (n + 1.0) / n
        features.append(float(gini))

        # Skewness, Kurtosis, CV of active pressure
        features.append(float(skew(active_pressures)))
        features.append(float(kurtosis(active_pressures)))
        cv = float(np.std(active_pressures) / (np.mean(active_pressures) + 1e-6))
        features.append(cv)

        # -------------------------------------------------------------
        # 5. Histogram Quantiles & Sparsity Indices = 15 features
        # -------------------------------------------------------------
        quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        q_vals = np.percentile(pressure_map, quantiles)
        for qv in q_vals:
            features.append(float(qv))

        # Dynamic range
        features.append(float(q_vals[-1] - q_vals[0]))
        
        # Hoyer Sparsity Index
        n_elements = H * W
        l1_norm = np.sum(np.abs(pressure_map))
        l2_norm = np.sqrt(np.sum(pressure_map * pressure_map)) + 1e-6
        sqrt_n = np.sqrt(n_elements)
        hoyer = (sqrt_n - (l1_norm / l2_norm)) / (sqrt_n - 1.0)
        features.append(float(hoyer))

        # Interquartile range (75% - 25%)
        iqr = float(np.percentile(pressure_map, 75) - np.percentile(pressure_map, 25))
        features.append(iqr)

        # Mean to Peak Ratio
        mean_to_peak = float(np.mean(pressure_map) / (np.max(pressure_map) + 1e-6))
        features.append(mean_to_peak)

        # -------------------------------------------------------------
        # 6. Bilateral Symmetry Indices (Left vs Right) = 8 features
        # -------------------------------------------------------------
        mid_c = W // 2
        left_half = pressure_map[:, :mid_c]
        right_half = pressure_map[:, mid_c:]
        
        load_left = np.sum(left_half)
        load_right = np.sum(right_half)
        overall_imbalance = (load_left - load_right) / (load_left + load_right + 1e-6)
        features.append(float(overall_imbalance))

        # Zone-specific bilateral asymmetries (Head, Thorax, Pelvis, Legs)
        macro_zones = [(0, 12), (12, 32), (32, 48), (48, 64)]
        for z_start, z_end in macro_zones:
            lz = np.sum(left_half[z_start:z_end, :])
            rz = np.sum(right_half[z_start:z_end, :])
            features.append(float((lz - rz) / (lz + rz + 1e-6)))

        # Vertical CoP offset between left and right sides
        ys_half = ys[:, :mid_c]
        cop_y_left = np.sum(ys_half * left_half) / (load_left + 1e-6)
        cop_y_right = np.sum(ys[:, mid_c:] * right_half) / (load_right + 1e-6)
        features.append(float(abs(cop_y_left - cop_y_right) / H))

        # Pearson correlation between left half and flipped right half
        right_flipped = np.fliplr(right_half)
        lf = left_half.flatten()
        rf = right_flipped.flatten()
        corr_mat = np.corrcoef(lf, rf)
        corr_val = corr_mat[0, 1] if not np.isnan(corr_mat[0, 1]) else 0.0
        features.append(float(corr_val))

        # Bilateral L2 difference
        l2_diff = np.linalg.norm(left_half - right_flipped) / (np.linalg.norm(left_half) + 1e-6)
        features.append(float(l2_diff))

        # -------------------------------------------------------------
        # 7. Multi-Scale Spatial Frequency Energy Metrics = 22 features
        # -------------------------------------------------------------
        # 2D FFT magnitude spectrum
        fft_mag = np.abs(np.fft.fft2(pressure_map))
        fft_shift = np.fft.fftshift(fft_mag)
        total_spectral_energy = np.sum(fft_shift * fft_shift) + 1e-6

        # DC / Low frequency core
        ch, cw = H // 2, W // 2
        dc_energy = np.sum(fft_shift[ch - 2:ch + 3, cw - 2:cw + 3] ** 2) / total_spectral_energy
        features.append(float(dc_energy))

        # 4 Horizontal subbands (low to high u frequency)
        for i in range(4):
            band_energy = np.sum(fft_shift[:, cw + i * 4: cw + (i + 1) * 4] ** 2) / total_spectral_energy
            features.append(float(band_energy))

        # 4 Vertical subbands (low to high v frequency)
        for i in range(4):
            band_energy = np.sum(fft_shift[ch + i * 7: ch + (i + 1) * 7, :] ** 2) / total_spectral_energy
            features.append(float(band_energy))

        # 4 Diagonal quadrant subbands
        for i in range(4):
            q_energy = np.sum(fft_shift[ch + i * 5: ch + (i + 1) * 5, cw + i * 3: cw + (i + 1) * 3] ** 2) / total_spectral_energy
            features.append(float(q_energy))

        # High frequency residual power
        hf_mask = np.ones((H, W), dtype=bool)
        hf_mask[ch - 6:ch + 7, cw - 4:cw + 5] = False
        hf_power = np.sum(fft_shift[hf_mask] ** 2) / total_spectral_energy
        features.append(float(hf_power))

        # Spectral Centroid (u and v)
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        spec_sum = np.sum(fft_shift) + 1e-6
        sc_u = np.sum(u_coords * fft_shift) / (spec_sum * W)
        sc_v = np.sum(v_coords * fft_shift) / (spec_sum * H)
        features.append(float(sc_u))
        features.append(float(sc_v))

        # Spectral Spread (variance in u and v)
        spread_u = np.sum(((u_coords / W - sc_u) ** 2) * fft_shift) / spec_sum
        spread_v = np.sum(((v_coords / H - sc_v) ** 2) * fft_shift) / spec_sum
        features.append(float(np.sqrt(spread_u)))
        features.append(float(np.sqrt(spread_v)))

        # Spectral roll-off (frequency covering 85% energy)
        radial_r = np.sqrt(((u_coords - cw) / cw) ** 2 + ((v_coords - ch) / ch) ** 2)
        sorted_rad_idx = np.argsort(radial_r.flatten())
        cum_energy = np.cumsum(fft_shift.flatten()[sorted_rad_idx] ** 2) / total_spectral_energy
        rolloff_idx = np.searchsorted(cum_energy, 0.85)
        rolloff_val = radial_r.flatten()[sorted_rad_idx[min(rolloff_idx, len(sorted_rad_idx) - 1)]]
        features.append(float(rolloff_val))

        # High-to-Low frequency energy ratio
        lf_energy = 1.0 - hf_power
        hl_ratio = float(hf_power / (lf_energy + 1e-6))
        features.append(hl_ratio)

        # Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
        flat_fft = fft_shift.flatten() + 1e-6
        geom_mean = np.exp(np.mean(np.log(flat_fft)))
        arith_mean = np.mean(flat_fft)
        spectral_flatness = float(geom_mean / arith_mean)
        features.append(spectral_flatness)

        # Spectral Crest Factor (Peak / RMS)
        peak_spec = np.max(fft_shift)
        rms_spec = np.sqrt(np.mean(fft_shift ** 2)) + 1e-6
        features.append(float(peak_spec / rms_spec))

        # Validation: exactly 94 features
        feature_vector = np.array(features, dtype=np.float32)
        assert len(feature_vector) == 94, f"Expected 94 features, but extracted {len(feature_vector)}"
        return feature_vector


if __name__ == "__main__":
    preprocessor = PressurePreprocessor()
    dummy_map = np.random.rand(64, 32).astype(np.float32)
    feats = preprocessor.extract_94_biomechanical_features(dummy_map)
    print(f"Features extracted successfully! Shape: {feats.shape}")
