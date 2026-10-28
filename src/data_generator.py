"""
Synthetic Tactile Pressure Mat Data Generator for Posture Recognition and Landmark Estimation.

Simulates a 64x32 discrete tactile pressure sensor grid (cranio-caudal x lateral)
underneath a compliant viscoelastic mattress topper, capturing the biomechanical
load distributions of diverse human anthropometries across 6 canonical postures.

Physical Modeling Principles:
1. Anthropometric Scaling: Landmark positions scale with subject height, leg-to-torso ratio,
   biacromial diameter (shoulder width), and bicristal diameter (hip width).
2. Contact Mechanics: Pressure distribution modeled as multi-component Gaussian contact bells
   derived from anatomical bony prominences (sacrum, scapulae, trochanters, heels, occiput).
3. Viscoelastic Compliant Layer: Mattress attenuation simulated via 2D spatial convolution
   with an elastic Point Spread Function (PSF).
4. Sensor Noise & Degradation: Additive thermal Gaussian noise, 8-bit ADC quantization,
   and random dead/stuck sensor elements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SubjectAnthropometry:
    """Anthropometric parameters for a simulated human subject."""
    subject_id: int
    gender: str
    height_cm: float          # 150 to 195 cm
    weight_kg: float          # 50 to 105 kg
    shoulder_width_cm: float  # Biacromial width (34 to 48 cm)
    hip_width_cm: float       # Bicristal width (28 to 44 cm)
    torso_fraction: float     # Torso length as fraction of height (~0.40 - 0.46)
    leg_fraction: float       # Leg length as fraction of height (~0.46 - 0.53)
    soft_tissue_compliance: float # Soft tissue elasticity factor (spread factor)


POSTURE_NAMES = [
    "supine",
    "prone",
    "left_lateral",
    "right_lateral",
    "seated",
    "fetal"
]

POSTURE_TO_IDX = {name: idx for idx, name in enumerate(POSTURE_NAMES)}
IDX_TO_POSTURE = {idx: name for idx, name in enumerate(POSTURE_NAMES)}

LANDMARK_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

LANDMARK_TO_IDX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
IDX_TO_LANDMARK = {idx: name for idx, name in enumerate(LANDMARK_NAMES)}


def get_default_subjects() -> List[SubjectAnthropometry]:
    """
    Generate 10 anthropometrically diverse subjects covering various BMI categories,
    heights, and skeletal proportions to evaluate cross-person generalisation.
    """
    subjects_spec = [
        # (id, gender, height, weight, shoulder_w, hip_w, torso_frac, leg_frac, compliance)
        (0, "female", 158.0, 52.0, 36.0, 34.0, 0.42, 0.48, 1.15),  # Slender, petite female
        (1, "male",   182.0, 84.0, 46.0, 36.0, 0.44, 0.49, 1.00),  # Broad-shouldered male
        (2, "female", 168.0, 78.0, 39.0, 41.0, 0.43, 0.47, 1.30),  # Higher BMI female, wider hips
        (3, "male",   174.0, 68.0, 42.0, 34.0, 0.43, 0.48, 0.95),  # Medium athletic male
        (4, "male",   192.0, 95.0, 48.0, 38.0, 0.45, 0.51, 1.05),  # Tall, heavy male
        (5, "female", 162.0, 60.0, 38.0, 36.0, 0.41, 0.49, 1.10),  # Average female
        (6, "male",   165.0, 62.0, 40.0, 33.0, 0.42, 0.47, 0.90),  # Compact male
        (7, "female", 175.0, 64.0, 39.0, 35.0, 0.43, 0.52, 1.00),  # Tall slender female
        (8, "male",   178.0, 102.0, 45.0, 42.0, 0.45, 0.47, 1.35), # High BMI male
        (9, "female", 152.0, 48.0, 35.0, 32.0, 0.41, 0.47, 1.05),  # Petite short female
    ]
    return [
        SubjectAnthropometry(
            subject_id=s[0],
            gender=s[1],
            height_cm=s[2],
            weight_kg=s[3],
            shoulder_width_cm=s[4],
            hip_width_cm=s[5],
            torso_fraction=s[6],
            leg_fraction=s[7],
            soft_tissue_compliance=s[8]
        )
        for s in subjects_spec
    ]


class PressureDataGenerator:
    """
    Synthesizes tactile pressure sensor arrays simulating bed/mat sensor grids.
    
    Grid resolution: 64 rows (cranio-caudal: y in [0, 63]) x 32 columns (lateral: x in [0, 31]).
    Coordinate system: (x=0, y=0) is top-left (patient's right if supine).
    x is horizontal (0 to 31), y is vertical (0 to 63, top to bottom).
    """

    def __init__(
        self,
        grid_height: int = 64,
        grid_width: int = 32,
        psf_sigma: float = 1.8,
        psf_kernel_size: int = 9,
        sensor_noise_std: float = 0.02,
        quantization_bits: int = 8,
        dead_pixel_prob: float = 0.008,
        random_seed: Optional[int] = 42
    ):
        self.H = grid_height
        self.W = grid_width
        self.psf_sigma = psf_sigma
        self.psf_kernel_size = psf_kernel_size
        self.sensor_noise_std = sensor_noise_std
        self.quantization_bits = quantization_bits
        self.dead_pixel_prob = dead_pixel_prob
        self.rng = np.random.default_rng(random_seed)

        # Precompute coordinate grids for fast Gaussian evaluation
        ys = np.arange(self.H, dtype=np.float32)
        xs = np.arange(self.W, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

    def _compute_canonical_landmarks(
        self,
        subject: SubjectAnthropometry,
        posture: str,
        frame_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates 2D positions (x, y) and contact visibility/load weights for 9 body landmarks.
        
        Returns:
            landmarks: (9, 2) array of coordinates in sensor pixels [x, y]
            weights: (9,) array of pressure load weight factors (0 if occluded/elevated)
        """
        # Anthropometric scaling factors mapped to 64x32 grid
        # Full mat represents 200 cm length and 100 cm width
        scale_y = (self.H - 8) / 200.0  # margin of 4 px top and bottom
        scale_x = (self.W - 6) / 100.0  # margin of 3 px left and right
        
        center_x = self.W / 2.0
        
        # Breathing & micro-postural oscillation (realistic dynamic jitter)
        breath_phase = frame_idx * 0.15
        chest_jitter_y = 0.35 * np.sin(breath_phase)
        micro_jitter_x = self.rng.normal(0, 0.25)
        micro_jitter_y = self.rng.normal(0, 0.25)
        
        # Anatomical lengths in pixels
        head_top_y = 6.0 + micro_jitter_y
        torso_len_px = subject.height_cm * subject.torso_fraction * scale_y
        leg_len_px = subject.height_cm * subject.leg_fraction * scale_y
        sh_w_px = (subject.shoulder_width_cm / 2.0) * scale_x
        hip_w_px = (subject.hip_width_cm / 2.0) * scale_x

        landmarks = np.zeros((9, 2), dtype=np.float32)
        weights = np.ones(9, dtype=np.float32)

        cx = center_x + micro_jitter_x

        if posture == "supine":
            # Symmetrical lying on back
            landmarks[0] = [cx, head_top_y + 2.5]                            # Head
            landmarks[1] = [cx - sh_w_px, head_top_y + 8.0 + chest_jitter_y] # Left shoulder (anatomical)
            landmarks[2] = [cx + sh_w_px, head_top_y + 8.0 + chest_jitter_y] # Right shoulder
            hip_y = head_top_y + 8.0 + torso_len_px
            landmarks[3] = [cx - hip_w_px, hip_y]                           # Left hip
            landmarks[4] = [cx + hip_w_px, hip_y]                           # Right hip
            knee_y = hip_y + leg_len_px * 0.48
            landmarks[5] = [cx - hip_w_px * 0.9, knee_y]                    # Left knee
            landmarks[6] = [cx + hip_w_px * 0.9, knee_y]                    # Right knee
            ankle_y = hip_y + leg_len_px
            landmarks[7] = [cx - hip_w_px * 0.8, min(ankle_y, self.H - 3.0)]# Left ankle / heel
            landmarks[8] = [cx + hip_w_px * 0.8, min(ankle_y, self.H - 3.0)]# Right ankle / heel
            
            # Load distribution weights (sacrum and scapulae bear major weight)
            weights = np.array([0.45, 0.85, 0.85, 1.20, 1.20, 0.35, 0.35, 0.55, 0.55], dtype=np.float32)

        elif posture == "prone":
            # Symmetrical lying on stomach (head rotated slightly to one side)
            head_turn = 2.5 if (subject.subject_id % 2 == 0) else -2.5
            landmarks[0] = [cx + head_turn, head_top_y + 2.0]
            landmarks[1] = [cx - sh_w_px * 1.05, head_top_y + 7.5]
            landmarks[2] = [cx + sh_w_px * 1.05, head_top_y + 7.5]
            hip_y = head_top_y + 7.5 + torso_len_px
            landmarks[3] = [cx - hip_w_px, hip_y]
            landmarks[4] = [cx + hip_w_px, hip_y]
            knee_y = hip_y + leg_len_px * 0.48
            landmarks[5] = [cx - hip_w_px * 0.95, knee_y]
            landmarks[6] = [cx + hip_w_px * 0.95, knee_y]
            ankle_y = hip_y + leg_len_px
            landmarks[7] = [cx - hip_w_px * 0.9, min(ankle_y, self.H - 3.0)]
            landmarks[8] = [cx + hip_w_px * 0.9, min(ankle_y, self.H - 3.0)]
            
            # Prone: anterior pelvis, chest, knees bear load; ankles elevated / minimal
            weights = np.array([0.40, 0.75, 0.75, 1.05, 1.05, 0.60, 0.60, 0.25, 0.25], dtype=np.float32)

        elif posture == "left_lateral":
            # Lying on left side: left side landmarks on bed; right side landmarks stacked/occluded!
            # Patient's left is towards lower x on grid
            lat_x = center_x - 3.0 + micro_jitter_x
            landmarks[0] = [lat_x - 1.0, head_top_y + 3.0]
            landmarks[1] = [lat_x - 1.5, head_top_y + 8.5]                   # Left shoulder (direct contact)
            landmarks[2] = [lat_x + 3.5, head_top_y + 8.0]                   # Right shoulder (elevated / stack)
            hip_y = head_top_y + 8.5 + torso_len_px
            landmarks[3] = [lat_x - 1.0, hip_y]                              # Left hip (greater trochanter)
            landmarks[4] = [lat_x + 3.0, hip_y - 1.0]                        # Right hip (elevated)
            knee_y = hip_y + leg_len_px * 0.44
            landmarks[5] = [lat_x - 1.5, knee_y]                             # Left knee
            landmarks[6] = [lat_x + 1.0, knee_y + 1.5]                       # Right knee (resting on left)
            ankle_y = hip_y + leg_len_px * 0.92
            landmarks[7] = [lat_x - 1.5, min(ankle_y, self.H - 4.0)]         # Left ankle
            landmarks[8] = [lat_x + 1.0, min(ankle_y + 1.0, self.H - 3.0)]   # Right ankle
            
            # Heavy concentrated load on left trochanter & shoulder; right side minimal/zero
            weights = np.array([0.50, 1.45, 0.10, 1.70, 0.15, 0.70, 0.20, 0.55, 0.10], dtype=np.float32)

        elif posture == "right_lateral":
            # Lying on right side: right side landmarks contact bed; left side stacked/occluded!
            lat_x = center_x + 3.0 + micro_jitter_x
            landmarks[0] = [lat_x + 1.0, head_top_y + 3.0]
            landmarks[1] = [lat_x - 3.5, head_top_y + 8.0]                   # Left shoulder (elevated)
            landmarks[2] = [lat_x + 1.5, head_top_y + 8.5]                   # Right shoulder (direct contact)
            hip_y = head_top_y + 8.5 + torso_len_px
            landmarks[3] = [lat_x - 3.0, hip_y - 1.0]                        # Left hip (elevated)
            landmarks[4] = [lat_x + 1.0, hip_y]                              # Right hip (greater trochanter)
            knee_y = hip_y + leg_len_px * 0.44
            landmarks[5] = [lat_x - 1.0, knee_y + 1.5]                       # Left knee (resting on right)
            landmarks[6] = [lat_x + 1.5, knee_y]                             # Right knee
            ankle_y = hip_y + leg_len_px * 0.92
            landmarks[7] = [lat_x - 1.0, min(ankle_y + 1.0, self.H - 3.0)]   # Left ankle
            landmarks[8] = [lat_x + 1.5, min(ankle_y, self.H - 4.0)]         # Right ankle
            
            # Heavy concentrated load on right trochanter & shoulder
            weights = np.array([0.50, 0.10, 1.45, 0.15, 1.70, 0.20, 0.70, 0.10, 0.55], dtype=np.float32)

        elif posture == "seated":
            # Fowler's position / upright seated on bed:
            # Backrest inclined, hips concentrated with high peak pressure at ischial tuberosities,
            # knees bent upwards, heels grounded.
            seat_hip_y = 34.0 + micro_jitter_y
            landmarks[0] = [cx, 8.0]                                          # Head resting against backrest
            landmarks[1] = [cx - sh_w_px * 0.85, 16.0 + chest_jitter_y]      # Left shoulder
            landmarks[2] = [cx + sh_w_px * 0.85, 16.0 + chest_jitter_y]      # Right shoulder
            landmarks[3] = [cx - hip_w_px * 0.8, seat_hip_y]                 # Left ischial tuberosity
            landmarks[4] = [cx + hip_w_px * 0.8, seat_hip_y]                 # Right ischial tuberosity
            knee_y = seat_hip_y + leg_len_px * 0.35
            landmarks[5] = [cx - hip_w_px * 0.95, knee_y]                    # Left knee (flexed, less contact)
            landmarks[6] = [cx + hip_w_px * 0.95, knee_y]                    # Right knee (flexed)
            heel_y = seat_hip_y + leg_len_px * 0.65
            landmarks[7] = [cx - hip_w_px * 0.7, min(heel_y, self.H - 3.0)]  # Left heel
            landmarks[8] = [cx + hip_w_px * 0.7, min(heel_y, self.H - 3.0)]  # Right heel
            
            # Massive pressure at hips/sacrum (~60% body weight concentrated), moderate upper back
            weights = np.array([0.30, 0.50, 0.50, 2.10, 2.10, 0.20, 0.20, 0.60, 0.60], dtype=np.float32)

        elif posture == "fetal":
            # Fetal posture: curled on side, knees tucked towards chest, spine flexed
            side_x = center_x - 2.0 if (subject.subject_id % 2 == 0) else center_x + 2.0
            direction = -1.0 if (subject.subject_id % 2 == 0) else 1.0
            
            fetal_head_y = 10.0 + micro_jitter_y
            landmarks[0] = [side_x + direction * 2.0, fetal_head_y]
            landmarks[1] = [side_x, fetal_head_y + 7.0]
            landmarks[2] = [side_x + direction * 4.0, fetal_head_y + 6.0]
            fetal_hip_y = fetal_head_y + torso_len_px * 0.78  # spine curved
            landmarks[3] = [side_x, fetal_hip_y]
            landmarks[4] = [side_x + direction * 3.5, fetal_hip_y - 1.0]
            # Knees pulled up to torso height!
            fetal_knee_y = fetal_hip_y - torso_len_px * 0.20
            landmarks[5] = [side_x + direction * 6.5, fetal_knee_y]
            landmarks[6] = [side_x + direction * 8.0, fetal_knee_y + 1.5]
            # Ankles tucked near hips
            fetal_ankle_y = fetal_hip_y + 4.0
            landmarks[7] = [side_x + direction * 5.0, fetal_ankle_y]
            landmarks[8] = [side_x + direction * 6.5, fetal_ankle_y + 2.0]
            
            weights = np.array([0.45, 1.20, 0.15, 1.50, 0.20, 0.80, 0.25, 0.50, 0.15], dtype=np.float32)

        # Clip landmarks within grid boundaries
        landmarks[:, 0] = np.clip(landmarks[:, 0], 1.0, self.W - 2.0)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 1.0, self.H - 2.0)

        return landmarks, weights

    def _render_contact_pressure(
        self,
        landmarks: np.ndarray,
        weights: np.ndarray,
        subject: SubjectAnthropometry,
        posture: str
    ) -> np.ndarray:
        """
        Renders continuous 2D contact pressure map (in kPa / arbitrary pressure units)
        based on anatomical contact bells and subject weight.
        """
        contact_map = np.zeros((self.H, self.W), dtype=np.float32)
        base_intensity = (subject.weight_kg / 70.0) * 1.5
        compliance = subject.soft_tissue_compliance

        # Anatomical spread radii (sigma_x, sigma_y) for different body segments
        # e.g., hips/sacrum have larger spread than heels/head
        landmark_spreads = [
            (2.2 * compliance, 2.4 * compliance),  # Head
            (2.0 * compliance, 2.0 * compliance),  # L Shoulder
            (2.0 * compliance, 2.0 * compliance),  # R Shoulder
            (3.0 * compliance, 3.2 * compliance),  # L Hip
            (3.0 * compliance, 3.2 * compliance),  # R Hip
            (1.8 * compliance, 2.0 * compliance),  # L Knee
            (1.8 * compliance, 2.0 * compliance),  # R Knee
            (1.4 * compliance, 1.6 * compliance),  # L Ankle/Heel
            (1.4 * compliance, 1.6 * compliance),  # R Ankle/Heel
        ]

        for i, ((lx, ly), w) in enumerate(zip(landmarks, weights)):
            if w <= 0.05:
                continue
            sig_x, sig_y = landmark_spreads[i]
            
            # 2D Gaussian contact bell
            dx = (self.grid_x - lx) / sig_x
            dy = (self.grid_y - ly) / sig_y
            bell = np.exp(-0.5 * (dx * dx + dy * dy))
            contact_map += w * base_intensity * bell

        # Render connecting anatomical torso/limb load bridges
        # Torso spine bridge between shoulders and hips
        mid_sh_x = (landmarks[1, 0] + landmarks[2, 0]) / 2.0
        mid_sh_y = (landmarks[1, 1] + landmarks[2, 1]) / 2.0
        mid_hip_x = (landmarks[3, 0] + landmarks[4, 0]) / 2.0
        mid_hip_y = (landmarks[3, 1] + landmarks[4, 1]) / 2.0
        
        # Intermediate vertebral contact
        if posture in ["supine", "seated"]:
            spine_weights = [0.4, 0.45, 0.5]
            for step, sw in zip([0.3, 0.5, 0.7], spine_weights):
                sx = mid_sh_x + step * (mid_hip_x - mid_sh_x)
                sy = mid_sh_y + step * (mid_hip_y - mid_sh_y)
                dx = (self.grid_x - sx) / (2.2 * compliance)
                dy = (self.grid_y - sy) / (2.5 * compliance)
                contact_map += sw * base_intensity * np.exp(-0.5 * (dx * dx + dy * dy))

        return contact_map

    def _apply_compliant_layer_psf(self, contact_map: np.ndarray) -> np.ndarray:
        """
        Simulates viscoelastic foam mattress attenuation.
        Continuous contact pressure diffuses through the foam slab, modeled
        as convolution with a 2D Gaussian point spread function (PSF).
        """
        blurred = gaussian_filter(contact_map, sigma=self.psf_sigma, mode="nearest")
        return blurred

    def _apply_sensor_noise_and_degradation(
        self,
        blurred_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies:
        1. Thermal/readout Gaussian sensor noise
        2. ADC quantization (e.g. 8-bit resolution)
        3. Dead/stuck sensor elements (dropouts)
        
        Returns:
            degraded_map: (H, W) sensor readout
            dead_mask: (H, W) boolean mask where True indicates a dead/faulty pixel
        """
        # 1. Thermal Gaussian noise
        noise = self.rng.normal(0.0, self.sensor_noise_std, size=blurred_map.shape).astype(np.float32)
        noisy = np.maximum(blurred_map + noise, 0.0)

        # 2. ADC Quantization to 8-bit range [0, 255] then normalized back to [0, 1] or raw ADC units
        max_val = np.max(noisy)
        if max_val > 1e-6:
            quant_steps = (2 ** self.quantization_bits) - 1
            scaled = (noisy / max_val) * quant_steps
            quantized = np.round(scaled) / quant_steps * max_val
        else:
            quantized = noisy

        # 3. Dead pixel injection (simulating broken tactile sensor cells)
        dead_mask = self.rng.random(size=quantized.shape) < self.dead_pixel_prob
        degraded_map = quantized.copy()
        degraded_map[dead_mask] = 0.0

        return degraded_map, dead_mask

    def generate_single_frame(
        self,
        subject: SubjectAnthropometry,
        posture: str,
        frame_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Generates a single synthetic pressure sensor frame with all physical degradations.
        """
        landmarks, weights = self._compute_canonical_landmarks(subject, posture, frame_idx)
        clean_contact = self._render_contact_pressure(landmarks, weights, subject, posture)
        blurred_map = self._apply_compliant_layer_psf(clean_contact)
        raw_sensor_map, dead_mask = self._apply_sensor_noise_and_degradation(blurred_map)

        return {
            "raw_sensor_map": raw_sensor_map,
            "clean_contact_map": clean_contact,
            "landmarks": landmarks,
            "landmark_weights": weights,
            "dead_mask": dead_mask,
            "posture": posture,
            "posture_idx": POSTURE_TO_IDX[posture],
            "subject_id": subject.subject_id
        }

    def generate_dataset(
        self,
        subjects: Optional[List[SubjectAnthropometry]] = None,
        frames_per_posture: int = 60
    ) -> Dict[str, np.ndarray]:
        """
        Generates full multi-subject, multi-posture synthetic dataset.
        
        Returns:
            Dictionary containing:
                'pressure_maps': (N, 64, 32)
                'clean_maps': (N, 64, 32)
                'landmarks': (N, 9, 2)
                'postures': (N,) int labels
                'subject_ids': (N,) int labels
                'dead_masks': (N, 64, 32) bool
        """
        if subjects is None:
            subjects = get_default_subjects()

        all_raw = []
        all_clean = []
        all_landmarks = []
        all_postures = []
        all_subjects = []
        all_masks = []

        total_frames = len(subjects) * len(POSTURE_NAMES) * frames_per_posture
        print(f"Generating synthetic tactile dataset: {len(subjects)} subjects, "
              f"{len(POSTURE_NAMES)} postures, {frames_per_posture} frames each ({total_frames} total frames)...")

        for subject in subjects:
            for posture in POSTURE_NAMES:
                for f_idx in range(frames_per_posture):
                    frame = self.generate_single_frame(subject, posture, f_idx)
                    all_raw.append(frame["raw_sensor_map"])
                    all_clean.append(frame["clean_contact_map"])
                    all_landmarks.append(frame["landmarks"])
                    all_postures.append(frame["posture_idx"])
                    all_subjects.append(frame["subject_id"])
                    all_masks.append(frame["dead_mask"])

        dataset = {
            "pressure_maps": np.array(all_raw, dtype=np.float32),
            "clean_maps": np.array(all_clean, dtype=np.float32),
            "landmarks": np.array(all_landmarks, dtype=np.float32),
            "postures": np.array(all_postures, dtype=np.int64),
            "subject_ids": np.array(all_subjects, dtype=np.int64),
            "dead_masks": np.array(all_masks, dtype=bool)
        }
        print(f"Dataset generation complete. Pressure maps shape: {dataset['pressure_maps'].shape}, "
              f"Landmarks shape: {dataset['landmarks'].shape}")
        return dataset


if __name__ == "__main__":
    # Quick self-test of the data generator
    generator = PressureDataGenerator(random_seed=42)
    sample_subject = get_default_subjects()[0]
    frame = generator.generate_single_frame(sample_subject, "supine", 0)
    print("Self-test succeeded!")
    print(f"Raw sensor map min: {frame['raw_sensor_map'].min():.3f}, max: {frame['raw_sensor_map'].max():.3f}")
    print(f"Landmarks shape: {frame['landmarks'].shape}")
    print(f"Dead pixels count: {np.sum(frame['dead_mask'])}")
