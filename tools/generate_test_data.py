"""
Synthetic Training Data Generator for Form Quality Classifier

Generates labeled pose feature vectors by sampling angle values around
biomechanically-informed thresholds with added noise to simulate real-world
measurement variation.

The thresholds are grounded in exercise science literature:
- Squat depth: 90° knee angle for parallel (Schoenfeld, 2010)
- Forward lean: 45° torso angle maximum (Fry et al., 2003)
- Knee cave: tracked via knee-ankle x-offset
- Push-up depth: 90° elbow angle (Cogley et al., 2005)
- Hip alignment: shoulder-hip-ankle collinearity
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path


# Feature names for the classifiers
SQUAT_FEATURES = [
    "avg_knee_angle",
    "left_knee_angle",
    "right_knee_angle",
    "torso_angle",
    "left_hip_angle",
    "right_hip_angle",
    "knee_angle_diff",        # |left - right| asymmetry
    "knee_ankle_x_offset",    # knee cave indicator
    "shoulder_hip_alignment", # lateral shift
]

PUSHUP_FEATURES = [
    "avg_elbow_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "hip_deviation",          # deviation from shoulder-ankle line
    "torso_angle",
    "left_hip_angle",
    "right_hip_angle",
    "shoulder_width_ratio",   # elbow flare indicator
]

# Labels
LABELS = {
    0: "good_form",
    1: "minor_issues",
    2: "major_issues",
}

# Error type labels for multi-label classification
SQUAT_ERROR_TYPES = [
    "insufficient_depth",
    "excessive_forward_lean",
    "knee_cave",
    "asymmetric_depth",
]

PUSHUP_ERROR_TYPES = [
    "insufficient_depth",
    "hip_sag",
    "hip_pike",
    "elbow_flare",
]


def generate_squat_sample(form_quality: int, rng: np.random.Generator) -> dict:
    """
    Generate a single squat pose feature sample.

    Args:
        form_quality: 0=good, 1=minor issues, 2=major issues
        rng: numpy random generator for reproducibility

    Returns:
        dict with feature values and error labels
    """
    noise_scale = 3.0  # degrees of noise

    if form_quality == 0:  # Good form
        avg_knee = rng.normal(85, noise_scale)  # Good depth
        torso = rng.normal(25, noise_scale)      # Upright torso
        knee_offset = rng.normal(0, 0.01)        # Aligned knees
        knee_diff = rng.exponential(3)           # Small asymmetry

    elif form_quality == 1:  # Minor issues (one error)
        error_type = rng.choice(["depth", "lean", "cave", "asymmetry"])

        if error_type == "depth":
            avg_knee = rng.normal(105, noise_scale)  # Slightly shallow
            torso = rng.normal(30, noise_scale)
            knee_offset = rng.normal(0, 0.01)
            knee_diff = rng.exponential(4)
        elif error_type == "lean":
            avg_knee = rng.normal(85, noise_scale)
            torso = rng.normal(50, noise_scale)  # Leaning forward
            knee_offset = rng.normal(0, 0.01)
            knee_diff = rng.exponential(3)
        elif error_type == "cave":
            avg_knee = rng.normal(85, noise_scale)
            torso = rng.normal(25, noise_scale)
            knee_offset = rng.normal(0.04, 0.01)  # Knees caving
            knee_diff = rng.exponential(4)
        else:  # asymmetry
            avg_knee = rng.normal(85, noise_scale)
            torso = rng.normal(25, noise_scale)
            knee_offset = rng.normal(0, 0.01)
            knee_diff = rng.normal(20, 3)  # Large asymmetry

    else:  # Major issues (multiple errors)
        avg_knee = rng.normal(120, noise_scale * 1.5)  # Very shallow
        torso = rng.normal(55, noise_scale)             # Bad lean
        knee_offset = rng.normal(0.05, 0.02)            # Cave
        knee_diff = rng.normal(18, 5)                   # Asymmetric

    # Derive individual knee angles from average + asymmetry
    left_knee = avg_knee + rng.normal(0, 2)
    right_knee = avg_knee + rng.normal(0, 2)
    knee_diff = abs(left_knee - right_knee)

    # Hip angles correlate with knee angles
    left_hip = rng.normal(90 + (avg_knee - 85) * 0.5, noise_scale)
    right_hip = rng.normal(90 + (avg_knee - 85) * 0.5, noise_scale)

    alignment = abs(rng.normal(0, 0.02))

    # Determine which errors are present
    errors = []
    if avg_knee > 100:
        errors.append("insufficient_depth")
    if torso > 45:
        errors.append("excessive_forward_lean")
    if abs(knee_offset) > 0.03:
        errors.append("knee_cave")
    if knee_diff > 15:
        errors.append("asymmetric_depth")

    return {
        "avg_knee_angle": float(np.clip(avg_knee, 30, 180)),
        "left_knee_angle": float(np.clip(left_knee, 30, 180)),
        "right_knee_angle": float(np.clip(right_knee, 30, 180)),
        "torso_angle": float(np.clip(torso, 0, 90)),
        "left_hip_angle": float(np.clip(left_hip, 30, 180)),
        "right_hip_angle": float(np.clip(right_hip, 30, 180)),
        "knee_angle_diff": float(np.clip(knee_diff, 0, 60)),
        "knee_ankle_x_offset": float(np.clip(knee_offset, -0.1, 0.1)),
        "shoulder_hip_alignment": float(np.clip(alignment, 0, 0.2)),
        "form_quality": form_quality,
        "errors": errors,
    }


def generate_pushup_sample(form_quality: int, rng: np.random.Generator) -> dict:
    """
    Generate a single push-up pose feature sample.

    Args:
        form_quality: 0=good, 1=minor issues, 2=major issues
        rng: numpy random generator for reproducibility
    """
    noise_scale = 3.0

    if form_quality == 0:  # Good form
        avg_elbow = rng.normal(85, noise_scale)    # Good depth
        hip_dev = rng.normal(0, 0.015)             # Aligned hips
        torso = rng.normal(5, noise_scale)         # Flat body
        shoulder_ratio = rng.normal(1.0, 0.05)     # Normal elbow width

    elif form_quality == 1:  # Minor issues
        error_type = rng.choice(["depth", "sag", "pike", "flare"])

        if error_type == "depth":
            avg_elbow = rng.normal(115, noise_scale)  # Shallow
            hip_dev = rng.normal(0, 0.015)
            torso = rng.normal(5, noise_scale)
            shoulder_ratio = rng.normal(1.0, 0.05)
        elif error_type == "sag":
            avg_elbow = rng.normal(85, noise_scale)
            hip_dev = rng.normal(0.07, 0.015)  # Hips sagging
            torso = rng.normal(5, noise_scale)
            shoulder_ratio = rng.normal(1.0, 0.05)
        elif error_type == "pike":
            avg_elbow = rng.normal(85, noise_scale)
            hip_dev = rng.normal(-0.07, 0.015)  # Hips piking
            torso = rng.normal(5, noise_scale)
            shoulder_ratio = rng.normal(1.0, 0.05)
        else:  # flare
            avg_elbow = rng.normal(85, noise_scale)
            hip_dev = rng.normal(0, 0.015)
            torso = rng.normal(5, noise_scale)
            shoulder_ratio = rng.normal(1.5, 0.1)  # Elbows too wide

    else:  # Major issues
        avg_elbow = rng.normal(125, noise_scale * 1.5)
        hip_dev = rng.normal(0.08, 0.03)
        torso = rng.normal(15, noise_scale)
        shoulder_ratio = rng.normal(1.4, 0.15)

    left_elbow = avg_elbow + rng.normal(0, 2)
    right_elbow = avg_elbow + rng.normal(0, 2)

    left_hip = rng.normal(170 - abs(hip_dev) * 100, noise_scale)
    right_hip = rng.normal(170 - abs(hip_dev) * 100, noise_scale)

    # Determine errors
    errors = []
    if avg_elbow > 110:
        errors.append("insufficient_depth")
    if hip_dev > 0.05:
        errors.append("hip_sag")
    if hip_dev < -0.05:
        errors.append("hip_pike")
    if shoulder_ratio > 1.3:
        errors.append("elbow_flare")

    return {
        "avg_elbow_angle": float(np.clip(avg_elbow, 30, 180)),
        "left_elbow_angle": float(np.clip(left_elbow, 30, 180)),
        "right_elbow_angle": float(np.clip(right_elbow, 30, 180)),
        "hip_deviation": float(np.clip(hip_dev, -0.2, 0.2)),
        "torso_angle": float(np.clip(torso, 0, 90)),
        "left_hip_angle": float(np.clip(left_hip, 90, 180)),
        "right_hip_angle": float(np.clip(right_hip, 90, 180)),
        "shoulder_width_ratio": float(np.clip(shoulder_ratio, 0.5, 2.5)),
        "form_quality": form_quality,
        "errors": errors,
    }


def generate_dataset(exercise: str, n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a complete labeled dataset for an exercise.

    Args:
        exercise: "squat" or "pushup"
        n_samples: total number of samples to generate
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Balanced class distribution with slight skew toward good form
    n_good = n_samples // 3 + n_samples // 10
    n_minor = n_samples // 3
    n_major = n_samples - n_good - n_minor

    generator = generate_squat_sample if exercise == "squat" else generate_pushup_sample

    samples = []
    for _ in range(n_good):
        samples.append(generator(0, rng))
    for _ in range(n_minor):
        samples.append(generator(1, rng))
    for _ in range(n_major):
        samples.append(generator(2, rng))

    # Shuffle
    rng.shuffle(samples)

    return pd.DataFrame(samples)


def main():
    """Generate and save training datasets."""
    output_dir = Path(__file__).parent.parent / "backend" / "models" / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for exercise in ["squat", "pushup"]:
        print(f"Generating {exercise} dataset...")
        df = generate_dataset(exercise, n_samples=3000)

        output_path = output_dir / f"{exercise}_training_data.csv"
        df.to_csv(output_path, index=False)

        print(f"  Saved {len(df)} samples to {output_path}")
        print(f"  Class distribution:")
        for quality in [0, 1, 2]:
            count = (df["form_quality"] == quality).sum()
            print(f"    {LABELS[quality]}: {count} ({count/len(df)*100:.1f}%)")

    print("\nDone! Training data saved.")


if __name__ == "__main__":
    main()
