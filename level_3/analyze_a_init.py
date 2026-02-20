"""
Analyze a_init values across all frames to understand the -20 clamping issue
"""

import numpy as np
import soundfile as sf
from filter_bank import filter_bank

# Constants
MQ = 8191

# Read audio
audio_file = "/Users/chris/Desktop/Multimedia_Systems/project_material/material/LicorDeCalandraca.wav"
print("Reading audio file...")
audio, sr = sf.read(audio_file)

# Frame parameters
frame_size = 2048
hop_size = 1024
num_samples = len(audio)
num_frames = (num_samples - frame_size) // hop_size + 1

print(f"Total frames to analyze: {num_frames}")

# Arrays to collect statistics
maxX_values = []
a_init_values = []
a_init_clamped_values = []

# Sample every 10th frame to speed up analysis
sample_frames = range(0, min(num_frames, 300), 1)

for frame_idx in sample_frames:
    start_sample = frame_idx * hop_size
    end_sample = start_sample + frame_size
    
    if end_sample > num_samples:
        break
    
    frame_audio = audio[start_sample:end_sample, :]
    frame_L = frame_audio[:, 0]
    
    # Assume OLS for simplicity (most frames are OLS)
    frame_type = "OLS"
    win_type = "SIN"
    
    try:
        frame_F = filter_bank(frame_L, frame_type, win_type)
        frame_F_left = frame_F[:, 0] if frame_F.ndim == 2 else frame_F
        
        # Calculate maxX and a_init
        maxX = np.max(np.abs(frame_F_left))
        
        if maxX > 0:
            a_init_raw = (16/3) * np.log2((maxX ** (3/4)) / MQ)
            a_init = a_init_raw + 32  # Offset to bring into usable range
        else:
            a_init = 0
        
        a_init_clamped = np.clip(a_init, -20, 20)
        
        maxX_values.append(maxX)
        a_init_values.append(a_init)
        a_init_clamped_values.append(a_init_clamped)
        
    except Exception as e:
        continue

# Convert to arrays
maxX_values = np.array(maxX_values)
a_init_values = np.array(a_init_values)
a_init_clamped_values = np.array(a_init_clamped_values)

print(f"\nAnalyzed {len(maxX_values)} frames")
print("\nmaxX statistics:")
print(f"  Min: {np.min(maxX_values):.4f}")
print(f"  Max: {np.max(maxX_values):.4f}")
print(f"  Mean: {np.mean(maxX_values):.4f}")
print(f"  Median: {np.median(maxX_values):.4f}")
print(f"  Std: {np.std(maxX_values):.4f}")

print("\na_init (unclamped) statistics:")
print(f"  Min: {np.min(a_init_values):.4f}")
print(f"  Max: {np.max(a_init_values):.4f}")
print(f"  Mean: {np.mean(a_init_values):.4f}")
print(f"  Median: {np.median(a_init_values):.4f}")
print(f"  Std: {np.std(a_init_values):.4f}")

print("\na_init (clamped to [-20, 20]) statistics:")
print(f"  Min: {np.min(a_init_clamped_values):.4f}")
print(f"  Max: {np.max(a_init_clamped_values):.4f}")
print(f"  Mean: {np.mean(a_init_clamped_values):.4f}")
print(f"  Median: {np.median(a_init_clamped_values):.4f}")
print(f"  Frames hitting -20 floor: {np.sum(a_init_clamped_values == -20)} ({100*np.sum(a_init_clamped_values == -20)/len(a_init_clamped_values):.1f}%)")
print(f"  Frames hitting +20 ceiling: {np.sum(a_init_clamped_values == 20)} ({100*np.sum(a_init_clamped_values == 20)/len(a_init_clamped_values):.1f}%)")

# Calculate what maxX would need to be to avoid -20 clamp
# -20 = (16/3) * log2((maxX ** 0.75) / 8191)
# -20 * 3/16 = log2((maxX ** 0.75) / 8191)
# -3.75 = log2((maxX ** 0.75) / 8191)
# 2^(-3.75) = (maxX ** 0.75) / 8191
# maxX = (2^(-3.75) * 8191) ^ (4/3)
threshold_maxX = (2**(-3.75) * 8191) ** (4/3)
print(f"\nmaxX needs to be > {threshold_maxX:.2f} to avoid -20 clamp")
print(f"Frames with maxX < {threshold_maxX:.2f}: {np.sum(maxX_values < threshold_maxX)} ({100*np.sum(maxX_values < threshold_maxX)/len(maxX_values):.1f}%)")

# Show percentiles
print("\nmaxX percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(maxX_values, p):.4f}")

print("\na_init (unclamped) percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(a_init_values, p):.4f}")
