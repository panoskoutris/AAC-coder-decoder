"""
Analyze the specific frame that causes the error spike
"""

import numpy as np
import soundfile as sf
from filter_bank import filter_bank
from psycho import psycho
from aac_quantizer import aac_quantizer
from i_aac_quantizer import i_aac_quantizer
from i_filter_bank import i_filter_bank

# Read audio
audio_file = "/Users/chris/Desktop/Multimedia_Systems/project_material/material/LicorDeCalandraca.wav"
print("Reading audio file...")
audio, sr = sf.read(audio_file)

# Extract frames (similar to demo_aac_3.py)
frame_size = 2048
hop_size = 1024

# Target frame based on spike analysis
target_frame_idx = 138

print(f"\nAnalyzing frame {target_frame_idx} (spike location)...")

# Get audio samples for this frame
start_sample = target_frame_idx * hop_size
end_sample = start_sample + frame_size

if end_sample > len(audio):
    print("Frame index out of bounds")
    exit(1)

frame_audio = audio[start_sample:end_sample, :]

# Determine frame type (simplified - would normally use SSC)
# For now, assume OLS
frame_type = "OLS"  
win_type = "SIN"  # Use SIN window

# Extract left channel
frame_L = frame_audio[:, 0]

print(f"Frame type: {frame_type}")
print(f"Frame samples: {start_sample} to {end_sample}")
print(f"Frame audio range: [{np.min(frame_L):.6f}, {np.max(frame_L):.6f}]")

# Apply filter bank
frame_F = filter_bank(frame_L, frame_type, win_type)
frame_F_left = frame_F[:, 0] if frame_F.ndim == 2 else frame_F

print(f"MDCT coefficients range: [{np.min(frame_F_left):.6f}, {np.max(frame_F_left):.6f}]")
print(f"MDCT coefficients shape: {frame_F_left.shape}")
print(f"MDCT coefficients > 1.0: {np.sum(np.abs(frame_F_left) > 1.0)}")
print(f"MDCT coefficients > 0.1: {np.sum(np.abs(frame_F_left) > 0.1)}")
print(f"MDCT coefficients > 0.01: {np.sum(np.abs(frame_F_left) > 0.01)}")
print(f"MDCT coefficients mean: {np.mean(np.abs(frame_F_left)):.6f}")
print(f"MDCT coefficients std: {np.std(frame_F_left):.6f}")

# Get psychoacoustic model (need previous frames - use zeros for simplicity)
prev1_L = np.zeros(frame_size)
prev2_L = np.zeros(frame_size)

SMR_left = psycho(frame_L, frame_type, prev1_L, prev2_L)
print(f"SMR range: [{np.min(SMR_left):.2f}, {np.max(SMR_left):.2f}]")
print(f"SMR shape: {SMR_left.shape}")

# Quantize
S, sfc, G = aac_quantizer(frame_F_left, frame_type, SMR_left)
print(f"\nQuantization results:")
print(f"  Global gain G: {G}")
print(f"  Scale factors range: [{np.min(sfc)}, {np.max(sfc)}]")
print(f"  Quantized values S range: [{np.min(S)}, {np.max(S)}]")
print(f"  Non-zero coefficients: {np.count_nonzero(S)}/{S.size}")

# Inverse quantize
frame_F_reconstructed = i_aac_quantizer(S, sfc, G, frame_type)
print(f"  Reconstructed MDCT range: [{np.min(frame_F_reconstructed):.6f}, {np.max(frame_F_reconstructed):.6f}]")

# Compute MDCT error
mdct_error = frame_F_left - frame_F_reconstructed.flatten()
mdct_error_rms = np.sqrt(np.mean(mdct_error**2))
mdct_error_max = np.max(np.abs(mdct_error))
print(f"  MDCT error RMS: {mdct_error_rms:.6f}")
print(f"  MDCT error Max: {mdct_error_max:.6f}")

# Inverse filter bank
if frame_F_reconstructed.shape == (1024, 1):
    frame_F_for_ifb = np.hstack([frame_F_reconstructed, frame_F_reconstructed])
elif frame_F_reconstructed.ndim == 1:
    frame_F_for_ifb = np.zeros((1024, 2))
    frame_F_for_ifb[:, 0] = frame_F_reconstructed
    frame_F_for_ifb[:, 1] = frame_F_reconstructed
else:
    frame_F_for_ifb = frame_F_reconstructed

frame_T_reconstructed = i_filter_bank(frame_F_for_ifb, frame_type, win_type)
reconstructed = frame_T_reconstructed[:, 0] if frame_T_reconstructed.ndim == 2 else frame_T_reconstructed

# Time domain error
time_error = frame_L - reconstructed[:len(frame_L)]
time_error_rms = np.sqrt(np.mean(time_error**2))
time_error_max = np.max(np.abs(time_error))
print(f"\nTime domain reconstruction:")
print(f"  Error RMS: {time_error_rms:.6f}")
print(f"  Error Max: {time_error_max:.6f}")

# Check if this is significantly worse than average
print(f"\nComparison to typical error (from plot_waveforms.py):")
print(f"  Typical RMS: 0.035391")
print(f"  This frame RMS: {time_error_rms:.6f}")
print(f"  Ratio: {time_error_rms / 0.035391:.2f}x")
