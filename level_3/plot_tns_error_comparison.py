"""
Plot error amplitude before and after TNS
Comparison of quantization error with and without TNS processing
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from filter_bank import filter_bank
from i_filter_bank import i_filter_bank
from SCC import SSC
from tns import tns
from i_tns import i_tns
from aac_quantizer import aac_quantizer
from i_aac_quantizer import i_aac_quantizer
from psycho import psycho

# Read audio file
audio_file = "/Users/chris/Desktop/Multimedia_Systems/project_material/material/LicorDeCalandraca.wav"
print("Reading audio file...")
audio_data, sr = sf.read(audio_file)

# Parameters
frame_size = 2048
hop_size = 1024

# Search for an ESH (transient) frame
print("Searching for transient (ESH) frame...")
num_frames = (len(audio_data) - frame_size) // hop_size
prev_frame_type = "OLS"
frame_idx = None
found_frame_type = None

for i in range(min(num_frames, 200)):  # Check first 200 frames
    start_idx = i * hop_size
    end_idx = start_idx + frame_size
    frame_T_temp = audio_data[start_idx:end_idx, :]
    
    next_start_idx = (i + 1) * hop_size
    next_end_idx = next_start_idx + frame_size
    if next_end_idx <= len(audio_data):
        next_frame_T_temp = audio_data[next_start_idx:next_end_idx, :]
    else:
        break
    
    frame_type_temp = SSC(frame_T_temp, next_frame_T_temp, prev_frame_type)
    
    if frame_type_temp == "ESH":
        frame_idx = i
        found_frame_type = "ESH"
        print(f"Found ESH frame at index {frame_idx}")
        break
    
    prev_frame_type = frame_type_temp

if frame_idx is None:
    print("No ESH frame found in first 200 frames, using frame 100 instead")
    frame_idx = 100
    found_frame_type = None

# Select the frame
start_idx = frame_idx * hop_size
end_idx = start_idx + frame_size
frame_T = audio_data[start_idx:end_idx, :]

# Get next frame for SSC
next_start_idx = (frame_idx + 1) * hop_size
next_end_idx = next_start_idx + frame_size
next_frame_T = audio_data[next_start_idx:next_end_idx, :]

# Get previous frames for psycho model
if frame_idx >= 1:
    prev1_start = (frame_idx - 1) * hop_size
    prev1_end = prev1_start + frame_size
    frame_T_prev_1 = audio_data[prev1_start:prev1_end, :]
else:
    frame_T_prev_1 = None

if frame_idx >= 2:
    prev2_start = (frame_idx - 2) * hop_size
    prev2_end = prev2_start + frame_size
    frame_T_prev_2 = audio_data[prev2_start:prev2_end, :]
else:
    frame_T_prev_2 = None

# Determine frame type - use the frame_type from search if ESH was found
if found_frame_type == "ESH":
    frame_type = "ESH"
else:
    prev_frame_type = "OLS"
    frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
    
win_type = "SIN"

print(f"Processing frame {frame_idx}, type: {frame_type}")

# Process left channel only
original_frame = frame_T[:, 0]

# Apply filter bank
frame_F = filter_bank(frame_T, frame_type, win_type)

# For ESH frames, frame_F has shape (8, 128, 2) - need to transpose to (128, 8)
# Extract left channel
if frame_type == "ESH":
    if frame_F.ndim == 3:  # (8, 128, 2)
        frame_F_left = frame_F[:, :, 0].T  # Transpose to (128, 8)
    elif frame_F.ndim == 2:
        if frame_F.shape[0] == 8:  # (8, 128)
            frame_F_left = frame_F.T  # Transpose to (128, 8)
        else:  # (128, 8) already correct
            frame_F_left = frame_F
    else:
        frame_F_left = frame_F
else:
    frame_F_left = frame_F[:, 0] if frame_F.ndim == 2 else frame_F

# Get psychoacoustic model
cur_L = frame_T[:, 0]
if frame_T_prev_1 is not None:
    prev1_L = frame_T_prev_1[:, 0]
else:
    prev1_L = np.zeros_like(cur_L)

if frame_T_prev_2 is not None:
    prev2_L = frame_T_prev_2[:, 0]
else:
    prev2_L = np.zeros_like(cur_L)

SMR_left = psycho(cur_L, frame_type, prev1_L, prev2_L)

# ===== PATH 1: WITHOUT TNS =====
print("Processing without TNS...")
# Quantize directly without TNS
S_no_tns, sfc_no_tns, G_no_tns = aac_quantizer(frame_F_left, frame_type, SMR_left)
print(f"No-TNS: G={G_no_tns}, sfc range=[{np.min(sfc_no_tns)}, {np.max(sfc_no_tns)}], S non-zero={np.count_nonzero(S_no_tns)}/{S_no_tns.size}")
print(f"No-TNS sfc first 5 bands subframe 0: {sfc_no_tns[:5, 0]}")
print(f"No-TNS S stats: min={np.min(S_no_tns)}, max={np.max(S_no_tns)}, mean={np.mean(np.abs(S_no_tns)):.2f}")

# Inverse quantize
frame_F_reconstructed_no_tns = i_aac_quantizer(S_no_tns, sfc_no_tns, G_no_tns, frame_type)

# Compute MDCT quantization error (in frequency domain)
mdct_error_no_tns = frame_F_left - frame_F_reconstructed_no_tns

# ===== PATH 2: WITH TNS =====
print("Processing with TNS...")
# Apply TNS
frame_F_tns, tns_coeffs = tns(frame_F_left, frame_type)
print(f"TNS coefficients: {tns_coeffs}")
print(f"Frame changed by TNS: {not np.allclose(frame_F_tns, frame_F_left)}")
if not np.allclose(frame_F_tns, frame_F_left):
    diff = np.abs(frame_F_tns - frame_F_left)
    print(f"TNS modified frame: max diff = {np.max(diff):.6f}, mean diff = {np.mean(diff):.6f}")

# Quantize with TNS
S_with_tns, sfc_with_tns, G_with_tns = aac_quantizer(frame_F_tns, frame_type, SMR_left)
print(f"With-TNS: G={G_with_tns}, sfc range=[{np.min(sfc_with_tns)}, {np.max(sfc_with_tns)}], S non-zero={np.count_nonzero(S_with_tns)}/{S_with_tns.size}")

# Inverse quantize
frame_F_reconstructed_tns = i_aac_quantizer(S_with_tns, sfc_with_tns, G_with_tns, frame_type)

# Compute MDCT quantization error in TNS-filtered domain
mdct_error_with_tns = frame_F_tns - frame_F_reconstructed_tns

# Inverse TNS
frame_F_reconstructed_after_itns = i_tns(frame_F_reconstructed_tns, frame_type, tns_coeffs)
print(f"After inverse TNS vs without TNS: {np.allclose(frame_F_reconstructed_after_itns, frame_F_reconstructed_no_tns)}")
if not np.allclose(frame_F_reconstructed_after_itns, frame_F_reconstructed_no_tns):
    diff = np.abs(frame_F_reconstructed_after_itns - frame_F_reconstructed_no_tns)
    print(f"Inverse TNS differs from no-TNS: max diff = {np.max(diff):.6f}, mean diff = {np.mean(diff):.6f}")

# ===== PLOTTING: MDCT QUANTIZATION ERROR =====
# For ESH frames: shape is (128, 8) - flatten or pick a subframe
if frame_type == "ESH":
    # Flatten all 8 subframes into a single array for visualization
    mdct_error_no_tns_flat = mdct_error_no_tns.T.flatten()  # (8, 128) -> (1024,)
    mdct_error_with_tns_flat = mdct_error_with_tns.T.flatten()  # (8, 128) -> (1024,)
else:
    # For OLS frames, squeeze to 1D
    mdct_error_no_tns_flat = mdct_error_no_tns.flatten()
    mdct_error_with_tns_flat = mdct_error_with_tns.flatten()

# Coefficient axis
coeffs = np.arange(len(mdct_error_no_tns_flat))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'MDCT Quantization Error Comparison (Frame {frame_idx}, Type: {frame_type})', fontsize=14, fontweight='bold')

# Plot (a): MDCT Error without TNS
ax1.plot(coeffs, mdct_error_no_tns_flat, color='black', linewidth=0.8)
ax1.set_xlabel('MDCT Coefficient Index', fontsize=11)
ax1.set_ylabel('Quantization Error', fontsize=11)
ax1.set_title('(a) Quantization Error without TNS', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim([0, len(coeffs)])
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot (b): MDCT Error with TNS
ax2.plot(coeffs, mdct_error_with_tns_flat, color='black', linewidth=0.8)
ax2.set_xlabel('MDCT Coefficient Index', fontsize=11)
ax2.set_ylabel('Quantization Error', fontsize=11)
ax2.set_title('(b) Quantization Error with TNS (TNS-filtered domain)', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim([0, len(coeffs)])
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# Save figure
output_file = 'tns_error_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Print statistics
print(f"\nMDCT Quantization Error statistics:")
print(f"Without TNS - RMS: {np.sqrt(np.mean(mdct_error_no_tns_flat**2)):.6f}, Max: {np.max(np.abs(mdct_error_no_tns_flat)):.6f}")
print(f"With TNS    - RMS: {np.sqrt(np.mean(mdct_error_with_tns_flat**2)):.6f}, Max: {np.max(np.abs(mdct_error_with_tns_flat)):.6f}")

plt.show()
