"""
Plot original vs decoded waveforms for Level 3
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Read original and decoded audio
original_file = "/Users/chris/Desktop/Multimedia_Systems/project_material/material/LicorDeCalandraca.wav"
decoded_file = "/Users/chris/Desktop/Multimedia_Systems/project_material/material/LicorDeCalandraca_decoded_level3.wav"

print("Reading audio files...")
original, sr = sf.read(original_file)
decoded, _ = sf.read(decoded_file)

# Truncate to same length
min_len = min(len(original), len(decoded))
original = original[:min_len]
decoded = decoded[:min_len]

# Use left channel
original_left = original[:, 0]
decoded_left = decoded[:, 0]

# Time axis in seconds
time = np.arange(len(original_left)) / sr

# Calculate error signal
error = original_left - decoded_left

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('AAC Level 3: Original vs Decoded Waveform', fontsize=16, fontweight='bold')

# Plot 1: Original vs Decoded overlaid
ax1.plot(time, original_left, color='blue', linewidth=0.6, alpha=0.7, label='Original')
ax1.plot(time, decoded_left, color='red', linewidth=0.6, alpha=0.7, label='Decoded (Level 3)')
ax1.set_xlabel('Time [s]', fontsize=11)
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title('Overlaid Waveforms', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, time[-1]])
ax1.legend(loc='upper right', fontsize=10)

# Plot 2: Error signal e[n] = x[n] - x_predicted[n]
ax2.plot(time, error, color='green', linewidth=0.5, alpha=0.8, label='Error e[n]')
ax2.set_xlabel('Time [s]', fontsize=11)
ax2.set_ylabel('Error Amplitude', fontsize=11)
ax2.set_title('Error Signal: e[n] = x[n] - x_decoded[n]', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, time[-1]])
ax2.legend(loc='upper right', fontsize=10)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()

# Save figure
output_file = 'level3_waveform_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Calculate SNR and error statistics
snr = 10 * np.log10(np.sum(original_left**2) / np.sum(error**2))
error_mse = np.mean(error**2)
error_rms = np.sqrt(error_mse)
error_max = np.max(np.abs(error))

print(f"\nSNR: {snr:.2f} dB")
print(f"Error MSE: {error_mse:.6f}")
print(f"Error RMS: {error_rms:.6f}")
print(f"Error Max: {error_max:.6f}")
print(f"Duration: {time[-1]:.2f} seconds")
print(f"Sample rate: {sr} Hz")

# Find the largest error spike
max_error_idx = np.argmax(np.abs(error))
max_error_time = time[max_error_idx]
max_error_value = error[max_error_idx]
print(f"\nLargest error spike:")
print(f"  Time: {max_error_time:.4f} seconds (sample {max_error_idx})")
print(f"  Value: {max_error_value:.6f}")
print(f"  Magnitude: {np.abs(max_error_value):.6f}")

# Frame analysis (assuming 2048 samples per frame, 50% overlap)
frame_size = 2048
hop_size = 1024
estimated_frame = max_error_idx // hop_size
print(f"  Estimated frame index: {estimated_frame}")

# Calculate error statistics in a window around the spike
window_size = 2048
start_idx = max(0, max_error_idx - window_size // 2)
end_idx = min(len(error), max_error_idx + window_size // 2)
window_error = error[start_idx:end_idx]
window_rms = np.sqrt(np.mean(window_error**2))
print(f"  RMS in 2048-sample window around spike: {window_rms:.6f}")

# Compare to typical error level (excluding the spike region)
mask = np.ones(len(error), dtype=bool)
mask[start_idx:end_idx] = False
typical_rms = np.sqrt(np.mean(error[mask]**2))
print(f"  Typical RMS (excluding spike region): {typical_rms:.6f}")
print(f"  Spike is {window_rms / typical_rms:.1f}x larger than typical error")

plt.show()
