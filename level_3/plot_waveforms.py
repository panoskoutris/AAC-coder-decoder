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

# Create figure with single plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('AAC Level 3: Original vs Decoded Waveform', fontsize=16, fontweight='bold')

# Plot both waveforms on same axes
ax.plot(time, original_left, color='blue', linewidth=0.6, alpha=0.7, label='Original')
ax.plot(time, decoded_left, color='red', linewidth=0.6, alpha=0.7, label='Decoded (Level 3)')
ax.set_xlabel('Time [s]', fontsize=11)
ax.set_ylabel('Amplitude', fontsize=11)
ax.set_title('Overlaid Waveforms', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, time[-1]])
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()

# Save figure
output_file = 'level3_waveform_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Calculate SNR
error = original_left - decoded_left
snr = 10 * np.log10(np.sum(original_left**2) / np.sum(error**2))
print(f"\nSNR: {snr:.2f} dB")
print(f"Duration: {time[-1]:.2f} seconds")
print(f"Sample rate: {sr} Hz")

plt.show()
