"""
AAC Level 2 - TNS Impact Analysis
Compares time-domain error signals with/without TNS on a transient frame,
AFTER full encode–decode (as in Figure 6 of the assignment).
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys

# Allow imports from level_1 / level_2
sys.path.append('../level_1')
sys.path.append('../level_2')

from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1
from aac_coder_2 import aac_coder_2
from i_aac_coder_2 import i_aac_coder_2


def find_transient_frame(aac_seq):
    """
    Find a frame with strong transient (ESH or LSS).

    Returns
    -------
    idx : int
        Index of the first transient frame found.
    """
    for i, frame in enumerate(aac_seq):
        if frame["frame_type"] in ["ESH", "LSS"]:
            return i
    # If no transient found, return first frame
    return 0


def plot_tns_comparison(original_file):
    """
    Create plots comparing time-domain error signals
    with and without TNS for a single transient frame.
    
    Creates 3 plots like Figures 5 & 6 in the assignment:
    - Original signal in time domain (Figure 5)
    - Error without TNS (Figure 6a)
    - Error with TNS (Figure 6b)

    Parameters
    ----------
    original_file : str
        Path to original WAV file.
    """

    print("Reading original audio...")
    original_audio, sr = sf.read(original_file, dtype='float32')

    # ------------------------------------------------------------------
    # 1) Encode WITHOUT TNS (Level 1) and decode
    # ------------------------------------------------------------------
    print("\nEncoding without TNS (Level 1)...")
    aac_seq_1 = aac_coder_1(original_file)
    print("Decoding Level 1...")
    decoded_1 = i_aac_coder_1(aac_seq_1, "Licor_level1_decoded_tmp.wav")

    # ------------------------------------------------------------------
    # 2) Encode WITH TNS (Level 2) and decode
    # ------------------------------------------------------------------
    print("\nEncoding with TNS (Level 2)...")
    aac_seq_2 = aac_coder_2(original_file)
    print("Decoding Level 2...")
    decoded_2 = i_aac_coder_2(aac_seq_2, "Licor_level2_decoded_tmp.wav")

    # ------------------------------------------------------------------
    # 3) Find a transient frame (strong onset like castanets)
    # ------------------------------------------------------------------
    print("\nSearching for transient frame...")
    transient_idx = find_transient_frame(aac_seq_2)
    frame_type = aac_seq_2[transient_idx]["frame_type"]
    print(f"Found transient frame at index {transient_idx}")
    print(f"Frame type: {frame_type}")

    # Frame parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Extract the transient frame from all signals
    start_idx = transient_idx * hop_size
    end_idx = start_idx + frame_size
    
    # Left channel only
    original_frame = original_audio[start_idx:end_idx, 0]
    decoded_frame_1 = decoded_1[start_idx:end_idx, 0]
    decoded_frame_2 = decoded_2[start_idx:end_idx, 0]
    
    # Ensure same length
    min_len = min(len(original_frame), len(decoded_frame_1), len(decoded_frame_2))
    original_frame = original_frame[:min_len]
    decoded_frame_1 = decoded_frame_1[:min_len]
    decoded_frame_2 = decoded_frame_2[:min_len]

    # ------------------------------------------------------------------
    # 4) Compute error signals for this frame
    # ------------------------------------------------------------------
    error_without_tns = original_frame - decoded_frame_1
    error_with_tns = original_frame - decoded_frame_2

    # RMS errors
    rms_without_tns = np.sqrt(np.mean(error_without_tns ** 2))
    rms_with_tns = np.sqrt(np.mean(error_with_tns ** 2))

    # Sample indices (relative to frame start)
    samples = np.arange(min_len)
    
    # Time in milliseconds (for original signal plot)
    time_ms = samples / sr * 1000  # Convert to ms

    print(f"\nFrame details:")
    print(f"  Frame samples: {min_len}")
    print(f"  Frame duration: {min_len / sr * 1000:.2f} ms")
    print(f"  Sample indices: {start_idx} to {end_idx}")
    
    # Calculate common y-axis limits for error plots
    error_max = max(np.max(np.abs(error_without_tns)), np.max(np.abs(error_with_tns)))
    error_ylim = [-error_max * 1.1, error_max * 1.1]  # Add 10% margin

    # ------------------------------------------------------------------
    # 5) Plotting - 3 subplots like Figures 5 & 6
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(
        f'Transient Frame Analysis (Frame #{transient_idx}, Type: {frame_type})',
        fontsize=16, fontweight='bold'
    )

    # ========== Plot 1: Original Signal (Figure 5) ==========
    ax1 = axes[0]
    ax1.plot(time_ms, original_frame, color='black', linewidth=0.8)
    ax1.set_xlabel('Time [ms]', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Original Signal (Transient Frame)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Mark the transient peak
    peak_idx = np.argmax(np.abs(original_frame))
    peak_val = original_frame[peak_idx]
    ax1.plot(time_ms[peak_idx], peak_val, 'ro', markersize=8, label='Transient peak')
    ax1.legend(loc='upper right')

    # ========== Plot 2: Error WITHOUT TNS (Figure 6a) ==========
    ax2 = axes[1]
    ax2.plot(samples, error_without_tns, color='darkred', linewidth=0.8)
    ax2.set_xlabel('Time [samples]', fontsize=11)
    ax2.set_ylabel('Error Amplitude', fontsize=11)
    ax2.set_title('(a) Error Signal WITHOUT TNS (Level 1)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_ylim(error_ylim)  # Set common y-axis limits

    # RMS box
    ax2.text(0.02, 0.95, f'RMS Error: {rms_without_tns:.6f}',
             transform=ax2.transAxes, fontsize=10,
             va='top',
             bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))

    # Peak error
    peak_error_1 = np.max(np.abs(error_without_tns))
    ax2.text(0.98, 0.95, f'Peak: {peak_error_1:.6f}',
             transform=ax2.transAxes, fontsize=10,
             va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== Plot 3: Error WITH TNS (Figure 6b) ==========
    ax3 = axes[2]
    ax3.plot(samples, error_with_tns, color='darkgreen', linewidth=0.8)
    ax3.set_xlabel('Time [samples]', fontsize=11)
    ax3.set_ylabel('Error Amplitude', fontsize=11)
    ax3.set_title('(b) Error Signal WITH TNS (Level 2)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(error_ylim)  # Set common y-axis limits
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # RMS box
    ax3.text(0.02, 0.95, f'RMS Error: {rms_with_tns:.6f}',
             transform=ax3.transAxes, fontsize=10,
             va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Peak error
    peak_error_2 = np.max(np.abs(error_with_tns))
    ax3.text(0.98, 0.95, f'Peak: {peak_error_2:.6f}',
             transform=ax3.transAxes, fontsize=10,
             va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # RMS comparison
    rms_diff = (rms_with_tns - rms_without_tns) / rms_without_tns * 100.0
    comparison_text = f'TNS Impact: {rms_diff:+.2f}% RMS change'
    ax3.text(0.5, -0.15, comparison_text,
             transform=ax3.transAxes, fontsize=11,
             va='top', ha='center',
             weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    plt.tight_layout()

    # Save figure
    output_plot = 'level2_tns_comparison.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print(f"Plot saved as: {output_plot}")
    print("=" * 60)
    print("\nTransient Frame Analysis:")
    print(f"  Frame index:           {transient_idx}")
    print(f"  Frame type:            {frame_type}")
    print(f"  Frame samples:         {min_len}")
    print(f"  Frame duration:        {min_len / sr * 1000:.2f} ms")
    print(f"  RMS Error WITHOUT TNS: {rms_without_tns:.6f}")
    print(f"  RMS Error WITH TNS:    {rms_with_tns:.6f}")
    print(f"  TNS Impact:            {rms_diff:+.2f}%")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    # Path to original audio file
    material_path = "/Users/chris/Desktop/Multimedia_Systems/project_material/material"
    original_file = f"{material_path}/LicorDeCalandraca.wav"

    print("Generating TNS comparison plots (transient frame analysis)...")
    print("=" * 60)
    plot_tns_comparison(original_file)

