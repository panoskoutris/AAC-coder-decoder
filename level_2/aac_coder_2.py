"""
AAC Encoder Level 2
Implements AAC encoding with Temporal Noise Shaping (TNS).
"""

import numpy as np
import soundfile as sf
from SCC import SSC
from filter_bank import filter_bank
from tns import tns


def aac_coder_2(filename_in):
    """
    AAC Encoder - Level 2
    
    Encodes a stereo audio file using the AAC encoding pipeline with TNS:
    1. Reads WAV file (stereo, 48kHz)
    2. Splits signal into frames of 2048 samples with 50% overlap
    3. For each frame:
       - Determines frame type using SSC (Sequence Segmentation Control)
       - Applies filterbank (MDCT) to get frequency coefficients
       - Applies TNS (Temporal Noise Shaping) to MDCT coefficients
    4. Returns a list of dictionaries with encoding results
    
    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    
    Returns
    -------
    aac_seq_2 : list of dict
        List of K encoded frames, where K is the number of frames.
        Each dictionary contains:
        - "frame_type": str - Frame type ("OLS", "LSS", "ESH", "LPS")
        - "win_type": str - Window type ("SIN" or "KBD")
        - "chl": dict with:
          * "tns_coeffs": Quantized LPC coefficients for left channel
            - shape (4, 1) for OLS/LSS/LPS
            - shape (4, 8) for ESH
          * "frame_F": TNS-filtered MDCT coefficients for left channel
            - shape (1024, 1) for OLS/LSS/LPS
            - shape (128, 8) for ESH
        - "chr": dict with:
          * "tns_coeffs": Quantized LPC coefficients for right channel
          * "frame_F": TNS-filtered MDCT coefficients for right channel
    """
    
    # Read WAV file using soundfile
    # soundfile returns data as float64 normalized to [-1.0, 1.0]
    audio_data, sample_rate = sf.read(filename_in, dtype='float32')
    
    # Ensure audio is stereo (2 channels)
    if audio_data.ndim == 1:
        raise ValueError("Audio must be stereo (2 channels)")
    
    # Verify sample rate
    if sample_rate != 48000:
        print(f"Warning: Expected 48kHz, got {sample_rate}Hz")
    
    # Get total number of samples
    total_samples = audio_data.shape[0]
    
    # Frame parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Calculate number of frames
    # We need at least 2048 samples for the first frame
    # and we can create frames as long as we have enough samples
    num_frames = (total_samples - frame_size) // hop_size + 1
    
    # Add zero padding if needed for the last frame
    # We need to ensure we have enough samples for all frames + one lookahead frame
    required_samples = (num_frames + 1) * hop_size + frame_size
    if total_samples < required_samples:
        padding_needed = required_samples - total_samples
        audio_data = np.pad(audio_data, ((0, padding_needed), (0, 0)), mode='constant')
    
    # Initialize result list
    aac_seq_2 = []
    
    # Window type (using SIN window for Level 2)
    win_type = "SIN"
    
    # Previous frame type (start with "OLS")
    prev_frame_type = "OLS"
    
    # Process each frame
    for i in range(num_frames):
        # Extract current frame (frame i)
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        frame_T = audio_data[start_idx:end_idx, :]
        
        # Extract next frame (frame i+1) for SSC decision
        next_start_idx = (i + 1) * hop_size
        next_end_idx = next_start_idx + frame_size
        next_frame_T = audio_data[next_start_idx:next_end_idx, :]
        
        # Determine frame type using SSC
        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        
        # Apply filter bank (MDCT)
        frame_F = filter_bank(frame_T, frame_type, win_type)
        
        # Split into left and right channels and prepare for TNS
        if frame_type == "ESH":
            # For ESH: frame_F shape is (8, 128, 2)
            # Transpose to (128, 8) for each channel (required by TNS)
            frame_F_left = frame_F[:, :, 0].T   # shape: (128, 8)
            frame_F_right = frame_F[:, :, 1].T  # shape: (128, 8)
        else:
            # For OLS/LSS/LPS: frame_F shape is (1024, 2)
            # Reshape to (1024, 1) for each channel (required by TNS)
            frame_F_left = frame_F[:, 0].reshape(-1, 1)   # shape: (1024, 1)
            frame_F_right = frame_F[:, 1].reshape(-1, 1)  # shape: (1024, 1)
        
        # Apply TNS to left channel
        frame_F_left_tns, tns_coeffs_left = tns(frame_F_left, frame_type)
        
        # Apply TNS to right channel
        frame_F_right_tns, tns_coeffs_right = tns(frame_F_right, frame_type)
        
        # Create dictionary for this frame
        frame_dict = {
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {
                "tns_coeffs": tns_coeffs_left,
                "frame_F": frame_F_left_tns
            },
            "chr": {
                "tns_coeffs": tns_coeffs_right,
                "frame_F": frame_F_right_tns
            }
        }
        
        # Add to result list
        aac_seq_2.append(frame_dict)
        
        # Update previous frame type for next iteration
        prev_frame_type = frame_type
    
    return aac_seq_2
