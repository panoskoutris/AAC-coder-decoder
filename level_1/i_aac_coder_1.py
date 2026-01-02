"""
AAC Decoder Level 1
Implements the inverse AAC encoding pipeline.
"""

import numpy as np
import soundfile as sf
from i_filter_bank import i_filter_bank


def i_aac_coder_1(aac_seq_1, filename_out):
    """
    AAC Decoder - Level 1
    
    Decodes an AAC encoded sequence back to audio:
    1. For each encoded frame:
       - Reconstructs MDCT coefficients from channels
       - Applies inverse filterbank (IMDCT) to get time-domain samples
    2. Overlaps and adds frames (50% overlap)
    3. Writes decoded audio to WAV file (stereo, 48kHz)
    4. Returns the decoded audio samples
    
    Parameters
    ----------
    aac_seq_1 : list of dict
        List of K encoded frames from aac_coder_1.
        Each dictionary contains:
        - "frame_type": str - Frame type ("OLS", "LSS", "ESH", "LPS")
        - "win_type": str - Window type ("SIN" or "KBD")
        - "chl": dict with "frame_F" - MDCT coefficients for left channel
          * For OLS/LSS/LPS: shape (1024,)
          * For ESH: shape (8, 128)
        - "chr": dict with "frame_F" - MDCT coefficients for right channel
          * For OLS/LSS/LPS: shape (1024,)
          * For ESH: shape (8, 128)
    filename_out : str
        Path to output WAV file (stereo, 48kHz)
    
    Returns
    -------
    x : numpy.ndarray
        Decoded audio samples, shape (total_samples, 2)
        Stereo audio normalized to [-1.0, 1.0]
    """
    
    # Get number of frames
    num_frames = len(aac_seq_1)
    
    # Frame parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Calculate total output length
    # We need space for all frames with overlap
    total_samples = (num_frames - 1) * hop_size + frame_size
    
    # Initialize output audio buffer
    audio_output = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Process each frame
    for i, frame_dict in enumerate(aac_seq_1):
        # Extract frame information
        frame_type = frame_dict["frame_type"]
        win_type = frame_dict["win_type"]
        frame_F_left = frame_dict["chl"]["frame_F"]
        frame_F_right = frame_dict["chr"]["frame_F"]
        
        # Reconstruct frame_F with both channels
        if frame_type == "ESH":
            # For ESH: combine into shape (8, 128, 2)
            frame_F = np.stack([frame_F_left, frame_F_right], axis=2)
        else:
            # For OLS/LSS/LPS: combine into shape (1024, 2)
            frame_F = np.stack([frame_F_left, frame_F_right], axis=1)
        
        # Apply inverse filter bank (IMDCT)
        frame_T = i_filter_bank(frame_F, frame_type, win_type)
        
        # Overlap-add into output buffer
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        audio_output[start_idx:end_idx, :] += frame_T
    
    # Write to WAV file
    sample_rate = 48000
    sf.write(filename_out, audio_output, sample_rate)
    
    return audio_output
