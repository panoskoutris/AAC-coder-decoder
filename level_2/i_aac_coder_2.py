"""
AAC Decoder Level 2
Implements the inverse AAC encoding pipeline with inverse TNS.
"""

import numpy as np
import soundfile as sf
from i_filter_bank import i_filter_bank
from i_tns import i_tns


def i_aac_coder_2(aac_seq_2, filename_out):
    """
    AAC Decoder - Level 2
    
    Decodes an AAC encoded sequence (with TNS) back to audio:
    1. For each encoded frame:
       - Applies inverse TNS to restore original MDCT coefficients
       - Reconstructs stereo MDCT coefficients
       - Applies inverse filterbank (IMDCT) to get time-domain samples
    2. Overlaps and adds frames (50% overlap)
    3. Writes decoded audio to WAV file (stereo, 48kHz)
    4. Returns the decoded audio samples
    
    Parameters
    ----------
    aac_seq_2 : list of dict
        List of K encoded frames from aac_coder_2.
        Each dictionary contains:
        - "frame_type": str - Frame type ("OLS", "LSS", "ESH", "LPS")
        - "win_type": str - Window type ("SIN" or "KBD")
        - "chl": dict with:
          * "tns_coeffs": Quantized LPC coefficients for left channel
          * "frame_F": TNS-filtered MDCT coefficients for left channel
        - "chr": dict with:
          * "tns_coeffs": Quantized LPC coefficients for right channel
          * "frame_F": TNS-filtered MDCT coefficients for right channel
    filename_out : str
        Path to output WAV file (stereo, 48kHz)
    
    Returns
    -------
    x : numpy.ndarray
        Decoded audio samples, shape (total_samples, 2)
        Stereo audio normalized to [-1.0, 1.0]
    """
    
    # Get number of frames
    num_frames = len(aac_seq_2)
    
    # Frame parameters
    frame_size = 2048
    hop_size = 1024  # 50% overlap
    
    # Calculate total output length
    # We need space for all frames with overlap
    total_samples = (num_frames - 1) * hop_size + frame_size
    
    # Initialize output audio buffer
    audio_output = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Process each frame
    for i, frame_dict in enumerate(aac_seq_2):
        # Extract frame information
        frame_type = frame_dict["frame_type"]
        win_type = frame_dict["win_type"]
        
        # Extract TNS-filtered MDCT coefficients and TNS coefficients
        frame_F_left_tns = frame_dict["chl"]["frame_F"]
        tns_coeffs_left = frame_dict["chl"]["tns_coeffs"]
        
        frame_F_right_tns = frame_dict["chr"]["frame_F"]
        tns_coeffs_right = frame_dict["chr"]["tns_coeffs"]
        
        # Apply inverse TNS to restore original MDCT coefficients
        frame_F_left = i_tns(frame_F_left_tns, frame_type, tns_coeffs_left)
        frame_F_right = i_tns(frame_F_right_tns, frame_type, tns_coeffs_right)
        
        # Reconstruct frame_F with both channels for inverse filter bank
        if frame_type == "ESH":
            # For ESH: frame_F from TNS is (128, 8)
            # Need to transpose back to (8, 128) then combine to (8, 128, 2)
            frame_F_left_T = frame_F_left.T    # (128, 8) -> (8, 128)
            frame_F_right_T = frame_F_right.T  # (128, 8) -> (8, 128)
            frame_F = np.stack([frame_F_left_T, frame_F_right_T], axis=2)
        else:
            # For OLS/LSS/LPS: frame_F from TNS is (1024, 1)
            # Flatten to (1024,) then combine to (1024, 2)
            frame_F_left_flat = frame_F_left.flatten()
            frame_F_right_flat = frame_F_right.flatten()
            frame_F = np.stack([frame_F_left_flat, frame_F_right_flat], axis=1)
        
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
