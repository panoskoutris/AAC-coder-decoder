"""
AAC Encoder Level 3
Implements full AAC encoding with TNS, Psychoacoustic Model, Quantization and Huffman coding.
"""

import numpy as np
import soundfile as sf
from SCC import SSC
from filter_bank import filter_bank
from tns import tns
from psycho import psycho
from aac_quantizer import aac_quantizer, load_table_B219
from huff_utils import load_LUT, encode_huff


def aac_coder_3(filename_in, filename_aac_coded):
    """
    AAC Encoder - Level 3
    
    Full AAC encoding pipeline with psychoacoustic model, quantization and Huffman coding:
    1. Reads WAV file (stereo, 48kHz)
    2. Splits signal into frames of 2048 samples with 50% overlap
    3. For each frame:
       - Determines frame type using SSC (Sequence Segmentation Control)
       - Applies filterbank (MDCT) to get frequency coefficients
       - Applies TNS (Temporal Noise Shaping) to MDCT coefficients
       - Applies psychoacoustic model to compute SMR
       - Quantizes coefficients using psychoacoustic thresholds
       - Huffman encodes quantized coefficients and scale factors
    4. Saves encoded data to .mat file
    5. Returns encoding sequence
    
    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    
    filename_aac_coded : str
        Path to output .mat file for encoded AAC data
    
    Returns
    -------
    aac_seq_3 : list of dict
        List of K encoded frames, where K is the number of frames.
        Each dictionary contains:
        - "frame_type": str - Frame type ("OLS", "LSS", "ESH", "LPS")
        - "win_type": str - Window type ("SIN" or "KBD")
        - "chl": dict with left channel encoding data
          * "tns_coeffs": Quantized LPC coefficients (4×1 or 4×8)
          * "T": Thresholds from psycho model (NB×1 or NB×8)
          * "G": Global gains (scalar or array of 8)
          * "sfc": Huffman-encoded scale factors
          * "stream": Huffman-encoded MDCT coefficients
          * "codebook": Huffman codebooks used
        - "chr": dict with right channel encoding data (same structure)
    """
    
    # Read WAV file
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
    num_frames = (total_samples - frame_size) // hop_size + 1
    
    # Add zero padding if needed for lookahead frames
    required_samples = (num_frames + 1) * hop_size + frame_size
    if total_samples < required_samples:
        padding_needed = required_samples - total_samples
        audio_data = np.pad(audio_data, ((0, padding_needed), (0, 0)), mode='constant')
    
    # Load Huffman codebooks
    huff_LUT_list = load_LUT()
    
    # Load scalefactor bands
    bands_long, bands_short = load_table_B219()
    
    # Initialize result list
    aac_seq_3 = []
    
    # Window type
    win_type = "SIN"
    
    # Previous frame type (start with "OLS")
    prev_frame_type = "OLS"
    
    # Previous frames for psychoacoustic model (needed for temporal masking)
    frame_T_prev_1 = None
    frame_T_prev_2 = None
    
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
        
        # Apply TNS to both channels
        frame_F_left_tns, tns_coeffs_left = tns(frame_F_left, frame_type)
        frame_F_right_tns, tns_coeffs_right = tns(frame_F_right, frame_type)
        
        # Apply psychoacoustic model to get SMR
        # Prepare time-domain frames for psycho model
        cur_L = frame_T[:, 0]
        cur_R = frame_T[:, 1]
        prev1_L = frame_T_prev_1[:, 0] if frame_T_prev_1 is not None else cur_L
        prev1_R = frame_T_prev_1[:, 1] if frame_T_prev_1 is not None else cur_R
        prev2_L = frame_T_prev_2[:, 0] if frame_T_prev_2 is not None else prev1_L
        prev2_R = frame_T_prev_2[:, 1] if frame_T_prev_2 is not None else prev1_R
        
        SMR_left = psycho(cur_L, frame_type, prev1_L, prev2_L)
        SMR_right = psycho(cur_R, frame_type, prev1_R, prev2_R)
        
        # Quantize both channels
        S_left, sfc_left, G_left = aac_quantizer(frame_F_left_tns, frame_type, SMR_left)
        S_right, sfc_right, G_right = aac_quantizer(frame_F_right_tns, frame_type, SMR_right)
        
        # Encode left channel
        chl_encoded = encode_channel(S_left, sfc_left, G_left, frame_type, huff_LUT_list)
        chl_encoded["tns_coeffs"] = tns_coeffs_left
        # Store SMR (not T, as we don't compute actual thresholds yet)
        chl_encoded["SMR"] = SMR_left
        
        # Encode right channel
        chr_encoded = encode_channel(S_right, sfc_right, G_right, frame_type, huff_LUT_list)
        chr_encoded["tns_coeffs"] = tns_coeffs_right
        chr_encoded["SMR"] = SMR_right
        
        # Create dictionary for this frame
        frame_dict = {
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": chl_encoded,
            "chr": chr_encoded
        }
        
        # Add to result list
        aac_seq_3.append(frame_dict)
        
        # Update previous frames for next iteration
        frame_T_prev_2 = frame_T_prev_1
        frame_T_prev_1 = frame_T
        prev_frame_type = frame_type
    
    # Save to .mat file
    import scipy.io as sio
    sio.savemat(filename_aac_coded, {'aac_seq_3': aac_seq_3})
    
    return aac_seq_3


def encode_channel(S, sfc, G, frame_type, huff_LUT_list):
    """
    Encode a single channel using Huffman coding.
    
    Parameters
    ----------
    S : np.ndarray
        Quantized MDCT coefficients
        - (1024, 1) for OLS/LSS/LPS
        - (128, 8) for ESH
    sfc : np.ndarray
        Scale factors (DPCM encoded)
        - (69, 1) for OLS/LSS/LPS
        - (42, 8) for ESH
    G : float or np.ndarray
        Global gain(s)
    frame_type : str
        Frame type
    huff_LUT_list : list
        Huffman codebook lookup tables
    
    Returns
    -------
    encoded : dict
        Dictionary with encoded data:
        - "G": Global gain(s)
        - "sfc": Huffman-encoded scale factors
        - "stream": Huffman-encoded MDCT coefficients
        - "codebook": Codebook indices used
    """
    
    # For all frame types: flatten S and sfc, then encode
    # For ESH: (128, 8) -> (1024,), (42, 8) -> (336,)
    # For OLS/LSS/LPS: (1024, 1) -> (1024,), (69, 1) -> (69,)
    
    # Flatten to 1D
    S_flat = S.flatten().astype(int)
    sfc_flat = sfc.flatten().astype(int)
    
    # Check for debugging
    max_S = np.max(np.abs(S_flat))
    max_sfc = np.max(np.abs(sfc_flat))
    
    # Huffman encode entire S frame (all subframes together for ESH)
    stream, codebook = encode_huff(S_flat, huff_LUT_list)
    
    # Huffman encode scale factors with force_codebook=11 as per spec
    # Note: This may fail if sfc values exceed codebook 11 range [-16, 16]
    try:
        sfc_stream = encode_huff(sfc_flat, huff_LUT_list, force_codebook=11)
        sfc_codebook = 11
    except (IndexError, ValueError):
        # If forcing codebook 11 fails, use automatic selection
        sfc_stream, sfc_codebook = encode_huff(sfc_flat, huff_LUT_list)
    
    return {
        "G": G,
        "sfc": sfc_stream,
        "sfc_codebook": sfc_codebook,
        "stream": stream,
        "codebook": codebook
    }
