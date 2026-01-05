"""
i_aac_coder_3.py - AAC Level 3 Decoder (Inverse Coder)

Implements the inverse AAC encoding process:
1. Huffman decoding
2. Inverse quantization
3. Inverse TNS
4. Inverse filter bank (IMDCT)
5. Overlap-add reconstruction
"""

import numpy as np
import scipy.io as sio
import soundfile as sf
from i_filter_bank import i_filter_bank
from i_tns import i_tns
from i_aac_quantizer import i_aac_quantizer
from huff_utils import load_LUT
from safe_huffman import safe_decode_huff


def i_aac_coder_3(aac_seq_3, filename_out):
    """
    AAC Level 3 decoder - inverse of aac_coder_3.
    
    Decodes AAC-encoded audio sequence back to PCM audio and saves to file.
    
    Parameters
    ----------
    aac_seq_3 : list of dict
        Encoded AAC sequence. Each frame contains:
        - "frame_type": Frame type ("OLS", "LSS", "ESH", "LPS")
        - "win_type": Window type ("KBD", "SIN")
        - "chl": Left channel data
            - "G": Global gain
            - "sfc": Huffman-encoded scale factors
            - "stream": Huffman-encoded MDCT coefficients
            - "codebook": Codebook indices used
            - "tns_coeffs": TNS coefficients
            - "SMR": Signal-to-Mask Ratio (not used in decoding)
        - "chr": Right channel data (same structure as chl)
    
    filename_out : str
        Output WAV filename. Audio will be 48 kHz stereo.
    
    Returns
    -------
    x : np.ndarray
        Decoded audio samples, shape (num_samples, 2)
    """
    
    # Load Huffman codebooks
    huff_LUT_list = load_LUT()
    
    # Initialize output buffer
    # We'll collect all decoded frames and then combine them
    decoded_frames = []
    
    # Process each frame
    for frame_data in aac_seq_3:
        frame_type = frame_data["frame_type"]
        win_type = frame_data["win_type"]
        
        # Decode left channel to MDCT coefficients
        frame_F_left = decode_channel_to_mdct(
            frame_data["chl"],
            frame_type,
            huff_LUT_list
        )
        
        # Decode right channel to MDCT coefficients
        frame_F_right = decode_channel_to_mdct(
            frame_data["chr"],
            frame_type,
            huff_LUT_list
        )
        
        # Combine channels before i_filter_bank (like in level_1 and level_2)
        if frame_type == "ESH":
            # For ESH: combine into shape (8, 128, 2)
            frame_F = np.stack([frame_F_left, frame_F_right], axis=2)
        else:
            # For OLS/LSS/LPS: combine into shape (1024, 2)
            frame_F = np.stack([frame_F_left, frame_F_right], axis=1)
        
        # Apply inverse filter bank (IMDCT)
        frame_T = i_filter_bank(frame_F, frame_type, win_type)
        
        decoded_frames.append(frame_T)
    
    # Overlap-add reconstruction
    # Each frame is 2048 samples with 50% overlap (hop = 1024)
    num_frames = len(decoded_frames)
    frame_size = 2048
    hop_size = 1024
    
    # Calculate total output length
    total_samples = frame_size + (num_frames - 1) * hop_size
    
    # Initialize output array
    x = np.zeros((total_samples, 2))
    
    # Overlap-add
    for i, frame_T in enumerate(decoded_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        x[start_idx:end_idx, :] += frame_T
    
    # Save to WAV file (48 kHz, 16-bit PCM)
    sf.write(filename_out, x, 48000, subtype='PCM_16')
    
    return x


def decode_channel_to_mdct(channel_data, frame_type, huff_LUT_list):
    """
    Decode a single channel to MDCT coefficients (before IMDCT).
    
    Parameters
    ----------
    channel_data : dict
        Encoded channel data with keys:
        - "G": Global gain
        - "sfc": Huffman-encoded scale factors
        - "stream": Huffman-encoded MDCT coefficients
        - "codebook": Codebook indices
        - "tns_coeffs": TNS coefficients
    
    frame_type : str
        Frame type ("OLS", "LSS", "ESH", "LPS")
    
    huff_LUT_list : list
        Huffman codebook lookup tables
    
    Returns
    -------
    frame_F : np.ndarray
        MDCT coefficients after inverse quantization and inverse TNS
        - For OLS/LSS/LPS: shape (1024,)
        - For ESH: shape (8, 128)
    """
    
    # Extract encoded data
    G = channel_data["G"]
    sfc_stream_raw = channel_data["sfc"]
    stream_raw = channel_data["stream"]
    codebook_raw = channel_data["codebook"]
    tns_coeffs = channel_data["tns_coeffs"]
    
    # Navigate nested MATLAB arrays to get strings
    if isinstance(sfc_stream_raw, np.ndarray):
        temp = sfc_stream_raw
        while isinstance(temp, np.ndarray) and temp.size > 0:
            temp = temp.item() if temp.size == 1 else temp[0]
        sfc_stream = str(temp)
    else:
        sfc_stream = str(sfc_stream_raw)
    
    if isinstance(stream_raw, np.ndarray):
        temp = stream_raw
        while isinstance(temp, np.ndarray) and temp.size > 0:
            temp = temp.item() if temp.size == 1 else temp[0]
        stream = str(temp)
    else:
        stream = str(stream_raw)
    
    # Extract codebook index
    if isinstance(codebook_raw, np.ndarray):
        codebook = int(codebook_raw.item() if codebook_raw.size == 1 else codebook_raw.flatten()[0])
    else:
        codebook = int(codebook_raw)
    
    # Determine expected lengths based on frame type
    if frame_type == "ESH":
        expected_S_len = 128 * 8  # 1024
        expected_sfc_len = 42 * 8  # 336
    else:
        expected_S_len = 1024
        expected_sfc_len = 69
    
    # Decode MDCT coefficients using safe Huffman decoder
    if codebook == 0:
        S_flat = np.zeros(expected_S_len, dtype=int)
    else:
        S_decoded = safe_decode_huff(stream, huff_LUT_list[codebook], max_symbols=expected_S_len)
        S_flat = np.array(S_decoded, dtype=int)
        # Ensure correct length
        if len(S_flat) > expected_S_len:
            S_flat = S_flat[:expected_S_len]
        elif len(S_flat) < expected_S_len:
            S_flat = np.pad(S_flat, (0, expected_S_len - len(S_flat)), mode='constant')
    
    # Decode scale factors using safe Huffman decoder (codebook 11)
    sfc_decoded = safe_decode_huff(sfc_stream, huff_LUT_list[11], max_symbols=expected_sfc_len)
    sfc_flat = np.array(sfc_decoded, dtype=int)
    # Ensure correct length
    if len(sfc_flat) > expected_sfc_len:
        sfc_flat = sfc_flat[:expected_sfc_len]
    elif len(sfc_flat) < expected_sfc_len:
        sfc_flat = np.pad(sfc_flat, (0, expected_sfc_len - len(sfc_flat)), mode='constant')
    
    # Reshape S and sfc based on frame type
    if frame_type == "ESH":
        # ESH: (1024,) -> (128, 8), (336,) -> (42, 8)
        S = S_flat.reshape((128, 8))
        sfc = sfc_flat.reshape((42, 8))
    else:
        # OLS/LSS/LPS: keep as 1D for consistency with quantizer
        S = S_flat
        sfc = sfc_flat
    
    # Inverse quantization
    frame_F_tns = i_aac_quantizer(S, sfc, G, frame_type)
    
    # Inverse TNS
    frame_F = i_tns(frame_F_tns, frame_type, tns_coeffs)
    
    # Return MDCT coefficients
    # For ESH: shape (128, 8) -> transpose to (8, 128) to match level_1/level_2
    if frame_type == "ESH":
        return frame_F.T  # (128, 8) -> (8, 128)
    else:
        # For long frames, flatten to (1024,)
        return frame_F.flatten()