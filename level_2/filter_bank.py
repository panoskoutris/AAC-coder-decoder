"""
Filter Bank Implementation for AAC Encoder (Level 1)
Implements MDCT (Modified Discrete Cosine Transform) analysis filter bank.
"""

import numpy as np


def filter_bank(frame_T, frame_type, win_type):
    """
    Implements the Filterbank stage (MDCT analysis).
    
    Parameters
    ----------
    frame_T : numpy.ndarray
        Frame in the time domain, shape (2048, 2) - 2048 samples × 2 channels
    frame_type : str
        Frame type: "OLS", "ESH", "LSS", "LPS"
        - "OLS": ONLY_LONG_SEQUENCE
        - "ESH": EIGHT_SHORT_SEQUENCE
        - "LSS": LONG_START_SEQUENCE
        - "LPS": LONG_STOP_SEQUENCE
    win_type : str
        Window type: "KBD" or "SIN"
    
    Returns
    -------
    frame_F : numpy.ndarray
        Frame in the frequency domain (MDCT coefficients)
        - For OLS/LSS/LPS: shape (1024, 2) - 1024 coefficients × 2 channels
        - For ESH: shape (8, 128, 2) - 8 subframes × 128 coefficients × 2 channels
    """
    
    if frame_type == "OLS":
        return _process_only_long(frame_T, win_type)
    elif frame_type == "ESH":
        return _process_eight_short(frame_T, win_type)
    elif frame_type == "LSS":
        return _process_long_start(frame_T, win_type)
    elif frame_type == "LPS":
        return _process_long_stop(frame_T, win_type)
    else:
        raise ValueError(f"Unknown frame_type: {frame_type}")


def _create_sin_window(N):
    """
    Creates a symmetric SIN window of length N.
    
    W_SIN(n) = sin(π/N * (n + 1/2))
    
    Parameters
    ----------
    N : int
        Window length (2048 for long, 256 for short)
    
    Returns
    -------
    window : numpy.ndarray
        SIN window of length N
    """
    n = np.arange(N)
    window = np.sin(np.pi / N * (n + 0.5))
    return window


def _create_sin_window_left(N):
    """
    Creates the left half of the SIN window.
    
    W_SIN_LEFT(n) = sin(π/N * (n + 1/2)), for n ∈ [0, N/2)
    
    Parameters
    ----------
    N : int
        Length of the full window
    
    Returns
    -------
    window : numpy.ndarray
        Left half of the SIN window (length N/2)
    """
    n = np.arange(N // 2)
    window = np.sin(np.pi / N * (n + 0.5))
    return window


def _create_sin_window_right(N):
    """
    Creates the right half of the SIN window.
    
    W_SIN_RIGHT(n) = sin(π/N * (n + 1/2)), for n ∈ [N/2, N)
    
    Parameters
    ----------
    N : int
        Length of the full window
    
    Returns
    -------
    window : numpy.ndarray
        Right half of the SIN window (length N/2)
    """
    n = np.arange(N // 2, N)
    window = np.sin(np.pi / N * (n + 0.5))
    return window


def _mdct(x, N):
    """
    Implements the Modified Discrete Cosine Transform (MDCT).
    
    X[k] = 2 * Σ(n=0 to N-1) x[n] * cos(2π/N * (n + n0) * (k + 1/2))
    where n0 = (N/2 + 1) / 2
    
    Parameters
    ----------
    x : numpy.ndarray
        Windowed signal of length N
    N : int
        Signal length (2048 for long, 256 for short)
    
    Returns
    -------
    X : numpy.ndarray
        MDCT coefficients of length N/2
    """
    # Calculate n0
    n0 = (N / 2 + 1) / 2
    
    # Create arrays for n and k
    n = np.arange(N)
    k = np.arange(N // 2)
    
    # Compute MDCT using broadcasting
    # n: (N,), k: (N/2,) -> meshgrid for all combinations
    n_matrix = n[:, np.newaxis]  # (N, 1)
    k_matrix = k[np.newaxis, :]  # (1, N/2)
    
    # Calculate cosine argument
    cos_arg = (2 * np.pi / N) * (n_matrix + n0) * (k_matrix + 0.5)
    
    # MDCT: X[k] = 2 * Σ x[n] * cos(...)
    X = 2 * np.sum(x[:, np.newaxis] * np.cos(cos_arg), axis=0)
    
    return X


def _process_only_long(frame_T, win_type):
    """
    Process ONLY_LONG_SEQUENCE frame.
    
    - Applies SIN window of length 2048
    - Computes MDCT → 1024 coefficients per channel
    """
    if win_type == "SIN":
        window = _create_sin_window(2048)
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # Process both channels
    # frame_T shape: (2048, 2)
    # window shape: (2048,)
    # Broadcasting: window[:, np.newaxis] -> (2048, 1)
    windowed = frame_T * window[:, np.newaxis]
    
    # MDCT for both channels
    # Process each channel separately
    frame_F = np.zeros((1024, 2))
    frame_F[:, 0] = _mdct(windowed[:, 0], 2048)
    frame_F[:, 1] = _mdct(windowed[:, 1], 2048)
    
    return frame_F


def _process_eight_short(frame_T, win_type):
    """
    Process EIGHT_SHORT_SEQUENCE frame.
    
    - Selects central 1152 samples (ignores 448+448)
    - Splits into 8 subframes with 50% overlap (256 samples each)
    - Applies SIN window and MDCT to each subframe
    """
    if win_type == "SIN":
        window = _create_sin_window(256)
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # Select central 1152 samples (skip 448 left and 448 right)
    # frame_T shape: (2048, 2)
    central_samples = frame_T[448:448+1152, :]
    
    # Create 8 subframes with 50% overlap
    # Subframe i: samples [i*128 : i*128+256]
    # Result shape: (8, 128, 2)
    frame_F = np.zeros((8, 128, 2))
    
    for i in range(8):
        start_idx = i * 128
        end_idx = start_idx + 256
        subframe = central_samples[start_idx:end_idx, :]  # shape: (256, 2)
        
        # Windowing (using sqrt for TDAC)
        windowed = subframe * window[:, np.newaxis]
        
        # MDCT for both channels
        frame_F[i, :, 0] = _mdct(windowed[:, 0], 256)
        frame_F[i, :, 1] = _mdct(windowed[:, 1], 256)
    
    return frame_F


def _process_long_start(frame_T, win_type):
    """
    Process LONG_START_SEQUENCE frame.
    
    Asymmetric window of 2048:
    - Wl_left (1024): left half of long window
    - ones (448): constant 1
    - Ws_right (128): right half of short window
    - zeros (448): constant 0
    """
    if win_type == "SIN":
        # Wl_left: left half of SIN window 2048
        wl_left = _create_sin_window_left(2048)
        
        # Ws_right: right half of SIN window 256
        ws_right = _create_sin_window_right(256)
        
        # Construct asymmetric window
        window = np.concatenate([
            wl_left,           # 1024 samples
            np.ones(448),      # 448 samples
            ws_right,          # 128 samples
            np.zeros(448)      # 448 samples
        ])
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # Windowing for both channels (using sqrt for TDAC)
    # frame_T shape: (2048, 2)
    windowed = frame_T * window[:, np.newaxis]
    
    # MDCT for both channels
    frame_F = np.zeros((1024, 2))
    frame_F[:, 0] = _mdct(windowed[:, 0], 2048)
    frame_F[:, 1] = _mdct(windowed[:, 1], 2048)
    
    return frame_F


def _process_long_stop(frame_T, win_type):
    """
    Process LONG_STOP_SEQUENCE frame.
    
    Asymmetric window of 2048:
    - zeros (448): constant 0
    - Ws_left (128): left half of short window
    - ones (448): constant 1
    - Wl_right (1024): right half of long window
    """
    if win_type == "SIN":
        # Ws_left: left half of SIN window 256
        ws_left = _create_sin_window_left(256)
        
        # Wl_right: right half of SIN window 2048
        wl_right = _create_sin_window_right(2048)
        
        # Construct asymmetric window
        window = np.concatenate([
            np.zeros(448),     # 448 samples
            ws_left,           # 128 samples
            np.ones(448),      # 448 samples
            wl_right           # 1024 samples
        ])
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # Windowing for both channels
    # frame_T shape: (2048, 2)
    windowed = frame_T * window[:, np.newaxis]
    
    # MDCT for both channels
    frame_F = np.zeros((1024, 2))
    frame_F[:, 0] = _mdct(windowed[:, 0], 2048)
    frame_F[:, 1] = _mdct(windowed[:, 1], 2048)
    
    return frame_F
