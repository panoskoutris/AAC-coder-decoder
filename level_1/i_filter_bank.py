"""
Inverse Filter Bank Implementation for AAC Decoder (Level 1)
"""

import numpy as np


def i_filter_bank(frame_F, frame_type, win_type):
    """
    Implements the inverse Filterbank stage (IMDCT synthesis).
    
    Parameters
    ----------
    frame_F : numpy.ndarray
        Frame in the frequency domain (MDCT coefficients)
        - For OLS/LSS/LPS: shape (1024, 2) - 1024 coefficients × 2 channels
        - For ESH: shape (8, 128, 2) - 8 subframes × 128 coefficients × 2 channels
    frame_type : str
        Frame type: "OLS", "ESH", "LSS", "LPS"
    win_type : str
        Window type: "KBD" or "SIN"
    
    Returns
    -------
    frame_T : numpy.ndarray
        Frame in the time domain, shape (2048, 2) - 2048 samples × 2 channels
    """
    
    if frame_type == "OLS":
        return _process_only_long_inv(frame_F, win_type)
    elif frame_type == "ESH":
        return _process_eight_short_inv(frame_F, win_type)
    elif frame_type == "LSS":
        return _process_long_start_inv(frame_F, win_type)
    elif frame_type == "LPS":
        return _process_long_stop_inv(frame_F, win_type)
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


def _imdct(X, N):
    """
    Implements the Inverse Modified Discrete Cosine Transform (IMDCT).
    
    s[n] = (2/N) * Σ(k=0 to N/2-1) X[k] * cos(2π/N * (n + n0) * (k + 1/2))
    where n0 = (N/2 + 1) / 2
    
    Parameters
    ----------
    X : numpy.ndarray
        MDCT coefficients of length N/2
    N : int
        Output signal length (2048 for long, 256 for short)
    
    Returns
    -------
    s : numpy.ndarray
        Reconstructed signal of length N
    """
    # Calculate n0
    n0 = (N / 2 + 1) / 2
    
    # Create arrays for n and k
    n = np.arange(N)
    k = np.arange(N // 2)
    
    # Compute IMDCT using broadcasting
    # n: (N,), k: (N/2,)
    n_matrix = n[:, np.newaxis]  # (N, 1)
    k_matrix = k[np.newaxis, :]  # (1, N/2)
    
    # Calculate cosine argument
    cos_arg = (2 * np.pi / N) * (n_matrix + n0) * (k_matrix + 0.5)
    
    # IMDCT: s[n] = (2/N) * Σ X[k] * cos(...)
    s = (2.0 / N) * np.sum(X[np.newaxis, :] * np.cos(cos_arg), axis=1)
    
    return s


def _process_only_long_inv(frame_F, win_type):
    """
    Process inverse ONLY_LONG_SEQUENCE frame.
    
    - Applies IMDCT: 1024 coefficients → 2048 samples
    - Applies SIN window of length 2048
    """
    if win_type == "SIN":
        window = _create_sin_window(2048)
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # IMDCT for both channels
    # frame_F shape: (1024, 2)
    frame_T = np.zeros((2048, 2))
    frame_T[:, 0] = _imdct(frame_F[:, 0], 2048)
    frame_T[:, 1] = _imdct(frame_F[:, 1], 2048)
    
    # Windowing (after IMDCT)
    windowed = frame_T * window[:, np.newaxis]
    
    return windowed


def _process_eight_short_inv(frame_F, win_type):
    """
    Process inverse EIGHT_SHORT_SEQUENCE frame.
    
    - Applies IMDCT to each subframe: 128 coefficients → 256 samples
    - Applies SIN window to each subframe
    - Overlap-adds 8 subframes into central 1152 samples
    - Adds zero padding (448 + 448)
    """
    if win_type == "SIN":
        window = _create_sin_window(256)
    else:
        raise NotImplementedError(f"Window type {win_type} not implemented yet")
    
    # frame_F shape: (8, 128, 2)
    # Reconstruct central 1152 samples with overlap-add
    central_samples = np.zeros((1152, 2))
    
    for i in range(8):
        # IMDCT for both channels
        subframe = np.zeros((256, 2))
        subframe[:, 0] = _imdct(frame_F[i, :, 0], 256)
        subframe[:, 1] = _imdct(frame_F[i, :, 1], 256)
        
        # Windowing
        windowed_subframe = subframe * window[:, np.newaxis]
        
        # Overlap-add into central region
        start_idx = i * 128
        end_idx = start_idx + 256
        central_samples[start_idx:end_idx, :] += windowed_subframe
    
    # Construct full frame with zero padding
    frame_T = np.zeros((2048, 2))
    frame_T[448:448+1152, :] = central_samples
    
    return frame_T


def _process_long_start_inv(frame_F, win_type):
    """
    Process inverse LONG_START_SEQUENCE frame.
    
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
    
    # IMDCT for both channels
    # frame_F shape: (1024, 2)
    frame_T = np.zeros((2048, 2))
    frame_T[:, 0] = _imdct(frame_F[:, 0], 2048)
    frame_T[:, 1] = _imdct(frame_F[:, 1], 2048)
    
    # Windowing (after IMDCT)
    windowed = frame_T * window[:, np.newaxis]
    
    return windowed


def _process_long_stop_inv(frame_F, win_type):
    """
    Process inverse LONG_STOP_SEQUENCE frame.
    
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
    
    # IMDCT for both channels
    # frame_F shape: (1024, 2)
    frame_T = np.zeros((2048, 2))
    frame_T[:, 0] = _imdct(frame_F[:, 0], 2048)
    frame_T[:, 1] = _imdct(frame_F[:, 1], 2048)
    
    # Windowing (after IMDCT)
    windowed = frame_T * window[:, np.newaxis]
    
    return windowed
