"""
Psychoacoustic Model - Level 3

Implements the psychoacoustic model for AAC encoding to compute
Signal-to-Mask Ratio (SMR) for perceptual quantization.

Based on ISO/IEC 13818-7 (MPEG-2 AAC) psychoacoustic model.
"""

import os
import numpy as np
from scipy.io import loadmat


# Load psychoacoustic band tables at module import
_module_dir = os.path.dirname(os.path.abspath(__file__))
_table_path = os.path.join(_module_dir, "TableB219.mat")

try:
    _tables = loadmat(_table_path)
    TABLE_B219a = _tables["B219a"]  # For long windows (1024 FFT)
    TABLE_B219b = _tables["B219b"]  # For short windows (128 FFT)
except FileNotFoundError:
    raise FileNotFoundError(
        f"TableB219.mat not found in {_module_dir}. "
        "Please ensure it exists in the level_3 folder."
    )


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic Model for AAC Encoding.
    
    Computes the Signal-to-Mask Ratio (SMR) for each scalefactor band,
    which guides the quantizer on how much quantization noise is acceptable.
    
    Parameters
    ----------
    frame_T : ndarray
        Current frame time-domain samples, shape (2048, 2) for stereo
    
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", "LPS"
    
    frame_T_prev_1 : ndarray
        Previous frame (t-1), shape (2048, 2)
    
    frame_T_prev_2 : ndarray
        Pre-previous frame (t-2), shape (2048, 2)
    
    Returns
    -------
    SMR : ndarray
        Signal-to-Mask Ratio for each scalefactor band and channel
        Shape:
            - (42, 8) for ESH frames (8 subframes)
            - (69, 1) for OLS/LSS/LPS frames
        Values are in linear scale (not dB)
    """
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":
        # ESH: 8 short subframes of 256 samples each
        return _psycho_esh(frame_T, frame_T_prev_1, frame_T_prev_2)
    else:
        # Long frames (OLS, LSS, LPS): 2048 samples
        return _psycho_long(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2)


def _psycho_long(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic model for long frames (OLS, LSS, LPS).
    
    Uses 2048-sample window and FFT, processing 2 previous frames.
    """
    N = 2048
    N_FFT = 1024  # FFT size (symmetric, so we use half)
    
    # Use Table B.2.1.9.a for long frames
    band_table = TABLE_B219a
    num_bands = band_table.shape[0]  # 69 bands
    
    # Extract stereo channels
    frame_curr = frame_T[:, 0]  # Left channel (can also process right separately)
    frame_prev_1 = frame_T_prev_1[:, 0]
    frame_prev_2 = frame_T_prev_2[:, 0]
    
    # Step 2: Apply Hann window
    n = np.arange(N)
    s_w = _hann_window(n, N)
    frame_windowed = frame_curr * s_w
    
    # Step 3: FFT - compute magnitude r(w) and phase f(w)
    fft_result = np.fft.fft(frame_windowed)
    fft_result = fft_result[:N_FFT]  # Keep only positive frequencies (0 to 1023)
    
    r = np.abs(fft_result)  # Magnitude
    f = np.angle(fft_result)  # Phase
    
    # Also compute FFT for previous frames for predictability
    frame_prev_1_windowed = frame_prev_1 * s_w
    frame_prev_2_windowed = frame_prev_2 * s_w
    
    fft_prev_1 = np.fft.fft(frame_prev_1_windowed)[:N_FFT]
    fft_prev_2 = np.fft.fft(frame_prev_2_windowed)[:N_FFT]
    
    r_prev_1 = np.abs(fft_prev_1)
    f_prev_1 = np.angle(fft_prev_1)
    r_prev_2 = np.abs(fft_prev_2)
    f_prev_2 = np.angle(fft_prev_2)
    
    # Step 4: Compute predictability c(w)
    # Predicted magnitude and phase
    r_pred = 2 * r_prev_1 - r_prev_2
    f_pred = 2 * f_prev_1 - f_prev_2
    
    # Predictability measure
    numerator = np.sqrt(
        (r * np.cos(f) - r_pred * np.cos(f_pred))**2 +
        (r * np.sin(f) - r_pred * np.sin(f_pred))**2
    )
    denominator = r + np.abs(r_pred)
    
    # Avoid division by zero
    c = np.zeros_like(r)
    mask = denominator > 1e-10
    c[mask] = numerator[mask] / denominator[mask]
    
    # Step 5: Compute energy e(b) and predictability c(b) per band
    e = np.zeros(num_bands)
    c_band = np.zeros(num_bands)
    
    for b in range(num_bands):
        w_low = int(band_table[b, 1])
        w_high = int(band_table[b, 2])
        
        if w_low >= N_FFT:
            continue
        w_high = min(w_high, N_FFT - 1)
        
        # Energy in band
        e[b] = np.sum(r[w_low:w_high + 1]**2)
        
        # Weighted predictability
        if e[b] > 1e-10:
            c_band[b] = np.sum(c[w_low:w_high + 1] * r[w_low:w_high + 1]**2)
    
    # Step 6: Apply spreading function
    ecb = np.zeros(num_bands)
    ct = np.zeros(num_bands)
    
    for b in range(num_bands):
        for bb in range(num_bands):
            spread = _spreading_function(bb, b, band_table)
            ecb[b] += e[bb] * spread
            ct[b] += c_band[bb] * spread
    
    # Normalize
    cb = np.zeros(num_bands)
    for b in range(num_bands):
        if ecb[b] > 1e-10:
            cb[b] = ct[b] / ecb[b]
        else:
            cb[b] = 0.0
    
    # Step 7: Tonality index tb(b)
    tb = -0.299 - 0.43 * np.log(cb + 1e-10)
    tb = np.clip(tb, 0.0, 1.0)
    
    # Step 8: SNR (Signal-to-Noise Ratio) per band
    NMT = 6.0  # Noise Masking Tone (dB)
    TMN = 18.0  # Tone Masking Noise (dB)
    
    SNR = tb * TMN + (1 - tb) * NMT
    
    # Step 9: Convert from dB to linear scale
    bc = 10.0 ** (-SNR / 10.0)
    
    # Step 10: Energy normalization en(b)
    en = np.zeros(num_bands)
    for b in range(num_bands):
        # Sum of spreading function (normalization factor)
        spread_sum = 0.0
        for bb in range(num_bands):
            spread_sum += _spreading_function(bb, b, band_table)
        
        if spread_sum > 1e-10:
            en[b] = ecb[b] / spread_sum
    
    # Noise threshold nb(b)
    nb = en * bc
    
    # Step 11: Map to scalefactor bands and compute npart(b)
    # For long frames, scalefactor bands are approximately 1/3 of critical bands
    # We use the qsthr column from the table as a threshold
    q_thr = band_table[:, 5]
    
    # Step 12: Compute npart(b) - the noise masking threshold
    # Use equation (9): npart(b) = max{nb(b), q_thr(b)}
    epsilon = np.finfo(np.float32).eps
    N_fft = 2048  # For long frames
    
    q_thr_linear = (N_fft / 2) * 10**(q_thr / 10.0)
    npart = np.maximum(nb, q_thr_linear)
    
    # Step 13: Signal-to-Mask Ratio SMR(b) = e(b) / npart(b)
    SMR = np.zeros(num_bands)
    for b in range(num_bands):
        if npart[b] > epsilon:
            SMR[b] = e[b] / npart[b]
        else:
            SMR[b] = 1.0
    
    # Return as column vector (69, 1)
    return SMR.reshape(-1, 1).astype(np.float32)


def _psycho_esh(frame_T, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic model for ESH (EIGHT SHORT SEQUENCE) frames.
    
    Processes 8 subframes of 256 samples each.
    """
    num_subframes = 8
    subframe_size = 256
    N_FFT = 128  # FFT size for short frames
    
    # Use Table B.2.1.9.b for short frames
    band_table = TABLE_B219b
    num_bands = band_table.shape[0]  # 42 bands
    
    # Extract left channel
    frame_curr = frame_T[:, 0]
    frame_prev_1 = frame_T_prev_1[:, 0]
    frame_prev_2 = frame_T_prev_2[:, 0]
    
    # Initialize SMR for all subframes
    SMR_all = np.zeros((num_bands, num_subframes), dtype=np.float32)
    
    # Process each subframe
    for s in range(num_subframes):
        start_idx = s * 128  # 50% overlap: 128 samples apart
        
        # Extract subframe (256 samples)
        # For first subframe (s=0), use samples 0-255
        # For subframe 1 (s=1), use 7th and 8th subframes of previous frame
        if s == 0:
            # Use last 2 subframes from previous frame
            curr_subframe = frame_curr[start_idx:start_idx + subframe_size]
            prev_1_subframe = frame_prev_1[-subframe_size:]
            prev_2_subframe = frame_prev_2[-subframe_size:]
        else:
            curr_subframe = frame_curr[start_idx:start_idx + subframe_size]
            prev_1_subframe = frame_curr[start_idx - 128:start_idx - 128 + subframe_size]
            if s >= 2:
                prev_2_subframe = frame_curr[start_idx - 256:start_idx - 256 + subframe_size]
            else:
                # Use from previous frame
                offset = 2048 - 256 + (s - 1) * 128
                prev_2_subframe = frame_prev_1[offset:offset + subframe_size]
        
        # Step 2: Apply Hann window
        n = np.arange(subframe_size)
        s_w = _hann_window(n, subframe_size)
        
        curr_windowed = curr_subframe * s_w
        prev_1_windowed = prev_1_subframe * s_w
        prev_2_windowed = prev_2_subframe * s_w
        
        # Step 3: FFT
        fft_curr = np.fft.fft(curr_windowed)[:N_FFT]
        fft_prev_1 = np.fft.fft(prev_1_windowed)[:N_FFT]
        fft_prev_2 = np.fft.fft(prev_2_windowed)[:N_FFT]
        
        r = np.abs(fft_curr)
        f = np.angle(fft_curr)
        r_prev_1 = np.abs(fft_prev_1)
        f_prev_1 = np.angle(fft_prev_1)
        r_prev_2 = np.abs(fft_prev_2)
        f_prev_2 = np.angle(fft_prev_2)
        
        # Step 4: Predictability
        r_pred = 2 * r_prev_1 - r_prev_2
        f_pred = 2 * f_prev_1 - f_prev_2
        
        numerator = np.sqrt(
            (r * np.cos(f) - r_pred * np.cos(f_pred))**2 +
            (r * np.sin(f) - r_pred * np.sin(f_pred))**2
        )
        denominator = r + np.abs(r_pred)
        
        c = np.zeros_like(r)
        mask = denominator > 1e-10
        c[mask] = numerator[mask] / denominator[mask]
        
        # Step 5: Energy and predictability per band
        e = np.zeros(num_bands)
        c_band = np.zeros(num_bands)
        
        for b in range(num_bands):
            w_low = int(band_table[b, 1])
            w_high = int(band_table[b, 2])
            
            if w_low >= N_FFT:
                continue
            w_high = min(w_high, N_FFT - 1)
            
            e[b] = np.sum(r[w_low:w_high + 1]**2)
            if e[b] > 1e-10:
                c_band[b] = np.sum(c[w_low:w_high + 1] * r[w_low:w_high + 1]**2)
        
        # Step 6: Apply spreading function
        ecb = np.zeros(num_bands)
        ct = np.zeros(num_bands)
        
        for b in range(num_bands):
            for bb in range(num_bands):
                spread = _spreading_function(bb, b, band_table)
                ecb[b] += e[bb] * spread
                ct[b] += c_band[bb] * spread
        
        cb = np.zeros(num_bands)
        for b in range(num_bands):
            if ecb[b] > 1e-10:
                cb[b] = ct[b] / ecb[b]
        
        # Step 7: Tonality index
        tb = -0.299 - 0.43 * np.log(cb + 1e-10)
        tb = np.clip(tb, 0.0, 1.0)
        
        # Step 8: SNR
        NMT = 6.0
        TMN = 18.0
        SNR = tb * TMN + (1 - tb) * NMT
        
        # Step 9: Convert to linear
        bc = 10.0 ** (-SNR / 10.0)
        
        # Step 10: Normalization
        en = np.zeros(num_bands)
        for b in range(num_bands):
            spread_sum = 0.0
            for bb in range(num_bands):
                spread_sum += _spreading_function(bb, b, band_table)
            if spread_sum > 1e-10:
                en[b] = ecb[b] / spread_sum
        
        nb = en * bc
        
        # Step 11-12: Noise threshold with qsthr
        q_thr = band_table[:, 5]
        N_fft = 256
        q_thr_linear = (N_fft / 2) * 10**(q_thr / 10.0)
        npart = np.maximum(nb, q_thr_linear)
        
        # Step 13: SMR
        epsilon = np.finfo(np.float32).eps
        SMR = np.zeros(num_bands)
        for b in range(num_bands):
            if npart[b] > epsilon:
                SMR[b] = e[b] / npart[b]
            else:
                SMR[b] = 1.0
        
        SMR_all[:, s] = SMR
    
    return SMR_all.astype(np.float32)


def _hann_window(n, N):
    """
    Compute Hann window.
    
    s_w(n) = s(n) * (0.5 - 0.5 * cos(π(n + 0.5) / N))
    
    where s(n) is the time-domain signal.
    
    Parameters
    ----------
    n : ndarray
        Sample indices
    N : int
        Window length
    
    Returns
    -------
    window : ndarray
        Hann window values
    """
    return 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)


def _spreading_function(bb, b, band_table):
    """
    Compute spreading function between bands bb and b.
    
    Implements the masking spread from one critical band to another.
    
    Parameters
    ----------
    bb : int
        Source band index
    b : int
        Target band index
    band_table : ndarray
        Band table (B219a or B219b)
    
    Returns
    -------
    x : float
        Spreading function value
    """
    if bb >= len(band_table) or b >= len(band_table):
        return 0.0
    
    # Get bval for each band (column 4, index 4)
    bval_bb = band_table[bb, 4]
    bval_b = band_table[b, 4]
    
    # Compute tmpx based on relationship between bands
    if bb >= b:
        tmpx = 3.0 * (bval_b - bval_bb)
    else:
        tmpx = 1.5 * (bval_b - bval_bb)
    
    # Compute tmpz
    tmpz = 8.0 * min((tmpx - 0.5)**2 - 2.0 * (tmpx - 0.5), 0.0)
    
    # Compute tmpy
    tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * np.sqrt(1.0 + (tmpx + 0.474)**2)
    
    # Check condition
    if tmpy < -100:
        x = 0.0
    else:
        # x = 10^((tmpz + tmpy) / 10)
        x = 10.0 ** ((tmpz + tmpy) / 10.0)
    
    return x
