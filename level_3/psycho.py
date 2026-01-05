"""
Psychoacoustic Model - Level 3

Implements the AAC psychoacoustic model to compute Signal-to-Mask Ratio (SMR)
for perceptual audio coding. The model determines how much quantization noise
is acceptable in each frequency band based on masking effects.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.fft import fft


# Load psychoacoustic band tables at module import
_module_dir = os.path.dirname(os.path.abspath(__file__))
_table_path = os.path.join(_module_dir, "TableB219.mat")

try:
    _tables = loadmat(_table_path)
    TABLE_B219a = _tables["B219a"]  # For long windows (1024 MDCT coeffs, 69 bands)
    TABLE_B219b = _tables["B219b"]  # For short windows (128 MDCT coeffs, 42 bands)
except FileNotFoundError:
    raise FileNotFoundError(
        f"TableB219.mat not found in {_module_dir}. "
        "Please ensure it exists in the level_3 folder."
    )


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic Model for AAC encoding (single channel).
    
    Computes the Signal-to-Mask Ratio (SMR) for each psychoacoustic band,
    which determines how much quantization noise is perceptually acceptable.
    For stereo audio, call this function twice (once per channel).
    
    Parameters
    ----------
    frame_T : ndarray
        Current time-domain frame for one channel, shape (2048,)
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", "LPS"
    frame_T_prev_1 : ndarray
        Previous frame for one channel (1 frame back), shape (2048,)
    frame_T_prev_2 : ndarray
        Frame before previous for one channel (2 frames back), shape (2048,)
    
    Returns
    -------
    SMR : ndarray
        Signal to Mask Ratio
        Shape (42, 8) for ESH (EIGHT_SHORT_SEQUENCE) frames
        Shape (69, 1) for all other frame types (OLS, LSS, LPS)
    """
    # Normalize frame type
    frame_type = frame_type.upper()
    
    # Determine which table and parameters to use
    if frame_type == "ESH":
        band_table = TABLE_B219b  # 42 bands for short windows
        spread_matrix = SPREADING_MATRIX_SHORT
        num_subframes = 8
        N = 256  # FFT length for short subframes
        num_bands = 42
    else:
        band_table = TABLE_B219a  # 69 bands for long windows
        spread_matrix = SPREADING_MATRIX_LONG
        num_subframes = 1
        N = 2048  # FFT length for long frames
        num_bands = 69
    
    # Initialize SMR output
    if frame_type == "ESH":
        SMR = np.zeros((num_bands, num_subframes), dtype=np.float32)
    else:
        SMR = np.zeros((num_bands, 1), dtype=np.float32)
    
    # Step 2: FFT analysis for current frame
    r_list, f_list = analyze_frame_fft(frame_T, frame_type)
    
    # Step 2: FFT analysis for previous frames
    r_prev_1_list, f_prev_1_list = analyze_frame_fft(frame_T_prev_1, frame_type)
    r_prev_2_list, f_prev_2_list = analyze_frame_fft(frame_T_prev_2, frame_type)
    
    # Compute absolute threshold once (same for all subframes)
    q_thr = compute_absolute_threshold(band_table, N)
    
    # Process each subframe (1 for long frames, 8 for ESH)
    for s in range(num_subframes):
        # Step 3: Compute predictions
        if frame_type == "ESH":
            r_pred, f_pred = compute_prediction_for_subframe(s, r_list, f_list, 
                                                             r_prev_1_list, f_prev_1_list)
        else:
            # For long frames, use previous two complete frames
            r_minus_1 = r_prev_1_list[0]
            f_minus_1 = f_prev_1_list[0]
            r_minus_2 = r_prev_2_list[0]
            f_minus_2 = f_prev_2_list[0]
            r_pred = 2 * r_minus_1 - r_minus_2
            f_pred = 2 * f_minus_1 - f_minus_2
        
        # Step 4: Compute predictability
        c = compute_predictability(r_list[s], f_list[s], r_pred, f_pred)
        
        # Step 5: Compute band energy and predictability
        e_band, c_band = compute_band_energy_and_predictability(r_list[s], c, band_table)
        
        # Step 6: Convolve with spreading function
        cb, en = convolve_with_spreading_function(e_band, c_band, spread_matrix)
        
        # Step 7: Compute tonality index
        tb = compute_tonality_index(cb)
        
        # Step 8: Compute required SNR
        SNR = compute_required_snr(tb)
        
        # Step 9: Convert to energy ratio
        bc = convert_snr_to_energy_ratio(SNR)
        
        # Step 10: Compute energy threshold
        nb = compute_energy_threshold(en, bc)
        
        # Step 11: Apply absolute threshold
        npart = apply_absolute_threshold(nb, q_thr)
        
        # Step 12: Compute SMR
        SMR_subframe = compute_smr(e_band, npart)
        
        # Store in output
        if frame_type == "ESH":
            SMR[:, s] = SMR_subframe
        else:
            SMR[:, 0] = SMR_subframe
    
    # Return appropriate shape
    # For ESH: (42, 8)
    # For others: (69,) - flatten to 1D
    if frame_type != "ESH":
        SMR = SMR.flatten()
    
    return SMR


def spreading_function(i, j, bval):
    """
    Compute the spreading function between two frequency bands.
    
    The spreading function models how masking energy spreads across
    frequency bands based on psychoacoustic principles.
    
    Parameters
    ----------
    i : int
        Index of the masker band
    j : int
        Index of the maskee band (band being masked)
    bval : ndarray
        Center frequency (Bark scale) for each band from the table
    
    Returns
    -------
    x : float
        Spreading factor from band i to band j
    """
    # Step 1-2: Compute tmpx based on band relationship
    if i >= j:
        tmpx = 3.0 * (bval[j] - bval[i])
    else:
        tmpx = 1.5 * (bval[j] - bval[i])
    
    # Step 7: Compute tmpz
    tmpz = 8 * min((tmpx - 0.5)**2 - 2*(tmpx - 0.5), 0)
    
    # Step 8: Compute tmpy
    tmpy = 15.811389 + 7.5*(tmpx + 0.474) - 17.5*np.sqrt(1.0 + (tmpx + 0.474)**2)
    
    # Step 9-12: Compute final spreading value
    if tmpy < -100:
        x = 0
    else:
        x = 10**((tmpz + tmpy) / 10)
    
    return x


def precompute_spreading_matrix(band_table):
    """
    Precompute the spreading function matrix for all band combinations.
    
    This creates a matrix where spread_matrix[i, j] contains the spreading
    factor from band i to band j. This can be computed once and reused.
    
    Parameters
    ----------
    band_table : ndarray
        Psychoacoustic band table (B219a or B219b)
        Columns: [index, w_low, w_high, width, bval, qsthr]
    
    Returns
    -------
    spread_matrix : ndarray
        Shape (num_bands, num_bands)
        spread_matrix[i, j] = spreading from band i to band j
    """
    num_bands = band_table.shape[0]
    bval = band_table[:, 4]  # Column 4 is bval
    
    spread_matrix = np.zeros((num_bands, num_bands), dtype=np.float32)
    
    for i in range(num_bands):
        for j in range(num_bands):
            spread_matrix[i, j] = spreading_function(i, j, bval)
    
    return spread_matrix


# Precompute spreading matrices for both long and short windows
SPREADING_MATRIX_LONG = precompute_spreading_matrix(TABLE_B219a)
SPREADING_MATRIX_SHORT = precompute_spreading_matrix(TABLE_B219b)


def apply_hann_window(signal):
    """
    Apply Hann window to a time-domain signal.
    
    Formula: s_w(n) = s(n) * (0.5 - 0.5 * cos(π(n + 0.5) / N))
    
    Parameters
    ----------
    signal : ndarray
        Time-domain signal, shape (N,) where N is window length
        (2048 for long frames, 256 for short subframes)
    
    Returns
    -------
    windowed_signal : ndarray
        Windowed signal, same shape as input
    """
    N = len(signal)
    n = np.arange(N)
    
    # Hann window formula
    window = 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)
    
    windowed_signal = signal * window
    
    return windowed_signal


def compute_fft_analysis(windowed_signal):
    """
    Compute FFT and extract magnitude and phase for each frequency bin.
    
    Parameters
    ----------
    windowed_signal : ndarray
        Windowed time-domain signal, shape (N,)
        N = 2048 for long frames, 256 for short subframes
    
    Returns
    -------
    r : ndarray
        Magnitude spectrum, shape (N//2 + 1,)
        Frequency indices 0 to 1023 for long frames (N=2048)
        Frequency indices 0 to 127 for short frames (N=256)
    f : ndarray
        Phase spectrum (in radians), shape (N//2 + 1,)
    """
    N = len(windowed_signal)
    
    # Compute FFT
    spectrum = fft(windowed_signal)
    
    # Keep only positive frequencies (0 to N//2, plus DC)
    # For N=2048: indices 0 to 1024 (1025 points)
    # For N=256: indices 0 to 128 (129 points)
    half_spectrum = spectrum[:N//2 + 1]
    
    # Compute magnitude
    r = np.abs(half_spectrum)
    
    # Compute phase
    f = np.angle(half_spectrum)
    
    return r, f


def analyze_frame_fft(frame_T, frame_type):
    """
    Analyze a frame using FFT after Hann windowing.
    
    For long frames (OLS, LSS, LPS): process entire 2048-sample frame
    For ESH: process 8 subframes of 256 samples each
    
    Parameters
    ----------
    frame_T : ndarray
        Time-domain frame, shape (2048,) for single channel
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", "LPS"
    
    Returns
    -------
    r_list : list of ndarray
        List of magnitude spectra
        For ESH: list of 8 arrays, each shape (128,)
        For others: list of 1 array, shape (1024,)
    f_list : list of ndarray
        List of phase spectra (same structure as r_list)
    """
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":
        # Process 8 short subframes of 256 samples each
        r_list = []
        f_list = []
        
        for i in range(8):
            # Extract 256-sample subframe
            start_idx = i * 256
            end_idx = start_idx + 256
            subframe = frame_T[start_idx:end_idx]
            
            # Apply Hann window
            windowed = apply_hann_window(subframe)
            
            # Compute FFT
            r, f = compute_fft_analysis(windowed)
            
            # Keep only first 128 coefficients (0 to 127)
            # Note: compute_fft_analysis returns 0 to 128, we want 0 to 127
            r_list.append(r[:128])
            f_list.append(f[:128])
        
        return r_list, f_list
    
    else:
        # Process entire 2048-sample frame
        # Apply Hann window
        windowed = apply_hann_window(frame_T)
        
        # Compute FFT
        r, f = compute_fft_analysis(windowed)
        
        # Keep only first 1024 coefficients (0 to 1023)
        # Note: compute_fft_analysis returns 0 to 1024, we want 0 to 1023
        r_list = [r[:1024]]
        f_list = [f[:1024]]
        
        return r_list, f_list


def compute_prediction_for_subframe(i, r_list, f_list, r_prev_1_list, f_prev_1_list):
    """
    Compute predicted magnitude and phase for a single subframe.
    
    Uses exactly 2 subframes back in chronological time:
    - Subframe 0: 1 back = prev_1[7], 2 back = prev_1[6]
    - Subframe 1: 1 back = current[0], 2 back = prev_1[7]
    - Subframe 2: 1 back = current[1], 2 back = current[0]
    - Subframe i (i>=2): 1 back = current[i-1], 2 back = current[i-2]
    
    Parameters
    ----------
    i : int
        Subframe index (0-7)
    r_list : list of ndarray
        Current frame magnitude spectra (being built)
    f_list : list of ndarray
        Current frame phase spectra (being built)
    r_prev_1_list : list of ndarray
        Previous frame magnitude spectra
    f_prev_1_list : list of ndarray
        Previous frame phase spectra
    
    Returns
    -------
    r_pred : ndarray
        Predicted magnitude spectrum for subframe i
    f_pred : ndarray
        Predicted phase spectrum for subframe i
    """
    if i == 0:
        # 1st subframe: use last 2 from previous frame
        r_minus_1 = r_prev_1_list[7]
        f_minus_1 = f_prev_1_list[7]
        r_minus_2 = r_prev_1_list[6]
        f_minus_2 = f_prev_1_list[6]
    elif i == 1:
        # 2nd subframe: use current[0] and prev_1[7]
        r_minus_1 = r_list[0]
        f_minus_1 = f_list[0]
        r_minus_2 = r_prev_1_list[7]
        f_minus_2 = f_prev_1_list[7]
    else:
        # Subframes 2-7: use previous 2 from current frame
        r_minus_1 = r_list[i - 1]
        f_minus_1 = f_list[i - 1]
        r_minus_2 = r_list[i - 2]
        f_minus_2 = f_list[i - 2]
    
    # Compute predictions
    r_pred = 2 * r_minus_1 - r_minus_2
    f_pred = 2 * f_minus_1 - f_minus_2
    
    return r_pred, f_pred


def compute_predictability(r, f, r_pred, f_pred):
    """
    Compute the predictability measure c(w) for each frequency bin.
    
    This measures how well the current spectrum can be predicted from
    previous frames, which helps identify tonal vs. noise-like components.
    
    Formula:
        c(w) = sqrt((r(w)cos(f(w)) - r_pred(w)cos(f_pred(w)))^2 + 
                    (r(w)sin(f(w)) - r_pred(w)sin(f_pred(w)))^2) / 
               (r(w) + |r_pred(w)|)
    
    Parameters
    ----------
    r : ndarray
        Current magnitude spectrum, shape (N,)
    f : ndarray
        Current phase spectrum, shape (N,)
    r_pred : ndarray
        Predicted magnitude spectrum, shape (N,)
    f_pred : ndarray
        Predicted phase spectrum, shape (N,)
    
    Returns
    -------
    c : ndarray
        Predictability measure for each frequency bin, shape (N,)
        Values range from 0 (perfectly predictable) to higher values (unpredictable)
    """
    # Compute real and imaginary parts of current spectrum
    real_current = r * np.cos(f)
    imag_current = r * np.sin(f)
    
    # Compute real and imaginary parts of predicted spectrum
    real_pred = r_pred * np.cos(f_pred)
    imag_pred = r_pred * np.sin(f_pred)
    
    # Compute differences
    real_diff = real_current - real_pred
    imag_diff = imag_current - imag_pred
    
    # Compute numerator: sqrt of sum of squared differences
    numerator = np.sqrt(real_diff**2 + imag_diff**2)
    
    # Compute denominator: r(w) + |r_pred(w)|
    denominator = r + np.abs(r_pred)
    
    # Avoid division by zero
    denominator = np.where(denominator > 1e-10, denominator, 1e-10)
    
    # Compute predictability measure
    c = numerator / denominator
    
    return c


def compute_band_energy_and_predictability(r, c, band_table):
    """
    Compute energy and weighted predictability for each psychoacoustic band.
    
    For each band b with index from the table:
        e(b) = Σ(w=w_low(b) to w_high(b)) r(w)^2
        c(b) = Σ(w=w_low(b) to w_high(b)) c(w)*r(w)^2
    
    The bands correspond to approximately 1/3 of the critical bands.
    
    Parameters
    ----------
    r : ndarray
        Magnitude spectrum, shape (N,)
        N = 1024 for long frames, 128 for short subframes
    c : ndarray
        Predictability measure for each frequency bin, shape (N,)
    band_table : ndarray
        Psychoacoustic band table (B219a or B219b)
        Columns: [index, w_low, w_high, width, bval, qsthr]
    
    Returns
    -------
    e_band : ndarray
        Energy for each band, shape (num_bands,)
    c_band : ndarray
        Weighted predictability for each band, shape (num_bands,)
    """
    num_bands = band_table.shape[0]
    N = len(r)
    
    # Extract w_low and w_high from table (columns 1 and 2)
    w_low = band_table[:, 1].astype(int)
    w_high = band_table[:, 2].astype(int)
    
    # Initialize output arrays
    e_band = np.zeros(num_bands, dtype=np.float32)
    c_band = np.zeros(num_bands, dtype=np.float32)
    
    # Compute energy and weighted predictability for each band
    for b in range(num_bands):
        # Get frequency range for this band
        w_start = w_low[b]
        w_end = w_high[b]
        
        # Ensure we don't exceed FFT length
        if w_start >= N:
            break
        w_end = min(w_end, N - 1)
        
        # Sum over frequency bins in this band (inclusive range)
        for w in range(w_start, w_end + 1):
            r_squared = r[w] ** 2
            e_band[b] += r_squared
            c_band[b] += c[w] * r_squared
    
    return e_band, c_band


def convolve_with_spreading_function(e_band, c_band, spread_matrix):
    """
    Convolve energy and predictability with the spreading function.
    
    This models how masking energy spreads across frequency bands.
    
    Formulas:
        ecb(b) = Σ(bb=0 to N_B-1) e(bb) * spreading_function(bb, b)
        ct(b) = Σ(bb=0 to N_B-1) c(bb) * spreading_function(bb, b)
    
    Then normalize:
        cb(b) = ct(b) / ecb(b)
        en(b) = ecb(b) / Σ(bb=0 to N_B-1) spreading_function(bb, b)
    
    Parameters
    ----------
    e_band : ndarray
        Energy for each band, shape (N_B,)
    c_band : ndarray
        Weighted predictability for each band, shape (N_B,)
    spread_matrix : ndarray
        Precomputed spreading function matrix, shape (N_B, N_B)
        spread_matrix[bb, b] = spreading from band bb to band b
    
    Returns
    -------
    cb : ndarray
        Normalized predictability for each band, shape (N_B,)
    en : ndarray
        Normalized energy for each band, shape (N_B,)
    """
    N_B = len(e_band)
    
    # Compute ecb(b) = Σ(bb=0 to N_B-1) e(bb) * spreading_function(bb, b)
    # This is matrix-vector multiplication: spread_matrix^T @ e_band
    # Since spread_matrix[bb, b] gives spreading from bb to b,
    # we need to sum over bb (rows) for each b (column)
    ecb = np.zeros(N_B, dtype=np.float32)
    ct = np.zeros(N_B, dtype=np.float32)
    
    for b in range(N_B):
        for bb in range(N_B):
            ecb[b] += e_band[bb] * spread_matrix[bb, b]
            ct[b] += c_band[bb] * spread_matrix[bb, b]
    
    # Normalize: cb(b) = ct(b) / ecb(b)
    cb = np.zeros(N_B, dtype=np.float32)
    for b in range(N_B):
        if ecb[b] > 1e-10:
            cb[b] = ct[b] / ecb[b]
        else:
            cb[b] = 0.0
    
    # Compute en(b) = ecb(b) / Σ(bb=0 to N_B-1) spreading_function(bb, b)
    en = np.zeros(N_B, dtype=np.float32)
    for b in range(N_B):
        # Sum of spreading function for this band b
        spread_sum = np.sum(spread_matrix[:, b])
        if spread_sum > 1e-10:
            en[b] = ecb[b] / spread_sum
        else:
            en[b] = 0.0
    
    return cb, en


def compute_tonality_index(cb):
    """
    Compute the tonality index tb(b) for each band from normalized predictability.
    
    Formula:
        tb(b) = -0.299 - 0.43 * ln(cb(b))
    
    The tonality index takes values in the range (0, 1), where:
    - Higher values indicate more tonal (predictable) components
    - Lower values indicate more noise-like (unpredictable) components
    
    Parameters
    ----------
    cb : ndarray
        Normalized predictability for each band, shape (N_B,)
    
    Returns
    -------
    tb : ndarray
        Tonality index for each band, shape (N_B,)
        Values are clipped to the range [0, 1]
    """
    N_B = len(cb)
    tb = np.zeros(N_B, dtype=np.float32)
    
    for b in range(N_B):
        if cb[b] > 1e-10:  # Avoid log(0)
            tb[b] = -0.299 - 0.43 * np.log(cb[b])
        else:
            # If cb is very small, set to maximum tonality
            tb[b] = 1.0
    
    # Clip to range [0, 1]
    tb = np.clip(tb, 0.0, 1.0)
    
    return tb


def compute_required_snr(tb):
    """
    Compute the required SNR for each band based on tonality index.
    
    Uses two constant values:
    - NMT (Noise Masking Tone) = 6 dB for all bands
    - TMN (Tone Masking Noise) = 18 dB for all bands
    
    Formula:
        SNR(b) = tb(b) * TMN(b) + (1 - tb(b)) * NMT(b)
    
    Interpretation:
    - When tb(b) → 1 (tonal): SNR(b) → TMN = 18 dB
      (tonal signal needs higher SNR - less quantization noise allowed)
    - When tb(b) → 0 (noise-like): SNR(b) → NMT = 6 dB
      (noise signal needs lower SNR - more quantization noise allowed)
    
    Note: There appears to be an error in the specification text which states
    the opposite values, but the formula and defined constants (TMN=18, NMT=6)
    are implemented as originally specified at the beginning of step 8.
    
    Parameters
    ----------
    tb : ndarray
        Tonality index for each band, shape (N_B,)
    
    Returns
    -------
    SNR : ndarray
        Required SNR for each band in dB, shape (N_B,)
    """
    # Constants from psychoacoustic model (as defined at start of step 8)
    NMT = 6.0   # Noise Masking Tone (dB)
    TMN = 18.0  # Tone Masking Noise (dB)
    
    # Compute weighted SNR based on tonality
    SNR = tb * TMN + (1.0 - tb) * NMT
    
    return SNR


def convert_snr_to_energy_ratio(SNR):
    """
    Convert SNR from dB to linear energy ratio.
    
    Formula:
        bc(b) = 10^(-SNR(b)/10)
    
    This converts the required SNR in decibels to a linear ratio that
    represents the allowed noise-to-signal energy ratio.
    
    Parameters
    ----------
    SNR : ndarray
        Required SNR for each band in dB, shape (N_B,)
    
    Returns
    -------
    bc : ndarray
        Energy ratio (noise-to-signal) for each band, shape (N_B,)
        Lower values mean less noise is allowed (higher quality requirement)
    """
    bc = 10.0 ** (-SNR / 10.0)
    
    return bc


def compute_energy_threshold(en, bc):
    """
    Compute the energy threshold (masking threshold) for each band.
    
    Formula:
        nb(b) = en(b) * bc(b)
    
    This combines the normalized signal energy with the allowed noise-to-signal
    ratio to determine the absolute masking threshold - the maximum amount of
    quantization noise energy that can be added to each band without being
    perceptually audible.
    
    Parameters
    ----------
    en : ndarray
        Normalized energy for each band, shape (N_B,)
    bc : ndarray
        Energy ratio (noise-to-signal) for each band, shape (N_B,)
    
    Returns
    -------
    nb : ndarray
        Energy threshold (masking threshold) for each band, shape (N_B,)
    """
    nb = en * bc
    
    return nb


def compute_absolute_threshold(band_table, N):
    """
    Compute the absolute threshold of hearing for each band.
    
    Formula:
        q̂_thr = ε * (N/2) * 10^(qsthr/10)
    
    where:
    - ε is machine epsilon (numpy.finfo('float').eps)
    - N is the FFT length (2048 for long frames, 256 for short)
    - qsthr is from column 5 of the band table
    
    Parameters
    ----------
    band_table : ndarray
        Psychoacoustic band table (B219a or B219b)
        Columns: [index, w_low, w_high, width, bval, qsthr]
    N : int
        FFT length (2048 for long frames, 256 for short subframes)
    
    Returns
    -------
    q_thr : ndarray
        Absolute threshold of hearing for each band, shape (N_B,)
    """
    # Get qsthr from column 5 of the table
    qsthr = band_table[:, 5]
    
    # Machine epsilon (float64 as per specification)
    eps = np.finfo(float).eps
    
    # Compute absolute threshold
    q_thr = eps * (N / 2.0) * (10.0 ** (qsthr / 10.0))
    
    return q_thr


def apply_absolute_threshold(nb, q_thr):
    """
    Apply absolute threshold of hearing to compute final noise part.
    
    Formula:
        npart(b) = max{nb(b), q̂_thr(b)}
    
    This ensures the masking threshold is at least the absolute threshold
    of hearing. Based on Eq. (9), at each position the hearing threshold
    is the maximum of the quiet threshold and the masking threshold.
    
    Parameters
    ----------
    nb : ndarray
        Energy threshold (masking threshold) for each band, shape (N_B,)
    q_thr : ndarray
        Absolute threshold of hearing for each band, shape (N_B,)
    
    Returns
    -------
    npart : ndarray
        Final noise part (masking threshold) for each band, shape (N_B,)
    """
    npart = np.maximum(nb, q_thr)
    
    return npart


def compute_smr(e_band, npart):
    """
    Compute the Signal to Mask Ratio (SMR) for each band.
    
    Formula:
        SMR(b) = e(b) / npart(b)
    
    SMR represents how much the signal energy exceeds the masking threshold
    in each band. Higher SMR means more bits are needed for that band to
    maintain perceptual quality.
    
    Parameters
    ----------
    e_band : ndarray
        Energy for each band, shape (N_B,)
    npart : ndarray
        Final noise part (masking threshold) for each band, shape (N_B,)
    
    Returns
    -------
    SMR : ndarray
        Signal to Mask Ratio for each band, shape (N_B,)
    """
    # Avoid division by zero
    npart_safe = np.where(npart > 1e-10, npart, 1e-10)
    
    SMR = e_band / npart_safe
    
    return SMR
