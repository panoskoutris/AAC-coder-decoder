"""
Temporal Noise Shaping (TNS) - Level 2

Implements TNS (Temporal Noise Shaping) for one channel of the AAC encoder.
TNS is applied to MDCT coefficients to shape quantization noise in the time
domain, improving coding quality for signals with strong temporal variations.
"""

import os
import numpy as np
from scipy.io import loadmat


# Load psychoacoustic band tables at module import
_module_dir = os.path.dirname(os.path.abspath(__file__))
_table_path = os.path.join(_module_dir, "TableB219.mat")

try:
    _tables = loadmat(_table_path)
    TABLE_B219a = _tables["B219a"]  # For long windows (1024 MDCT coeffs)
    TABLE_B219b = _tables["B219b"]  # For short windows (128 MDCT coeffs)
except FileNotFoundError:
    raise FileNotFoundError(
        f"TableB219.mat not found in {_module_dir}. "
        "Please ensure it exists in the level_2 folder."
    )


def tns(frame_F_in, frame_type):
    """
    Apply Temporal Noise Shaping (TNS) to one channel of MDCT coefficients.
    
    TNS uses 4th-order LPC analysis to shape quantization noise in the time
    domain, improving quality for transient signals.
    
    Parameters
    ----------
    frame_F_in : ndarray
        MDCT coefficients before TNS.
        Shape:
            - (1024, 1) for OLS, LSS, LPS frames
            - (128, 8) for ESH (8 short subframes)
    
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", "LPS"
    
    Returns
    -------
    frame_F_out : ndarray
        MDCT coefficients after TNS (same shape as frame_F_in)
    
    tns_coeffs : ndarray
        Quantized LPC coefficients (4 coefficients)
        Shape:
            - (4, 1) for non-ESH frames
            - (4, 8) for ESH frames (one set per subframe)
    """
    # Normalize frame type
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":
        # ESH: 8 short subframes, each with 128 MDCT coefficients
        num_subframes = 8
        frame_F_out = np.zeros_like(frame_F_in, dtype=np.float32)
        tns_coeffs = np.zeros((4, num_subframes), dtype=np.float32)
        
        # Process each subframe separately with short-window table (B219b)
        for s in range(num_subframes):
            X = frame_F_in[:, s].astype(np.float32).copy()
            Y, a_q = _apply_tns_to_subframe(X, TABLE_B219b)
            
            frame_F_out[:, s] = Y
            tns_coeffs[:, s] = a_q
        
        return frame_F_out, tns_coeffs
    
    else:
        # Long frames (OLS, LSS, LPS): 1024 MDCT coefficients
        X = frame_F_in.flatten().astype(np.float32).copy()
        Y, a_q = _apply_tns_to_subframe(X, TABLE_B219a)
        
        # Reshape back to column vector (1024, 1)
        frame_F_out = Y.reshape(-1, 1)
        tns_coeffs = a_q.reshape(-1, 1)
        
        return frame_F_out, tns_coeffs


def _apply_tns_to_subframe(X, band_table):
    """
    Apply TNS to a single MDCT frame/subframe.
    
    Process:
    1. Compute band energies P(j) from band table
    2. Build and smooth Sw(k)
    3. Normalization: Xw(k) = X(k) / Sw(k)
    4. LPC analysis (autocorrelation method, p=4)
    5. Coefficient quantization (4-bit, step=0.1)
    6. Stability check
    7. Apply FIR filter to original X(k)
    
    Parameters
    ----------
    X : ndarray
        1-D array of MDCT coefficients (length N=1024 or 128)
    
    band_table : ndarray
        Psychoacoustic band table (B219a or B219b)
        Columns: [index, w_low, w_high, width, bval, qsthr]
    
    Returns
    -------
    Y : ndarray
        TNS-filtered MDCT coefficients (same length as X)
    
    a_q : ndarray
        Quantized LPC coefficients (length 4)
    """
    X = np.asarray(X, dtype=np.float32)
    N = len(X)
    
    # LPC order (according to AAC standard)
    LPC_ORDER = 4
    QUANT_STEP = 0.1  # 4-bit quantization step
    
    # === 1. Compute Band Energies P(j) (Eq. 3) ===
    w_low = band_table[:, 1].astype(int)
    w_high = band_table[:, 2].astype(int)
    
    # Keep only bands that overlap with MDCT length
    valid_mask = w_low < N
    w_low = w_low[valid_mask]
    w_high = w_high[valid_mask]
    num_bands = len(w_low)
    
    # Compute energy per band
    P = np.zeros(num_bands, dtype=np.float32)
    for j in range(num_bands):
        k_start = w_low[j]
        k_end = min(w_high[j], N - 1)
        
        if k_start <= k_end:
            band_coeffs = X[k_start : k_end + 1]
            P[j] = np.sum(band_coeffs ** 2)
    
    # === 2. Build Sw(k) and Smoothing (Eq. 4) ===
    Sw = np.ones(N, dtype=np.float32)  # Default = 1 to avoid division by zero
    
    # Assign sqrt(P(j)) to each band
    for j in range(num_bands):
        k_start = w_low[j]
        k_end = min(w_high[j], N - 1)
        
        if k_start <= k_end and P[j] > 0:
            Sw[k_start : k_end + 1] = np.sqrt(P[j])
    
    # Two-pass smoothing as specified in the assignment
    # First pass: right to left
    for k in range(N - 2, -1, -1):
        Sw[k] = 0.5 * (Sw[k] + Sw[k + 1])
    
    # Second pass: left to right
    for k in range(1, N):
        Sw[k] = 0.5 * (Sw[k] + Sw[k - 1])
    
    # === 3. Normalization: Xw(k) = X(k) / Sw(k) (Eq. 2) ===
    Xw = X / Sw
    
    # === 4. LPC Analysis - Autocorrelation Method (Eqs. 5-7) ===
    # Compute autocorrelation r(l) for l = 0..p
    r = np.zeros(LPC_ORDER + 1, dtype=np.float32)
    for lag in range(LPC_ORDER + 1):
        if lag < N:
            r[lag] = np.dot(Xw[lag:], Xw[: N - lag])
    
    # If energy is nearly zero, disable TNS
    if r[0] <= 1e-12:
        return X.copy(), np.zeros(LPC_ORDER, dtype=np.float32)
    
    # Build Toeplitz matrix R (p×p): R[i,j] = r(|i-j|)
    R = np.zeros((LPC_ORDER, LPC_ORDER), dtype=np.float32)
    for i in range(LPC_ORDER):
        for j in range(LPC_ORDER):
            R[i, j] = r[abs(i - j)]
    
    # Vector r_vec = [r(1), r(2), ..., r(p)]
    r_vec = r[1 : LPC_ORDER + 1]
    
    # Solve system Ra = r_vec for LPC coefficients
    try:
        a = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        # Singular matrix - disable TNS
        return X.copy(), np.zeros(LPC_ORDER, dtype=np.float32)
    
    # === 5. Quantization (4-bit, step=0.1) ===
    # Quantization indices: -8 to +7 (16 levels)
    q_indices = np.round(a / QUANT_STEP)
    q_indices = np.clip(q_indices, -8, 7)
    
    # De-quantization
    a_q = q_indices * QUANT_STEP
    
    # === 6. Stability Check ===
    # Filter H_TNS(z) = 1 - a1*z^-1 - a2*z^-2 - a3*z^-3 - a4*z^-4
    # must be stable (all roots inside unit circle)
    filter_poly = np.concatenate(([1.0], -a_q))
    poles = np.roots(filter_poly)
    
    if np.any(np.abs(poles) >= 1.0):
        # Unstable filter - disable TNS
        a_q = np.zeros(LPC_ORDER, dtype=np.float32)
    
    # === 7. Apply TNS FIR Filter to original X(k) ===
    # Y(k) = X(k) - Σ(i=1..p) a_q[i-1] * X(k-i)  (Eq. 8)
    Y = np.zeros(N, dtype=np.float32)
    for k in range(N):
        Y[k] = X[k]
        for i in range(1, LPC_ORDER + 1):
            if k - i >= 0:
                Y[k] -= a_q[i - 1] * X[k - i]
    
    return Y, a_q

