"""
Inverse Temporal Noise Shaping (iTNS) - Level 2

Implements the inverse TNS operation for AAC decoding.
Applies the inverse filter to restore the original MDCT coefficients
from TNS-filtered coefficients using the quantized LPC coefficients
transmitted by the encoder.
"""

import numpy as np


def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Apply inverse Temporal Noise Shaping (TNS) to restore MDCT coefficients.
    
    The inverse TNS applies the inverse filter H^(-1)_TNS to the TNS-filtered
    coefficients, using the quantized LPC coefficients from the encoder.
    
    Forward TNS:  Y(k) = X(k) - Σ(i=1..p) a_q[i-1] * X(k-i)    [FIR]
    Inverse TNS:  X(k) = Y(k) + Σ(i=1..p) a_q[i-1] * X(k-i)    [IIR - recursive]
    
    Parameters
    ----------
    frame_F_in : ndarray
        TNS-filtered MDCT coefficients from decoder.
        Shape:
            - (1024, 1) for OLS, LSS, LPS frames
            - (128, 8) for ESH (8 short subframes)
    
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", "LPS"
    
    tns_coeffs : ndarray
        Quantized LPC coefficients from encoder.
        Shape:
            - (4, 1) for non-ESH frames
            - (4, 8) for ESH frames (one set per subframe)
    
    Returns
    -------
    frame_F_out : ndarray
        Reconstructed MDCT coefficients (before TNS was applied)
        Same shape as frame_F_in
    """
    # Normalize frame type
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":
        # ESH: 8 short subframes, each with 128 MDCT coefficients
        num_subframes = 8
        frame_F_out = np.zeros_like(frame_F_in, dtype=np.float32)
        
        # Process each subframe separately
        for s in range(num_subframes):
            Y = frame_F_in[:, s].astype(np.float32).copy()
            a_q = tns_coeffs[:, s].astype(np.float32)
            
            X = _apply_inverse_tns_filter(Y, a_q)
            frame_F_out[:, s] = X
        
        return frame_F_out
    
    else:
        # Long frames (OLS, LSS, LPS): 1024 MDCT coefficients
        Y = frame_F_in.flatten().astype(np.float32).copy()
        a_q = tns_coeffs.flatten().astype(np.float32)
        
        X = _apply_inverse_tns_filter(Y, a_q)
        
        # Reshape back to column vector (1024, 1)
        frame_F_out = X.reshape(-1, 1)
        
        return frame_F_out


def _apply_inverse_tns_filter(Y, a_q):
    """
    Apply inverse TNS filter to reconstruct original MDCT coefficients.
    
    Implements the recursive (IIR) inverse filter:
        X(k) = Y(k) + Σ(i=1..p) a_q[i-1] * X(k-i)
    
    This is the inverse of the forward TNS FIR filter.
    
    Parameters
    ----------
    Y : ndarray
        TNS-filtered MDCT coefficients (1-D array, length N)
    
    a_q : ndarray
        Quantized LPC coefficients (length 4)
    
    Returns
    -------
    X : ndarray
        Reconstructed MDCT coefficients (same length as Y)
    """
    Y = np.asarray(Y, dtype=np.float32)
    a_q = np.asarray(a_q, dtype=np.float32)
    N = len(Y)
    
    # LPC order (always 4 for AAC TNS)
    LPC_ORDER = 4
    
    # Apply inverse TNS filter (recursive/IIR)
    # X(k) = Y(k) + Σ(i=1..p) a_q[i-1] * X(k-i)
    X = np.zeros(N, dtype=np.float32)
    
    for k in range(N):
        X[k] = Y[k]
        
        # Add feedback from previously computed X values
        for i in range(1, LPC_ORDER + 1):
            if k - i >= 0:
                X[k] += a_q[i - 1] * X[k - i]
    
    return X
