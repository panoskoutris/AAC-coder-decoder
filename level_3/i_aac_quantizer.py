"""
AAC Inverse Quantizer - Level 3
Implements the inverse AAC quantization (dequantization) algorithm.
"""

import numpy as np
from aac_quantizer import load_table_B219


def i_aac_quantizer(S, sfc, G, frame_type):
    """
    AAC Inverse Quantizer (Dequantizer)
    
    Reconstructs MDCT coefficients from quantized values using scale factors.
    
    Parameters
    ----------
    S : np.ndarray
        Quantized MDCT coefficients (integers).
        - For OLS/LSS/LPS: shape (1024, 1) or (1024,)
        - For ESH: shape (128, 8)
    
    sfc : np.ndarray
        Scale factor coefficients (DPCM encoded).
        - For OLS/LSS/LPS: shape (NB, 1) or (NB,)
        - For ESH: shape (NB, 8)
    
    G : float or np.ndarray
        Global gain for the frame.
        - For OLS/LSS/LPS: scalar float
        - For ESH: array of shape (8,) or (1, 8)
    
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    
    Returns
    -------
    frame_F : np.ndarray
        Reconstructed MDCT coefficients.
        - For OLS/LSS/LPS: shape (1024, 1)
        - For ESH: shape (128, 8)
    """
    
    # Load scalefactor bands
    bands_long, bands_short = load_table_B219()
    
    # Process based on frame type
    if frame_type == "ESH":
        # -------- EIGHT SHORT SEQUENCE --------
        bands = bands_short
        NB = len(bands)  # Number of short bands (typically 42)
        
        # Initialize output
        frame_F = np.zeros((128, 8))
        
        # Process each of the 8 subframes
        for sf in range(8):
            # Extract quantized coefficients for this subframe
            S_sf = S[:, sf]  # 128 quantized coefficients
            sfc_sf = sfc[:, sf]  # Scale factors for this subframe
            G_sf = G[sf] if isinstance(G, np.ndarray) else G
            
            # --- 1. Reconstruct actual scale factors from DPCM ---
            a = np.zeros(NB)
            a[0] = G_sf  # First scale factor is the global gain
            for b in range(1, NB):
                a[b] = a[b-1] + sfc_sf[b]
            
            # --- 2. Inverse quantization ---
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                
                for k in range(w_low, w_high + 1):
                    # Dequantization formula: X_hat = sign(S) * |S|^(4/3) * 2^(0.25*a(b))
                    if S_sf[k] >= 0:
                        frame_F[k, sf] = (S_sf[k] ** (4/3)) * (2 ** (0.25 * a[b]))
                    else:
                        frame_F[k, sf] = -(np.abs(S_sf[k]) ** (4/3)) * (2 ** (0.25 * a[b]))
        
        return frame_F
    
    else:
        # -------- NON-ESH FRAMES (OLS / LSS / LPS) --------
        bands = bands_long
        NB = len(bands)  # Number of long bands (typically 69)
        
        # Ensure S is 1D
        if S.ndim == 2:
            S_flat = S.flatten()
        else:
            S_flat = S
        
        # Ensure sfc is 1D
        if sfc.ndim == 2:
            sfc_flat = sfc.flatten()
        else:
            sfc_flat = sfc
        
        # Initialize output
        frame_F = np.zeros(1024)
        
        # --- 1. Reconstruct actual scale factors from DPCM ---
        a = np.zeros(NB)
        a[0] = G  # First scale factor is the global gain
        for b in range(1, NB):
            a[b] = a[b-1] + sfc_flat[b]
        
        # --- 2. Inverse quantization ---
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            
            for k in range(w_low, w_high + 1):
                # Dequantization formula: X_hat = sign(S) * |S|^(4/3) * 2^(0.25*a(b))
                if S_flat[k] >= 0:
                    frame_F[k] = (S_flat[k] ** (4/3)) * (2 ** (0.25 * a[b]))
                else:
                    frame_F[k] = -(np.abs(S_flat[k]) ** (4/3)) * (2 ** (0.25 * a[b]))
        
        # Reshape to (1024, 1) for consistency
        frame_F = frame_F.reshape(1024, 1)
        
        return frame_F
