"""
AAC Quantizer - Level 3
Implements the AAC quantization algorithm using psychoacoustic model thresholds.
"""

import numpy as np
import scipy.io as sio
import os


def aac_quantizer(frame_F, frame_type, SMR):
    """
    AAC Quantizer
    
    Quantizes MDCT coefficients using the psychoacoustic model thresholds.
    Implements iterative scale factor refinement to meet the masking thresholds.
    
    Parameters
    ----------
    frame_F : np.ndarray
        MDCT coefficients from the current frame.
        - For OLS/LSS/LPS: shape (1024, 1) or (1024,)
        - For ESH: shape (128, 8)
    
    frame_type : str
        Frame type: "OLS", "LSS", "ESH", or "LPS"
    
    SMR : np.ndarray
        Signal-to-Mask Ratio for each scalefactor band.
        - For OLS/LSS/LPS: shape (NB,) where NB is number of long bands
        - For ESH: shape (NB, 8) where NB is number of short bands
    
    Returns
    -------
    S : np.ndarray
        Quantized MDCT coefficients (integers).
        - For OLS/LSS/LPS: shape (1024, 1)
        - For ESH: shape (128, 8)
    
    sfc : np.ndarray
        Scale factor coefficients (DPCM encoded).
        - For OLS/LSS/LPS: shape (NB, 1)
        - For ESH: shape (NB, 8)
    
    G : np.ndarray or float
        Global gain for the frame.
        - For OLS/LSS/LPS: scalar float
        - For ESH: shape (1, 8) or (8,)
    """
    
    # Constants
    MagicNumber = 0.4054
    MQ = 8191
    
    # Load scalefactor bands
    bands_long, bands_short = load_table_B219()
    
    # Process based on frame type
    if frame_type == "ESH":
        # -------- EIGHT SHORT SEQUENCE --------
        bands = bands_short
        NB = len(bands)  # Number of short bands (typically 42)
        
        # Initialize outputs
        S = np.zeros((128, 8), dtype=int)
        sfc = np.zeros((NB, 8), dtype=int)
        G = np.zeros(8)
        
        # Process each of the 8 subframes
        for sf in range(8):
            # Extract MDCT coefficients for this subframe
            X = frame_F[:, sf]  # 128 coefficients
            SMR_sf = SMR[:, sf]  # SMR for this subframe
            
            # --- 1. Compute energy per band P(b) ---
            P = np.zeros(NB)
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                # Sum of squared MDCT coefficients in this band
                P[b] = np.sum(X[w_low:w_high+1] ** 2)
            
            # --- 2. Compute threshold T(b) = P(b) / SMR(b) ---
            T = P / (SMR_sf + 1e-10)  # Add small epsilon to avoid division by zero
            
            # --- 3. Initial scale factor (same for all bands) ---
            maxX = np.max(np.abs(X))
            if maxX > 0:
                a_init = (16/3) * np.log2((maxX ** (3/4)) / MQ)
                # Limit initial scale factor to prevent overflow
                a_init = np.clip(a_init, -50, 50)
            else:
                a_init = 0
            
            a = np.ones(NB) * a_init  # Scale factor for each band
            
            # --- 4. Iterative refinement per band ---
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                
                max_iterations = 100  # Safety limit
                iteration = 0
                
                while iteration < max_iterations:
                    # Quantize coefficients in this band with current a(b)
                    Pe = 0.0  # Quantization error energy
                    
                    for k in range(w_low, w_high + 1):
                        # Quantization formula
                        if X[k] >= 0:
                            S_temp = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                        else:
                            S_temp = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                        
                        # Inverse quantization (dequantization)
                        if S_temp >= 0:
                            X_hat = (S_temp ** (4/3)) * (2 ** (0.25 * a[b]))
                        else:
                            X_hat = -(np.abs(S_temp) ** (4/3)) * (2 ** (0.25 * a[b]))
                        
                        # Accumulate squared error
                        Pe += (X[k] - X_hat) ** 2
                    
                    # Check if error is below threshold
                    if Pe <= T[b]:
                        break
                    
                    # Increase scale factor
                    a[b] += 1
                    
                    # Safety check: prevent excessive scale factor differences
                    if b > 0 and np.abs(a[b] - a[b-1]) > 60:
                        break
                    
                    iteration += 1
            
            # --- 5. Final quantization with refined scale factors ---
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                
                for k in range(w_low, w_high + 1):
                    if X[k] >= 0:
                        S[k, sf] = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                    else:
                        S[k, sf] = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
            
            # --- 6. Global gain and DPCM scale factors ---
            G[sf] = a[0]
            # Clip to [-16, 16] which is maxAbsCodeVal for codebook 11 with nTupleSize=2
            sfc[0, sf] = int(np.clip(a[0], -16, 16))
            for b in range(1, NB):
                # Compute DPCM and clip to [-16, 16]
                dpcm_val = int(a[b] - a[b-1])
                sfc[b, sf] = int(np.clip(dpcm_val, -16, 16))
        
        return S, sfc, G
    
    else:
        # -------- NON-ESH FRAMES (OLS / LSS / LPS) --------
        bands = bands_long
        NB = len(bands)  # Number of long bands (typically 69)
        
        # Ensure X is 1D
        if frame_F.ndim == 2:
            X = frame_F.flatten()
        else:
            X = frame_F
        
        # Initialize outputs
        S = np.zeros(1024, dtype=int)
        sfc = np.zeros(NB, dtype=int)
        
        # --- 1. Compute energy per band P(b) ---
        P = np.zeros(NB)
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            # Sum of squared MDCT coefficients in this band
            P[b] = np.sum(X[w_low:w_high+1] ** 2)
        
        # --- 2. Compute threshold T(b) = P(b) / SMR(b) ---
        T = P / (SMR + 1e-10)  # Add small epsilon to avoid division by zero
        
        # --- 3. Initial scale factor (same for all bands) ---
        maxX = np.max(np.abs(X))
        if maxX > 0:
            a_init = (16/3) * np.log2((maxX ** (3/4)) / MQ)
            # Limit initial scale factor to prevent overflow
            a_init = np.clip(a_init, -50, 50)
        else:
            a_init = 0
        
        a = np.ones(NB) * a_init  # Scale factor for each band
        
        # --- 4. Iterative refinement per band ---
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            
            max_iterations = 100  # Safety limit
            iteration = 0
            
            while iteration < max_iterations:
                # Quantize coefficients in this band with current a(b)
                Pe = 0.0  # Quantization error energy
                
                for k in range(w_low, w_high + 1):
                    # Quantization formula
                    if X[k] >= 0:
                        S_temp = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                    else:
                        S_temp = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                    
                    # Inverse quantization (dequantization)
                    if S_temp >= 0:
                        X_hat = (S_temp ** (4/3)) * (2 ** (0.25 * a[b]))
                    else:
                        X_hat = -(np.abs(S_temp) ** (4/3)) * (2 ** (0.25 * a[b]))
                    
                    # Accumulate squared error
                    Pe += (X[k] - X_hat) ** 2
                
                # Check if error is below threshold
                # Ensure Pe and T[b] are scalars for comparison
                Pe_scalar = float(Pe)
                T_scalar = float(T[b])
                if Pe_scalar <= T_scalar:
                    break
                
                # Increase scale factor
                a[b] += 1
                
                # Safety check: prevent excessive scale factor differences and absolute values
                if b > 0 and np.abs(a[b] - a[b-1]) > 60:
                    break
                if np.abs(a[b]) > 100:  # Absolute limit to prevent overflow
                    break
                
                iteration += 1
        
        # --- 5. Final quantization with refined scale factors ---
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            
            for k in range(w_low, w_high + 1):
                if X[k] >= 0:
                    S[k] = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                else:
                    S[k] = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
        
        # --- 6. Global gain and DPCM scale factors ---
        G = a[0]
        # Clip to [-16, 16] which is maxAbsCodeVal for codebook 11 with nTupleSize=2
        sfc[0] = int(np.clip(a[0], -16, 16))
        for b in range(1, NB):
            # Compute DPCM and clip to [-16, 16]
            dpcm_val = int(a[b] - a[b-1])
            sfc[b] = int(np.clip(dpcm_val, -16, 16))
        
        # Reshape S to (1024, 1) for consistency
        S = S.reshape(1024, 1)
        sfc = sfc.reshape(NB, 1)
        
        return S, sfc, G


def load_table_B219():
    """
    Load scalefactor band tables from TableB219.mat
    
    Returns
    -------
    bands_long : np.ndarray
        Long window bands (B219a), shape (69, 6)
    bands_short : np.ndarray
        Short window bands (B219b), shape (42, 6)
    
    Notes
    -----
    Each row contains:
        [band_index, w_low, w_high, width, frequency, threshold]
    We use columns 1 and 2 for w_low and w_high.
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mat_filename = os.path.join(current_dir, "TableB219.mat")
    
    # Load the .mat file
    mat_data = sio.loadmat(mat_filename)
    
    bands_long = mat_data['B219a']   # For OLS/LSS/LPS frames
    bands_short = mat_data['B219b']  # For ESH frames
    
    return bands_long, bands_short
