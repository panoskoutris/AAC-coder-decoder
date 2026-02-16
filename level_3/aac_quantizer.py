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
            T = P / (SMR_sf + 1e-10)
            
            # --- 3. Initial scale factor (same for all bands) ---
            maxX = np.max(np.abs(X))
            if maxX > 0:
                a_init = (16/3) * np.log2((maxX ** (3/4)) / MQ)
            else:
                a_init = 0
            
            # Clamp initial α to reasonable range
            a_init = np.clip(a_init, -20, 20)
            
            a = np.ones(NB) * a_init  # Scale factor for each band
            
            # --- 4. Iterative refinement per band ---
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                
                max_iterations = 20  # Stable value that works with Huffman decoder
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
                        
                        # Clamp to valid range to prevent overflow
                        S_temp = np.clip(S_temp, -MQ, MQ)
                        
                        # Inverse quantization (dequantization)
                        if S_temp >= 0:
                            X_hat = (S_temp ** (4/3)) * (2 ** (0.25 * a[b]))
                        else:
                            X_hat = -(np.abs(S_temp) ** (4/3)) * (2 ** (0.25 * a[b]))
                        
                        # Accumulate squared error
                        Pe += (X[k] - X_hat) ** 2
                    
                    # Check if error exceeds threshold
                    if Pe > T[b]:
                        break
                    
                    # If Pe <= T, we can use coarser quantization (save more bits)
                    # Increase scale factor
                    a[b] += 1
                    
                    # Safety check: prevent excessive scale factor differences
                    if b > 0 and np.abs(a[b] - a[b-1]) > 60:
                        break
                    
                    # Safety check: prevent extreme α values
                    if a[b] > 30:
                        break
                    
                    iteration += 1
            
            # --- COMPRESSION BOOST: Increase all scale factors ---
            # Higher value = more compression, lower SNR
            compression_boost = 8.5  
            a = a + compression_boost
            
            # --- 5. Final quantization with refined scale factors ---
            for b in range(NB):
                w_low = int(bands[b, 1])
                w_high = int(bands[b, 2])
                
                for k in range(w_low, w_high + 1):
                    if X[k] >= 0:
                        S[k, sf] = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                    else:
                        S[k, sf] = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                    # Clamp final S as well
                    S[k, sf] = np.clip(S[k, sf], -MQ, MQ)
            
            # --- 6. Global gain and DPCM scale factors ---
            G[sf] = a[0]
            # sfc[0] is not encoded - decoder uses G for α(0)
            sfc[0, sf] = 0  # Sentinel/dummy value
            for b in range(1, NB):
                diff = int(a[b] - a[b-1])
                # Clamp to codebook 11 range: -60 to +60 (to be safe)
                sfc[b, sf] = np.clip(diff, -60, 60)
        
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
        
        # Ensure SMR is 1D
        if SMR.ndim == 2:
            SMR = SMR.flatten()
        
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
        T = P / (SMR + 1e-10)
        
        # --- 3. Initial scale factor (same for all bands) ---
        maxX = np.max(np.abs(X))
        if maxX > 0:
            a_init = (16/3) * np.log2((maxX ** (3/4)) / MQ)
        else:
            a_init = 0
        
        # Clamp initial α to reasonable range
        a_init = np.clip(a_init, -20, 20)
        
        a = np.ones(NB) * a_init  # Scale factor for each band
        
        # --- 4. Iterative refinement per band ---
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            
            max_iterations = 20  # Stable value that works with Huffman decoder
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
                    
                    # Clamp to valid range to prevent overflow
                    S_temp = np.clip(S_temp, -MQ, MQ)
                    
                    # Inverse quantization (dequantization)
                    if S_temp >= 0:
                        X_hat = (S_temp ** (4/3)) * (2 ** (0.25 * a[b]))
                    else:
                        X_hat = -(np.abs(S_temp) ** (4/3)) * (2 ** (0.25 * a[b]))
                    
                    # Accumulate squared error
                    Pe += (X[k] - X_hat) ** 2
                
                # Check if error exceeds threshold
                if Pe > T[b]:
                    break
                
                # If Pe <= T, we can use coarser quantization (save more bits)
                # Increase scale factor to make quantization coarser
                a[b] += 1
                
                # Safety check: prevent excessive scale factor differences
                if b > 0 and np.abs(a[b] - a[b-1]) > 60:
                    break
                
                # Safety check: prevent extreme α values
                if a[b] > 30:
                    break
                
                iteration += 1
        
        # --- COMPRESSION BOOST: Increase all scale factors ---
        # Higher value = more compression, lower SNR
        compression_boost = 8.5  # Try 2, 3, 4, 5 for different compression levels
        a = a + compression_boost
        
        # --- 5. Final quantization with refined scale factors ---
        for b in range(NB):
            w_low = int(bands[b, 1])
            w_high = int(bands[b, 2])
            
            for k in range(w_low, w_high + 1):
                if X[k] >= 0:
                    S[k] = int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                else:
                    S[k] = -int((np.abs(X[k]) * (2 ** (-0.25 * a[b]))) ** (3/4) + MagicNumber)
                # Clamp final S as well
                S[k] = np.clip(S[k], -MQ, MQ)
                
        # --- 6. Global gain and DPCM scale factors ---
        G = a[0]
        # sfc[0] is not encoded - decoder uses G for α(0)
        sfc[0] = 0  # Sentinel/dummy value
        for b in range(1, NB):
            diff = int(a[b] - a[b-1])
            # Clamp to codebook 11 range: -60 to +60 (to be safe)
            sfc[b] = np.clip(diff, -60, 60)
        
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
