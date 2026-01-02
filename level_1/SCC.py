import numpy as np
from scipy.signal import lfilter

def SSC(frame_T, next_frame_T, prev_frame_type):
    """Sequence Segmentation Control (SSC).

    Decides the type for frame i using information from the next frame (i+1).

    Args:
        frame_T: Current frame samples, stereo. Shape (2048, 2). (Not used in the decision.)
        next_frame_T: Next frame samples, stereo. Shape (2048, 2).
        prev_frame_type: One of "OLS", "LSS", "ESH", "LPS".

    Returns:
        A frame type string: "OLS", "LSS", "ESH", or "LPS".
    """
    
    def detect_attack(samples):
        # 1) High-pass filtering
        #    H(z) = (0.7548 - 0.7548 z^-1) / (1 - 0.5095 z^-1)
        b = [0.7548, -0.7548]
        a = [1.0, -0.5095]
        filtered_signal = lfilter(b, a, samples)
        
        # 2) Divide into 8 segments of 128 samples and compute energy s_l^2
        #    Segments start at offset 448 and end at 1472 (8 × 128 = 1024 samples).
        energies = []
        start_idx = 448
        for l in range(8):
            region = filtered_signal[start_idx + l*128 : start_idx + (l+1)*128]
            energies.append(np.sum(region**2))
            
        # 3) Attack values: ds_l^2 = s_l^2 / mean(previous energies)
        is_attack = False
        for l in range(1, 8):
            mean_prev_energy = np.mean(energies[:l])
            if mean_prev_energy > 0:
                ds_l_sq = energies[l] / mean_prev_energy
            else:
                ds_l_sq = 0
                
            # 4) Decision: mark attack if s_l^2 > 1e-3 and ds_l^2 > 10 (for l = 1..7)
            if energies[l] > 10**-3 and ds_l_sq > 10:
                is_attack = True
                break
        return is_attack

    # Determine individual channel types based on previous frame type
    def get_channel_type(is_attack, prev_type):
        if prev_type == "OLS":
            return "LSS" if is_attack else "OLS"
        elif prev_type == "ESH":
            return "ESH" if is_attack else "LPS"
        elif prev_type == "LSS":
            return "ESH"
        elif prev_type == "LPS":
            return "OLS"
        return "OLS"

    # Process left (0) and right (1) channels
    attack_ch0 = detect_attack(next_frame_T[:, 0])
    attack_ch1 = detect_attack(next_frame_T[:, 1])
    
    type_ch0 = get_channel_type(attack_ch0, prev_frame_type)
    type_ch1 = get_channel_type(attack_ch1, prev_frame_type)

    # 5) Resolve common final type using Table 1 logic
    #    Priority: ESH > (LSS or LPS) > OLS; LSS + LPS -> ESH
    decision_set = {type_ch0, type_ch1}
    
    if "ESH" in decision_set:
        return "ESH"
    elif "LSS" in decision_set:
        if "LPS" in decision_set:
            return "ESH"
        return "LSS"
    elif "LPS" in decision_set:
        return "LPS"
    else:
        return "OLS"