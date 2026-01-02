"""
Demo for AAC Level 1 Encoder and Decoder
Demonstrates the complete encode-decode pipeline and calculates SNR.
"""

import numpy as np
import soundfile as sf
from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1


def demo_aac_1(filename_in, filename_out):
    """
    Demonstrates Level 1 AAC encoding and decoding.
    
    Takes an input WAV file, encodes it using aac_coder_1, decodes it using
    i_aac_coder_1, writes the decoded audio to an output file, and calculates
    the Signal-to-Noise Ratio (SNR) between the original and decoded audio.
    
    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_out : str
        Path to output WAV file (stereo, 48kHz) for decoded audio
    
    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB between original and decoded audio
    """
    
    print(f"Reading input file: {filename_in}")
    # Read the original audio
    original_audio, sample_rate = sf.read(filename_in, dtype='float32')
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(original_audio)/sample_rate:.2f} seconds")
    print(f"  Channels: {original_audio.shape[1]}")
    
    # Encode the audio
    print("\nEncoding...")
    aac_seq_1 = aac_coder_1(filename_in)
    print(f"  Encoded into {len(aac_seq_1)} frames")
    
    # Decode the audio
    print("\nDecoding...")
    decoded_audio = i_aac_coder_1(aac_seq_1, filename_out)
    print(f"  Decoded {len(decoded_audio)} samples")
    print(f"  Written to: {filename_out}")
    
    # Calculate SNR
    print("\nCalculating SNR...")
    # Ensure both signals have the same length
    min_length = min(len(original_audio), len(decoded_audio))
    original_trimmed = original_audio[:min_length, :]
    decoded_trimmed = decoded_audio[:min_length, :]
    
    # Calculate signal power (original)
    signal_power = np.mean(original_trimmed ** 2)
    
    # Calculate noise power (difference)
    noise = original_trimmed - decoded_trimmed
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')  # Perfect reconstruction
    
    print(f"  SNR: {SNR:.2f} dB")
    
    return SNR
