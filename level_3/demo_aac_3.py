"""
Demo for AAC Level 3 Encoder and Decoder
Demonstrates the complete encode-decode pipeline with quantization, 
psychoacoustic model, and Huffman coding. Calculates SNR, bitrate, and compression.
"""

import numpy as np
import soundfile as sf
import scipy.io as sio
from aac_coder_3 import aac_coder_3
from i_aac_coder_3 import i_aac_coder_3


def demo_aac_3(filename_in, filename_out, filename_aac_coded):
    """
    Demonstrates Level 3 AAC encoding and decoding with full compression pipeline.
    
    Takes an input WAV file, encodes it using aac_coder_3 (with quantization,
    psychoacoustic model, and Huffman coding), saves the encoded data to a .mat file,
    decodes it using i_aac_coder_3, writes the decoded audio to an output file, and 
    calculates the Signal-to-Noise Ratio (SNR), bitrate, and compression ratio.
    
    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_out : str
        Path to output WAV file (stereo, 48kHz) for decoded audio
    filename_aac_coded : str
        Path to .mat file where encoded aac_seq_3 will be saved
    
    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB between original and decoded audio
    bitrate : float
        Bitrate in bits per second
    compression : float
        Compression ratio (original size / compressed size)
    """
    
    print(f"Reading input file: {filename_in}")
    # Read the original audio
    original_audio, sample_rate = sf.read(filename_in, dtype='float32')
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(original_audio)/sample_rate:.2f} seconds")
    print(f"  Channels: {original_audio.shape[1]}")
    
    # Encode the audio
    print("\nEncoding with quantization, psychoacoustic model, and Huffman coding...")
    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    print(f"  Encoded into {len(aac_seq_3)} frames")
    print(f"  Saved encoded data to: {filename_aac_coded}")
    
    # Decode the audio - USE the aac_seq_3 from the encoder directly
    # (optional) you can load from file for verification, but DON'T use it for decoding
    print("\nDecoding...")
    decoded_audio = i_aac_coder_3(aac_seq_3, filename_out)
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
    
    # Calculate bitrate
    print("\nCalculating bitrate...")
    # Count total bits in Huffman bitstreams
    total_bits = 0
    for frame_data in aac_seq_3:
        # Left channel
        total_bits += len(frame_data["chl"]["stream"])  # MDCT coefficients bitstream
        total_bits += len(frame_data["chl"]["sfc"])      # Scale factors bitstream
        
        # Right channel
        total_bits += len(frame_data["chr"]["stream"])
        total_bits += len(frame_data["chr"]["sfc"])
    
    # Calculate duration
    duration = len(original_audio) / sample_rate
    
    # Bitrate in bits per second
    bitrate = total_bits / duration
    
    print(f"  Total bits: {total_bits}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Bitrate: {bitrate:.2f} bits/sec ({bitrate/1000:.2f} kbits/sec)")
    
    # Calculate compression ratio
    print("\nCalculating compression ratio...")
    # Original size: 16-bit PCM, stereo, sample_rate Hz
    original_bits = len(original_audio) * 2 * 16  # samples × channels × bits_per_sample
    
    # Compression ratio
    compression = original_bits / total_bits
    
    print(f"  Original bits: {original_bits}")
    print(f"  Compressed bits: {total_bits}")
    print(f"  Compression ratio: {compression:.2f}:1")
    
    return SNR, bitrate, compression
