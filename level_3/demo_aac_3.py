"""
Demo for AAC Level 3 Encoder and Decoder
Demonstrates the complete encode-decode pipeline with psychoacoustic model,
quantization, Huffman coding and calculates SNR, bitrate, and compression ratio.
"""

import numpy as np
import soundfile as sf
import os
from aac_coder_3 import aac_coder_3
from i_aac_coder_3 import i_aac_coder_3


def demo_aac_3(filename_in, filename_out, filename_aac_coded):
    """
    Demonstrates Level 3 AAC encoding and decoding with full compression.
    
    Takes an input WAV file, encodes it using aac_coder_3 (with psychoacoustic
    model, quantization, and Huffman coding), saves the encoded data to a .mat
    file, decodes it using i_aac_coder_3, writes the decoded audio to an output
    file, and calculates the Signal-to-Noise Ratio (SNR), bitrate, and compression
    ratio.
    
    Parameters
    ----------
    filename_in : str
        Path to input WAV file (stereo, 48kHz)
    filename_out : str
        Path to output WAV file (stereo, 48kHz) for decoded audio
    filename_aac_coded : str
        Path to .mat file where encoded AAC data will be stored
    
    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio in dB between original and decoded audio
    bitrate : float
        Bitrate in bits per second (bps)
    compression : float
        Compression ratio (bitrate before encoding / bitrate after encoding)
    """
    
    print(f"Reading input file: {filename_in}")
    # Read the original audio
    original_audio, sample_rate = sf.read(filename_in, dtype='float32')
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(original_audio)/sample_rate:.2f} seconds")
    print(f"  Channels: {original_audio.shape[1]}")
    
    # Calculate original bitrate (uncompressed PCM)
    # 48kHz * 2 channels * 16 bits/sample = 1,536,000 bps
    original_bitrate = sample_rate * original_audio.shape[1] * 16
    print(f"  Original bitrate: {original_bitrate} bps ({original_bitrate/1000:.1f} kbps)")
    
    # Encode the audio
    print("\nEncoding with psychoacoustic model, quantization and Huffman coding...")
    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    print(f"  Encoded into {len(aac_seq_3)} frames")
    print(f"  Saved to: {filename_aac_coded}")
    
    # Calculate actual bitrate from Huffman streams (not .mat file overhead)
    total_bits = 0
    mdct_bits = 0
    sfc_bits = 0
    gain_bits = 0
    for frame in aac_seq_3:
        # Count bits from left channel
        mdct_bits += len(frame["chl"]["stream"])  # Huffman stream for MDCT coefficients
        sfc_bits += len(frame["chl"]["sfc"])      # Huffman stream for scale factors
        # Count bits from right channel  
        mdct_bits += len(frame["chr"]["stream"])
        sfc_bits += len(frame["chr"]["sfc"])
        # Add bits for global gains (assume 8 bits each)
        if isinstance(frame["chl"]["G"], np.ndarray):
            gain_bits += len(frame["chl"]["G"]) * 8  # 8 values for ESH
        else:
            gain_bits += 8  # 1 value for long frames
        if isinstance(frame["chr"]["G"], np.ndarray):
            gain_bits += len(frame["chr"]["G"]) * 8
        else:
            gain_bits += 8
        # Add bits for codebook indices (4 bits each)
        gain_bits += 8  # 4 bits per channel × 2 channels
    
    total_bits = mdct_bits + sfc_bits + gain_bits
    print(f"  Total Huffman bits: {total_bits} bits ({total_bits/8:.2f} bytes)")
    print(f"    MDCT coefficients: {mdct_bits} bits ({100*mdct_bits/total_bits:.1f}%)")
    print(f"    Scale factors:     {sfc_bits} bits ({100*sfc_bits/total_bits:.1f}%)")
    print(f"    Gains/overhead:    {gain_bits} bits ({100*gain_bits/total_bits:.1f}%)")
    
    # Calculate duration in seconds
    duration_seconds = len(original_audio) / sample_rate
    
    # Calculate bitrate from actual Huffman bits
    bitrate = total_bits / duration_seconds
    print(f"  AAC bitrate: {bitrate:.2f} bps ({bitrate/1000:.2f} kbps)")
    
    # Calculate compression ratio
    compression = original_bitrate / bitrate
    print(f"  Compression ratio: {compression:.2f}:1")
    
    # Also show .mat file size for reference
    file_size_bytes = os.path.getsize(filename_aac_coded)
    print(f"\n  .mat file size (with overhead): {file_size_bytes} bytes ({file_size_bytes/1024:.2f} KB)")
    mat_bitrate = (file_size_bytes * 8) / duration_seconds
    print(f"  .mat bitrate (with overhead): {mat_bitrate:.2f} bps ({mat_bitrate/1000:.2f} kbps)")
    
    # Decode the audio
    print("\nDecoding with inverse quantization, inverse TNS, and inverse filterbank...")
    decoded_audio = i_aac_coder_3(aac_seq_3, filename_out)
    print(f"  Decoded {len(decoded_audio)} samples")
    print(f"  Written to: {filename_out}")
    
    # Calculate SNR with proper alignment and artifact removal
    print("\nCalculating SNR...")
    # Ensure both signals have the same length
    min_length = min(len(original_audio), len(decoded_audio))
    original_trimmed = original_audio[:min_length, :]
    decoded_trimmed = decoded_audio[:min_length, :]
    
    # Step 1: Remove edge artifacts (first and last 4096 samples per channel)
    edge_guard = 4096
    if min_length > 2 * edge_guard:
        original_trimmed = original_trimmed[edge_guard:-edge_guard, :]
        decoded_trimmed = decoded_trimmed[edge_guard:-edge_guard, :]
        print(f"  Removed {edge_guard} edge samples from start and end")
    
    # Step 2: Time alignment using cross-correlation (per channel)
    # We'll align each channel independently and take the average lag
    lags = []
    for ch in range(original_trimmed.shape[1]):
        orig_ch = original_trimmed[:, ch]
        dec_ch = decoded_trimmed[:, ch]
        
        # Compute cross-correlation
        correlation = np.correlate(orig_ch, dec_ch, mode='full')
        # Find the lag that maximizes correlation
        lag = np.argmax(correlation) - (len(dec_ch) - 1)
        lags.append(lag)
    
    avg_lag = int(np.mean(lags))
    print(f"  Estimated time alignment lag: {avg_lag} samples")
    
    # Apply lag correction
    if avg_lag > 0:
        # Decoded is delayed, shift it back
        original_aligned = original_trimmed[avg_lag:, :]
        decoded_aligned = decoded_trimmed[:-avg_lag, :]
    elif avg_lag < 0:
        # Decoded is ahead, shift it forward
        original_aligned = original_trimmed[:avg_lag, :]
        decoded_aligned = decoded_trimmed[-avg_lag:, :]
    else:
        # No shift needed
        original_aligned = original_trimmed
        decoded_aligned = decoded_trimmed
    
    # Step 3: Apply optimal gain matching per channel
    # alpha = sum(original * decoded) / sum(decoded**2)
    for ch in range(original_aligned.shape[1]):
        orig_ch = original_aligned[:, ch]
        dec_ch = decoded_aligned[:, ch]
        
        numerator = np.sum(orig_ch * dec_ch)
        denominator = np.sum(dec_ch ** 2)
        
        if denominator > 1e-10:  # Avoid division by zero
            alpha = numerator / denominator
            decoded_aligned[:, ch] = alpha * dec_ch
            print(f"  Channel {ch}: applied gain correction factor {alpha:.6f}")
    
    # Step 4: Calculate SNR on aligned and gain-matched signals
    signal_power = np.mean(original_aligned ** 2)
    noise = original_aligned - decoded_aligned
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')  # Perfect reconstruction
    
    print(f"  SNR (after alignment and gain matching): {SNR:.2f} dB")
    
    return SNR, bitrate, compression
