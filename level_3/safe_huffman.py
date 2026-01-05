"""
Safe Huffman decoder wrapper that prevents out-of-bounds errors.

This module wraps the decode_huff function from huff_utils.py to prevent
IndexError when the decoder tries to read past the end of the bitstream.
"""

import numpy as np
from huff_utils import decode_huff


def safe_decode_huff(stream, codebook_LUT, max_symbols=None):
    """
    Safely decode Huffman stream with protection against out-of-bounds errors.
    
    This function wraps decode_huff() and catches IndexError exceptions that
    occur when the decoder tries to read past the end of the bitstream.
    
    Parameters
    ----------
    stream : str
        Binary string to decode (e.g., "010110...")
    codebook_LUT : dict
        Huffman lookup table from huffCodebooks.mat
    max_symbols : int, optional
        Maximum number of symbols to decode. If None, decode until error or end.
    
    Returns
    -------
    decoded : list
        List of decoded integer values
    """
    
    if not stream or len(stream) == 0:
        return []
    
    # Try decoding with the original function
    try:
        decoded = decode_huff(stream, codebook_LUT)
        
        # Convert to list if needed and ensure integer values
        if not isinstance(decoded, list):
            decoded = list(decoded)
        decoded = [int(x) for x in decoded]
        
        # If max_symbols specified, truncate
        if max_symbols is not None and len(decoded) > max_symbols:
            return decoded[:max_symbols]
        
        return decoded
        
    except (IndexError, KeyError, ValueError) as e:
        # The decoder hit an error
        # Strategy: Try to decode as much as possible by shortening the stream
        
        # Start from full length and work backwards
        for trim in range(0, min(100, len(stream)), 1):
            test_len = len(stream) - trim
            if test_len <= 0:
                break
            
            try:
                decoded = decode_huff(stream[:test_len], codebook_LUT)
                if not isinstance(decoded, list):
                    decoded = list(decoded)
                decoded = [int(x) for x in decoded]
                
                # Successfully decoded something
                if max_symbols is not None and len(decoded) > max_symbols:
                    return decoded[:max_symbols]
                return decoded
                
            except (IndexError, KeyError, ValueError):
                continue
        
        # Last resort: return empty list
        # print(f"Warning: Could not decode any symbols from {len(stream)}-bit stream")
        return []
