import numpy as np
import scipy.io as sio
import os

# ------------------ LOAD LUT ------------------

def load_LUT(mat_filename=None):
    """
    Loads the list of Huffman Codebooks (LUTs)

    Returns:
        huffLUT : list (index 1..11 used, index 0 unused)
    """
    if mat_filename is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mat_filename = os.path.join(current_dir, "huffCodebooks.mat")

    mat = sio.loadmat(mat_filename)


    huffCodebooks_raw = mat['huffCodebooks'].squeeze()

    huffCodebooks = []
    for i in range(11):
        huffCodebooks.append(np.array(huffCodebooks_raw[i]))

    # Build inverse VLC tables
    invTable = [None] * 11

    for i in range(11):
        h = huffCodebooks[i][:, 2].astype(int)       # column 3
        hlength = huffCodebooks[i][:, 1].astype(int) # column 2

        hbin = []
        for j in range(len(h)):
            hbin.append(format(h[j], f'0{hlength[j]}b'))

        invTable[i] = vlc_table(hbin)

    # Build Huffman LUT dicts
    huffLUT = [None] * 12  # index 0 unused
    params = [
        (4, 1, True),
        (4, 1, True),
        (4, 2, False),
        (4, 2, False),
        (2, 4, True),
        (2, 4, True),
        (2, 7, False),
        (2, 7, False),
        (2, 12, False),
        (2, 12, False),
        (2, 16, False),
    ]

    for i, (nTupleSize, maxAbs, signed) in enumerate(params, start=1):
        huffLUT[i] = {
            'LUT': huffCodebooks[i-1],
            'invTable': invTable[i-1],
            'codebook': i,
            'nTupleSize': nTupleSize,
            'maxAbsCodeVal': maxAbs,
            'signedValues': signed
        }

    return huffLUT

def vlc_table(code_array):
    """
    codeArray: list of strings, each string is a Huffman codeword (e.g. '0101')
    returns:
        h : NumPy array of shape (num_nodes, 3)
            columns:
              [ next_if_0 , next_if_1 , symbol_index ]
    """
    h = np.zeros((1, 3), dtype=int)

    for code_index, code in enumerate(code_array, start=1):
        word = [int(bit) for bit in code]
        h_index = 0

        for bit in word:
            k = bit
            next_node = h[h_index, k]
            if next_node == 0:
                h = np.vstack([h, [0, 0, 0]])
                new_index = h.shape[0] - 1
                h[h_index, k] = new_index
                h_index = new_index
            else:
                h_index = next_node

        h[h_index, 2] = code_index

    return h

# ------------------ ENCODE ------------------

def encode_huff(coeff_sec, huff_LUT_list, force_codebook = None):
    """
    Huffman-encode a sequence of quantized coefficients.

    This function selects the appropriate Huffman codebook based on the
    maximum absolute value of the input coefficients, encodes the coefficients
    into a binary Huffman bitstream, and returns both the bitstream and the
    selected codebook index.

    This is the Python equivalent of the MATLAB `encodeHuff.m` function used
    in audio/image coding (e.g., scale factor band encoding). The input
    coefficient sequence is grouped into fixed-size tuples as defined by
    the chosen Huffman LUT. Zero-padding may be applied internally.

    Parameters
    ----------
    coeff_sec : array_like of int
        1-D array of quantized integer coefficients to encode.
        Typically corresponds to a "section" or scale-factor band.

    huff_LUT_list : list
        List of Huffman lookup-table dictionaries as returned by `loadLUT()`.
        Index 1..11 correspond to valid Huffman codebooks.
        Index 0 is unused.

    Returns
    -------
    huffSec : str
        Huffman-encoded bitstream represented as a string of '0' and '1'
        characters.

    huffCodebook : int
        Index (1..11) of the Huffman codebook used for encoding.
        A value of 0 indicates a special all-zero section.
    """
    if force_codebook is not None:
            return huff_LUT_code_1(huff_LUT_list[force_codebook], coeff_sec), force_codebook
    
    maxAbsVal = np.max(np.abs(coeff_sec))

    if maxAbsVal == 0:
        huffCodebook = 0
        huffSec = huff_LUT_code_0()

    elif maxAbsVal == 1:
        candidates = [1, 2]
        huffSec1 = huff_LUT_code_1(huff_LUT_list[candidates[0]], coeff_sec)
        huffSec2 = huff_LUT_code_1(huff_LUT_list[candidates[1]], coeff_sec)
        if len(huffSec1) <= len(huffSec2):
            huffSec = huffSec1
            huffCodebook = candidates[0]
        else:
            huffSec = huffSec2
            huffCodebook = candidates[1]

    elif maxAbsVal == 2:
        candidates = [3, 4]
        huffSec1 = huff_LUT_code_1(huff_LUT_list[candidates[0]], coeff_sec)
        huffSec2 = huff_LUT_code_1(huff_LUT_list[candidates[1]], coeff_sec)
        if len(huffSec1) <= len(huffSec2):
            huffSec = huffSec1
            huffCodebook = candidates[0]
        else:
            huffSec = huffSec2
            huffCodebook = candidates[1]

    elif maxAbsVal in (3, 4):
        candidates = [5, 6]
        huffSec1 = huff_LUT_code_1(huff_LUT_list[candidates[0]], coeff_sec)
        huffSec2 = huff_LUT_code_1(huff_LUT_list[candidates[1]], coeff_sec)
        if len(huffSec1) <= len(huffSec2):
            huffSec = huffSec1
            huffCodebook = candidates[0]
        else:
            huffSec = huffSec2
            huffCodebook = candidates[1]

    elif maxAbsVal in (5, 6, 7):
        candidates = [7, 8]
        huffSec1 = huff_LUT_code_1(huff_LUT_list[candidates[0]], coeff_sec)
        huffSec2 = huff_LUT_code_1(huff_LUT_list[candidates[1]], coeff_sec)
        if len(huffSec1) <= len(huffSec2):
            huffSec = huffSec1
            huffCodebook = candidates[0]
        else:
            huffSec = huffSec2
            huffCodebook = candidates[1]

    elif maxAbsVal in (8, 9, 10, 11, 12):
        candidates = [9, 10]
        huffSec1 = huff_LUT_code_1(huff_LUT_list[candidates[0]], coeff_sec)
        huffSec2 = huff_LUT_code_1(huff_LUT_list[candidates[1]], coeff_sec)
        if len(huffSec1) <= len(huffSec2):
            huffSec = huffSec1
            huffCodebook = candidates[0]
        else:
            huffSec = huffSec2
            huffCodebook = candidates[1]

    elif maxAbsVal in (13, 14, 15):
        huffCodebook = 11
        huffSec = huff_LUT_code_1(huff_LUT_list[huffCodebook], coeff_sec)

    else:
        huffCodebook = 11
        huffSec = huff_LUT_code_ESC(huff_LUT_list[huffCodebook], coeff_sec)

    return huffSec, huffCodebook

def huff_LUT_code_1(huff_LUT, coeff_sec):
    LUT = huff_LUT['LUT']
    nTupleSize = huff_LUT['nTupleSize']
    maxAbsCodeVal = huff_LUT['maxAbsCodeVal']
    signedValues = huff_LUT['signedValues']

    numTuples = int(np.ceil(len(coeff_sec) / nTupleSize))

    if signedValues:
        coeff = coeff_sec + maxAbsCodeVal
        base = 2 * maxAbsCodeVal + 1
    else:
        coeff = coeff_sec
        base = maxAbsCodeVal + 1

    coeffPad = np.zeros(numTuples * nTupleSize, dtype=int)
    coeffPad[:len(coeff)] = coeff

    huffSec = []

    powers = base ** np.arange(nTupleSize - 1, -1, -1)

    for i in range(numTuples):
        nTuple = coeffPad[i*nTupleSize:(i+1)*nTupleSize]
        huffIndex = int(np.abs(nTuple) @ powers)

        hexVal = LUT[huffIndex, 2]
        huffLen = LUT[huffIndex, 1]

        bits = format(int(hexVal), f'0{int(huffLen)}b')

        if signedValues:
            huffSec.append(bits)
        else:
            signBits = ''.join('1' if v < 0 else '0' for v in nTuple)
            huffSec.append(bits + signBits)

    return ''.join(huffSec)

def huff_LUT_code_0():
    return ''

def huff_LUT_code_ESC(huff_LUT, coeff_sec):
    LUT = huff_LUT['LUT']
    nTupleSize = huff_LUT['nTupleSize']
    maxAbsCodeVal = huff_LUT['maxAbsCodeVal']

    numTuples = int(np.ceil(len(coeff_sec) / nTupleSize))
    base = maxAbsCodeVal + 1

    coeffPad = np.zeros(numTuples * nTupleSize, dtype=int)
    coeffPad[:len(coeff_sec)] = coeff_sec

    huffSec = []
    powers = base ** np.arange(nTupleSize - 1, -1, -1)

    for i in range(numTuples):
        nTuple = coeffPad[i*nTupleSize:(i+1)*nTupleSize]

        lnTuple = nTuple.astype(float)
        lnTuple[lnTuple == 0] = np.finfo(float).eps

        N4 = np.maximum(0, np.floor(np.log2(np.abs(lnTuple))).astype(int))
        N = np.maximum(0, N4 - 4)
        esc = np.abs(nTuple) > 15

        nTupleESC = nTuple.copy()
        nTupleESC[esc] = np.sign(nTupleESC[esc]) * 16

        huffIndex = int(np.abs(nTupleESC) @ powers)

        hexVal = LUT[huffIndex, 2]
        huffLen = LUT[huffIndex, 1]

        bits = format(int(hexVal), f'0{int(huffLen)}b')

        escSeq = ''
        for k in range(nTupleSize):
            if esc[k]:
                escSeq += '1' * N[k]
                escSeq += '0'
                escSeq += format(abs(nTuple[k]) - (1 << N4[k]), f'0{N4[k]}b')

        signBits = ''.join('1' if v < 0 else '0' for v in nTuple)
        huffSec.append(bits + signBits + escSeq)

    return ''.join(huffSec)

# ------------------ DECODE ------------------

def decode_huff(huff_sec, huff_LUT):
    """
    Decode a Huffman-encoded stream.

    Parameters
    ----------
    huff_sec : array-like of int or str
        Huffman encoded stream as a sequence of 0 and 1 (string or list/array).
    huff_LUT : dict
        Huffman lookup table with keys:
            - 'invTable': inverse table (numpy array)
            - 'codebook': codebook number
            - 'nTupleSize': tuple size
            - 'maxAbsCodeVal': maximum absolute code value
            - 'signedValues': True/False

    Returns
    -------
    decCoeffs : list of int
        Decoded quantized coefficients.
    """

    h = huff_LUT['invTable']
    huffCodebook = huff_LUT['codebook']
    nTupleSize = huff_LUT['nTupleSize']
    maxAbsCodeVal = huff_LUT['maxAbsCodeVal']
    signedValues = huff_LUT['signedValues']

    # Convert string to array of ints
    if isinstance(huff_sec, str):
        huff_sec = np.array([int(b) for b in huff_sec])

    eos = False
    decCoeffs = []
    streamIndex = 0

    while not eos:
        wordbit = 0
        r = 0  # start at root

        # Decode Huffman word using inverse table
        while True:
            b = huff_sec[streamIndex + wordbit]
            wordbit += 1
            rOld = r
            r = h[rOld, b]
            if h[r, 0] == 0 and h[r, 1] == 0:
                symbolIndex = h[r, 2] - 1  # zero-based
                streamIndex += wordbit
                break

        # Decode n-tuple magnitudes
        if signedValues:
            base = 2 * maxAbsCodeVal + 1
            nTupleDec = []
            tmp = symbolIndex
            for p in reversed(range(nTupleSize)):
                val = tmp // (base ** p)
                nTupleDec.append(val - maxAbsCodeVal)
                tmp = tmp % (base ** p)
            nTupleDec = np.array(nTupleDec)
        else:
            base = maxAbsCodeVal + 1
            nTupleDec = []
            tmp = symbolIndex
            for p in reversed(range(nTupleSize)):
                val = tmp // (base ** p)
                nTupleDec.append(val)
                tmp = tmp % (base ** p)
            nTupleDec = np.array(nTupleDec)

            # Apply sign bits
            nTupleSignBits = huff_sec[streamIndex:streamIndex + nTupleSize]
            nTupleSign = -(np.sign(nTupleSignBits - 0.5))
            streamIndex += nTupleSize
            nTupleDec = nTupleDec * nTupleSign

        # Handle escape sequences
        escIndex = np.where(np.abs(nTupleDec) == 16)[0]
        if huffCodebook == 11 and escIndex.size > 0:
            for idx in escIndex:
                N = 0
                b = huff_sec[streamIndex]
                while b:
                    N += 1
                    b = huff_sec[streamIndex + N]
                streamIndex += (N + 1)
                N4 = N + 4
                escape_word = huff_sec[streamIndex:streamIndex + N4]
                escape_value = 2 ** N4 + int("".join(map(str, escape_word)), 2)
                nTupleDec[idx] = escape_value
                streamIndex += N4
            # Apply signs again
            nTupleDec[escIndex] *= nTupleSign[escIndex]

        decCoeffs.extend(nTupleDec.tolist())

        if streamIndex >= len(huff_sec):
            eos = True

    return decCoeffs
    
