# AAC Encoder/Decoder in Python (Simplified AAC Pipeline)

Simplified AAC (Advanced Audio Coding) encoder/decoder implemented in Python.
The project is structured in **three incremental levels** that progressively add core AAC building blocks.

## What’s implemented

### Level 1 — Framing + SSC + Filterbank
- Framing: **2048 samples/frame** with **50% overlap** (hop = 1024)
- **SSC (Sequence Segmentation Control)**: selects AAC frame types `OLS`, `LSS`, `ESH`, `LPS` via attack detection
- **MDCT analysis filterbank** (SIN windows) + **IMDCT synthesis** + overlap-add reconstruction

### Level 2 — Temporal Noise Shaping (TNS)
- Adds **TNS** using **4th‑order LPC** on MDCT coefficients (per channel; per subframe for `ESH`)
- Uses scalefactor-band tables from `TableB219.mat`
- Includes inverse TNS (`i_tns`) in the decoder

### Level 3 — Psychoacoustic model + Quantization + Huffman coding
- **Psychoacoustic model** computes SMR per scalefactor band (long: 69 bands, short: 42 bands)
- **AAC quantizer** with iterative scalefactor refinement guided by SMR
- **Huffman entropy coding** using `huffCodebooks.mat`
- Full decode path: iHuffman → iQuantizer → iTNS → iFilterbank

Note: **Bit reservoir is intentionally not implemented**, so the bitrate is variable/uncontrolled.

## Repository layout

- `level_1/`: SSC + MDCT/IMDCT + encoder/decoder + demo
- `level_2/`: Level 1 + TNS (`tns`, `i_tns`) + encoder/decoder + demo
- `level_3/`: Level 2 + psychoacoustic model + quantizer + Huffman utils + encoder/decoder + demo + plotting helpers

Each level folder is **self-contained** (it includes its own copies of required modules).

## Requirements

- Python 3.x
- `numpy`, `scipy`, `soundfile`
- Optional: `matplotlib` (only needed for plotting scripts)

Install dependencies:

```bash
pip install numpy scipy soundfile
# optional
pip install matplotlib
```

If you’re on Linux, you may also need a system `libsndfile` package for `soundfile`.

## Input audio requirements

- WAV, **stereo (2 channels)**
- **48 kHz** sampling rate

The code prints a warning if the sample rate is not 48 kHz, but it does **not** resample.

## How to run

The `demo_aac_*` modules expose functions you can call from Python.
There are also `run_licor_test.py` convenience scripts in each level folder, but they contain machine-specific absolute paths — edit them to point to your input WAV.

### Level 1

```bash
cd level_1
python -c "from demo_aac_1 import demo_aac_1; demo_aac_1(r'PATH_TO_INPUT.wav', r'PATH_TO_OUTPUT.wav')"
```

### Level 2

```bash
cd level_2
python -c "from demo_aac_2 import demo_aac_2; demo_aac_2(r'PATH_TO_INPUT.wav', r'PATH_TO_OUTPUT.wav')"
```

### Level 3

```bash
cd level_3
python -c "from demo_aac_3 import demo_aac_3; demo_aac_3(r'PATH_TO_INPUT.wav', r'PATH_TO_OUTPUT.wav', r'PATH_TO_AAC_CODED.mat')"
```

## Results

Evaluation was performed on `LicorDeCalandraca.wav` (stereo, 48 kHz).

- **Level 1**: SNR = **33.28 dB**
- **Level 2**: SNR = **33.28 dB**
	- (TNS + inverse TNS is lossless when there is no quantization stage)
- **Level 3** (full compression):
	- Original bitrate: **1536 kbps** (PCM stereo, 48 kHz, 16-bit)
	- AAC bitrate: **349.0 kbps** (counted from Huffman bitstreams)
	- Compression ratio: **4.40:1**
	- SNR: **25.65 dB**
	- Encoded frames: **275**
	- Total Huffman bits: **2,057,447 bits** (257.2 KB)

An additional experiment (“compression boost”) reports **6.20:1** compression with **SNR = 18.78 dB**, while the decoded audio was still subjectively acceptable.

## Notes / limitations

- Not a production AAC bitstream/mux format (this is a simplified pipeline)
- No bit reservoir (variable bitrate)
- No resampling: provide 48 kHz stereo input
- MDCT/IMDCT are implemented directly and can be slow for long audio

## Authors

- Χρήστος Αλεξόπουλος
- Παναγιώτης Κούτρης