"""
Test AAC Level 3 with LicorDeCalandraca.wav
Runs the complete encode-decode pipeline with psychoacoustic model,
quantization and Huffman coding on the actual audio file.
"""

from demo_aac_3 import demo_aac_3


if __name__ == "__main__":
    # Input and output files
    material_path = "/Users/chris/Desktop/Multimedia_Systems/project_material/material"
    input_file = f"{material_path}/LicorDeCalandraca.wav"
    output_file = f"{material_path}/LicorDeCalandraca_decoded_level3.wav"
    aac_coded_file = f"{material_path}/LicorDeCalandraca_aac_coded.mat"
    
    # Run the demo
    SNR, bitrate, compression = demo_aac_3(input_file, output_file, aac_coded_file)
    
    print("\n" + "="*60)
    print(f"LEVEL 3 DEMO COMPLETE!")
    print("="*60)
    print(f"SNR:              {SNR:.2f} dB")
    print(f"Bitrate:          {bitrate:.2f} bps ({bitrate/1000:.2f} kbps)")
    print(f"Compression:      {compression:.2f}:1")
    print("="*60)
