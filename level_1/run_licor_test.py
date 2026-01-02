"""
Test AAC Level 1 with LicorDeCalandraca.wav
Runs the complete encode-decode pipeline on the actual audio file.
"""

from demo_aac_1 import demo_aac_1


if __name__ == "__main__":
    # Input and output files
    input_file = r"C:\Users\panoc\OneDrive\Υπολογιστής\LicorDeCalandraca.wav"
    output_file = r"C:\Users\panoc\AAC-coder-decoder\LicorDeCalandraca_decoded.wav"
    
    # Run the demo
    SNR = demo_aac_1(input_file, output_file)
    
    print("\n" + "="*50)
    print(f"DEMO COMPLETE!")
    print(f"SNR: {SNR:.2f} dB")
    print("="*50)
