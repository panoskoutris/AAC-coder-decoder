"""
Test AAC Level 2 with LicorDeCalandraca.wav
Runs the complete encode-decode pipeline with TNS on the actual audio file.
"""

from demo_aac_2 import demo_aac_2


if __name__ == "__main__":
    # Input and output files
    material_path = "/Users/chris/Desktop/Multimedia_Systems/project_material/material"
    input_file = f"{material_path}/LicorDeCalandraca.wav"
    output_file = f"{material_path}/LicorDeCalandraca_decoded_level2.wav"
    
    # Run the demo
    SNR = demo_aac_2(input_file, output_file)
    
    print("\n" + "="*50)
    print(f"DEMO COMPLETE!")
    print(f"SNR: {SNR:.2f} dB")
    print("="*50)
