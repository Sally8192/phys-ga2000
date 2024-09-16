import numpy as np
def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

for value in [100.98763]:
    bitlist=get_bits(np.float32(value))
    sign = bitlist[0]
    exponent = bitlist[1:9]
    mantissa = bitlist[9:32]
    template = """{value} decimal ->
       sign = {sign} 
       exponent = {exponent} 
       mantissa = {mantissa}"""
    print(template.format(value=value, sign=sign, exponent=exponent, mantissa=mantissa))
import struct

def bits_to_float32(bitlist):
    """Convert a list of bits to a 32-bit float."""
    # Ensure bitlist is 32 bits
    assert len(bitlist) == 32
    # Convert bits to a 32-bit integer
    bitstring = ''.join(str(b) for b in bitlist)
    int_value = int(bitstring, 2)
    # Convert the integer to bytes and unpack as float32
    float_bytes = struct.pack('I', int_value)
    return struct.unpack('f', float_bytes)[0]

# Reconstruct the value from its bit representation
reconstructed_value = bits_to_float32(bitlist)
difference = abs(value - reconstructed_value)

print(f"Reconstructed value from 32-bit representation: {reconstructed_value}")
print(f"Difference between original and 32-bit representation: {difference}")