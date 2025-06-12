def float_to_fixed(input_value: float, total_width: int, frac_bits: int) -> int:
    """
    Convert a floating-point number to a signed fixed-point representation.

    Parameters:
    - input_value (float): The input floating-point number.
    - total_width (int): Total bit width (including sign bit).
    - frac_bits (int): Number of fractional bits.

    Returns:
    - int: Signed fixed-point representation.
    """
    # Calculate scaling factor
    scale_factor = 2 ** frac_bits

    # Calculate maximum and minimum values based on bit width
    max_value = (2 ** (total_width - 1)) - 1
    min_value = -(2 ** (total_width - 1))

    # Apply scaling and rounding
    fixed_value = round(input_value * scale_factor)

    # Clip to the range
    if fixed_value > max_value:
        fixed_value = max_value
    elif fixed_value < min_value:
        fixed_value = min_value

    # Convert to signed integer
    return fixed_value

# Example Usage
if __name__ == "__main__":

    input_value = 1/13
    total_width = 24
    frac_bits = 16
    fixed_point = float_to_fixed(input_value, total_width, frac_bits)
    print(f"Input: {input_value}, Fixed-Point: {fixed_point:#x}")

#64000                                      01100100000000000000