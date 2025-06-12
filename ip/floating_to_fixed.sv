module float_to_fixed #(
    parameter TOTAL_BITS = 32,       // Total output bits (e.g., 16)
    parameter FRACTIONAL_BITS = 20    // Number of fractional bits (e.g., 13)
)(
    input  wire [31:0] float_in,                      // 32-bit IEEE-754 float (unsigned raw bits)
    output reg  signed [TOTAL_BITS-1:0] fixed_out     // Signed fixed-point output
);

// Step 1: Extract fields
wire sign;
wire [7:0] exponent;
wire [22:0] mantissa;

assign sign     = float_in[31];
assign exponent = float_in[30:23];
assign mantissa = float_in[22:0];

// Step 2: Build full mantissa with implicit leading 1
wire [23:0] full_mantissa = {1'b1, mantissa};

// Step 3: Calculate shift amount
wire signed [9:0] shift_amount;  // Signed shift
assign shift_amount = exponent - 127 + (FRACTIONAL_BITS+1);

// Step 4: Shift mantissa accordingly
wire [55:0] shifted_mantissa;
assign shifted_mantissa = (shift_amount >= 0) ? 
    (full_mantissa << shift_amount) : 
    (full_mantissa >> -shift_amount);

// Step 5: Truncate or clip if needed
wire [TOTAL_BITS-1:0] fixed_raw = shifted_mantissa[TOTAL_BITS-1+24:24];

// Step 6: Apply sign properly
always @(*) begin
    if (exponent == 8'b0) begin
        // Special case: float is zero or subnormal
        fixed_out = 0;
    end else begin
        if (sign) begin
            fixed_out = -$signed(fixed_raw);  // Use Verilog signed negation
        end else begin
            fixed_out = $signed(fixed_raw);
        end
    end
end

endmodule
