module fixed_to_float #(
    parameter TOTAL_BITS = 16,       // e.g., 16-bit fixed-point input
    parameter FRACTIONAL_BITS = 10   // e.g., 10 fractional bits = Q5.10
)(
    input  wire signed [TOTAL_BITS-1:0] fixed_in,  // Signed fixed-point input
    output reg signed  [31:0] float_out                   // IEEE-754 32-bit float
);

// Step 1: Sign and absolute value
wire sign = fixed_in[TOTAL_BITS-1];
wire signed [TOTAL_BITS-1:0] abs_fixed = (sign) ? -fixed_in : fixed_in;

// Step 2: Normalize to find leading '1'
integer i;
reg [7:0] leading_one_pos;
always @(*) begin
    leading_one_pos = 0;
    for (i = TOTAL_BITS-2; i >= 0; i = i - 1) begin
        if (abs_fixed[i]) begin
            leading_one_pos = i;
            break;  // exit after first 1
        end
    end
end

// Step 3: Build exponent
// Float exponent = (bit position of first 1) - fractional offset + bias (127)
wire [7:0] exponent = (abs_fixed == 0) ? 8'd0 : (leading_one_pos - FRACTIONAL_BITS + 127);

// Step 4: Normalize mantissa
wire [47:0] shifted_mantissa = abs_fixed << (23 - leading_one_pos);
wire [22:0] mantissa = shifted_mantissa[22:0];  // 23 bits after leading 1

// Step 5: Assemble IEEE float
always @(*) begin
    if (abs_fixed == 0) begin
        float_out = 32'b0;
    end else begin
        float_out = {sign, exponent, mantissa};
    end
end

endmodule