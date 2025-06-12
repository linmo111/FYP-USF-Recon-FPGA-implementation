module modulo_residual_piped #(
    parameter WIDTH = 32,          // Data width of input/output
    parameter LAMBDA = 0.75,          // Modulo threshold λ
	 parameter block_delay=13,
	 parameter FRACTIONAL_BITS =20
) (
    input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  valid_in,      // Valid input signal
	 input	logic 						clk_en,
    input  logic signed [WIDTH-1:0] diff_in,     // Higher-order difference Δ^N y[k]
    output logic                  valid_out,     // Output valid signal
    output logic signed [WIDTH-1:0] residual_out // Modulo residual Δ^N ε_y[k]
);
localparam pipe_len=block_delay;
    // Internal signal for modulo operation
    logic signed [WIDTH-1:0] modulo_result;
	 logic [block_delay-1:0] valid_out_shift;
	 logic  signed [WIDTH-1:0] diff_in_shift [0:pipe_len-1];
//	 parameter logic[WIDTH-1:0] lambda_shifted, two_lambda_shifted;
	 
//	 reg signed [WIDTH-1:0] residual_intermediate1,residual_intermediate2,residual_intermediate3;
//	 reg signed [WIDTH-1-8:0] residual_intermediate4;
	 localparam logic signed [WIDTH-1:0] lambda_shifted = LAMBDA ;
//	 assign two_lambda_shifted= 1;
	 localparam logic signed [WIDTH-1:0] two_lambda_shifted= (LAMBDA )*2;
	 
	 
    // Centered modulo operation
	 
	 logic signed [WIDTH-1:0] mod_denom, mod_numer, mod_quotient, mod_remain;
	 
mod_divide	mod_divide_inst (
	.clock ( clk ),
	.clken (clk_en),
	.aclr(reset),
	.denom ( mod_denom ),
	.numer ( mod_numer ),
	.quotient ( mod_quotient ),
	.remain ( mod_remain )
	);

	 
	 
	 
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            modulo_result <= 0;
            residual_out <= 0;
            valid_out <= 0;
				valid_out_shift<=0;
//				residual_intermediate1<=0;
				mod_denom<=0;
				mod_numer<=0;
        end 
		  else begin

					
					
				if (clk_en) begin
					valid_out_shift[0] <= valid_in;
					
					for (int i = 1; i <= block_delay-1; i++) begin
						 valid_out_shift[i] <= valid_out_shift[i-1];
					end
			
					valid_out <= valid_out_shift[block_delay-1];
					
					diff_in_shift[0] <= diff_in;
					
					for (int i = 1; i <= pipe_len-1; i++) begin
						 diff_in_shift[i] <= diff_in_shift[i-1];
					end
			
					
					
						

			
				if (valid_in) begin
//					residual_intermediate1 <= (diff_in+lambda_shifted);
				
					mod_numer <= diff_in + lambda_shifted;
					mod_denom <= two_lambda_shifted;
					
					
				end
				if (valid_out_shift[block_delay-1]) begin
				
					residual_out<=mod_remain- lambda_shifted-diff_in_shift[pipe_len-1];
				end


//            end
				end

		  end
    end

endmodule
