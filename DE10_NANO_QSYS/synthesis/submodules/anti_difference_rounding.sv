module anti_difference_rounding #(
    parameter WIDTH = 24,        // Data width
    parameter LAMBDA = 0.75,        // Modulo threshold λ
	 parameter block_delay =14,
	 parameter FRACTIONAL_BITS =16
) (
    input  logic                  clk,          // Clock signal
	 input logic							clk_en, 
    input  logic                  reset,        // Reset signal
    input  logic                  valid_in,     // Valid input signal
    input  logic signed [WIDTH-1:0] residual_diff_in, // Residual difference Δ^2 ε_y[k]
    output logic                  valid_out,    // Output valid signal
    output logic signed [WIDTH-1:0] residual_out // Recovered ε_y[k]
);

    // Internal registers to store previous values
    logic signed [WIDTH-1:0] first_order_diff;  // Stores Δε_y[k]
    logic signed [WIDTH-1:0] residual;  
//	logic signed [WIDTH-1-8:0]residual_intermediate1,residual_intermediate2	 ;
	 logic [block_delay-1:0] valid_out_shift;
//	 logic signed [WIDTH-1:0] two_lambda_shifted;

	 localparam logic signed [WIDTH-1:0] two_lambda = (LAMBDA*2) ;
	 
	  logic signed [WIDTH-1:0] denom, numer, quotient, remain;
	 
	 anti_diff_div	anti_diff_div_inst (
	.aclr ( reset ),
	.clken ( clk_en ),
	.clock ( clk ),
	.denom ( denom ),
	.numer ( numer ),
	.quotient ( quotient ),
	.remain ( remain )
	);



    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            first_order_diff <= 0;
            residual <= 0;
            valid_out <= 0;
				valid_out_shift <=0;
				residual_out <=0;
				denom <=0;
				numer <=0;
        end 
		  
		  
		  else begin
		  		valid_out_shift[0] <= valid_in;
            for (int i = 1; i <= block_delay-1; i++) begin
                valid_out_shift[i] <= valid_out_shift[i-1];
            end
				valid_out <= valid_out_shift[block_delay-1];
				
				if (clk_en) begin
		  
				  if (valid_in) begin
						// Step 1: Compute first-order difference (cumulative sum)
						residual <= residual + residual_diff_in;
				  end
////			  
//				  if (valid_out_shift[0]) begin
//				  
//						
//						residual_intermediate1 <= residual;
//						end
					if (valid_out_shift[0]) begin
				  
						numer<=residual;
						denom <= (two_lambda );
						end
					
						// Step 3: Round to nearest multiple of 2λ
					if (valid_out_shift[block_delay-1]) begin
					residual_out <= quotient * two_lambda;  // no shifting needed because this shifts back
 
					end
				end

					// Output valid signal
////					valid_out <= 1;
//			  end else begin
////					valid_out <= 0;
//			  end
//		  end
		end	
    end

endmodule
