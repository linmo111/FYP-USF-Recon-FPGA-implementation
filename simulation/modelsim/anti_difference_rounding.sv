module anti_difference_rounding #(
    parameter WIDTH = 16,        // Data width
    parameter LAMBDA = 10,        // Modulo threshold ?
	 parameter block_delay =2
) (
    input  logic                  clk,          // Clock signal
    input  logic                  reset,        // Reset signal
    input  logic                  valid_in,     // Valid input signal
    input  logic signed [WIDTH-1:0] residual_diff_in, // Residual difference ?^2 ?_y[k]
    output logic                  valid_out,    // Output valid signal
    output logic signed [WIDTH-1:0] residual_out // Recovered ?_y[k]
);

    // Internal registers to store previous values
    logic signed [WIDTH-1:0] first_order_diff;  // Stores ??_y[k]
    logic signed [WIDTH-1:0] residual;         // Stores ?_y[k]
	 logic [block_delay:0] valid_out_shift;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            first_order_diff <= 0;
            residual <= 0;
            valid_out <= 0;
				valid_out_shift <=0;
				residual_out <=0;
        end 
		  
		  
		  else begin
		  		valid_out_shift[0] <= valid_in;
            for (int i = 1; i <= block_delay; i++) begin
                valid_out_shift[i] <= valid_out_shift[i-1];
            end
				valid_out <= valid_out_shift[block_delay];
		  
//			  if (valid_in) begin
					// Step 1: Compute first-order difference (cumulative sum)
					first_order_diff <= first_order_diff + residual_diff_in;

					// Step 2: Compute residual function (cumulative sum)
					residual <= residual + first_order_diff;

					// Step 3: Round to nearest multiple of 2?
//					if (valid_out_shift[block_delay-1]) begin
					residual_out <= (residual) / (2 * LAMBDA) * (2 * LAMBDA);
//					end

					// Output valid signal
////					valid_out <= 1;
//			  end else begin
////					valid_out <= 0;
//			  end
//		  end
		end	
    end

endmodule

