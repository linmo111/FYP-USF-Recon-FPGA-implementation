module higher_order_difference #(
    parameter N = 1,             // Order of difference
    parameter WIDTH = 32,         // Data width of input/output
	 parameter block_delay = 1
	 
) (
    input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  valid_in,      // Valid input signal
	 input 	logic 					clk_en,
    input  logic signed [WIDTH-1:0]      y_in,          // Input signal y[k]
    output logic                  valid_out,     // Output valid signal
    output logic signed [WIDTH-1:0]      diff_out       // N-th order difference output
);

    // Internal registers for shift registers
	 logic [WIDTH-1:0] prev_in; 
//    logic [WIDTH-1:0] shift_reg [0:N];  // Stores N+1 samples for recursive differences
//    logic [WIDTH-1:0] differences [0:N-1];  // Temporary storage for differences
//    logic [3:0]       stage_counter;  // Tracks processing stage
//    logic [WIDTH-1:0] pref_1st_diff;
	 logic [block_delay-1:0] valid_out_shift ;
    // Initialize registers
    always_ff @(posedge clk or posedge reset) begin
		
        if (reset) begin
//            for (int i = 0; i <= N; i++) begin
//                shift_reg[i] <= 0;
//            end
//				differences[0]=0
//				differences[1]=0;
//            stage_counter <= 0;
            valid_out <= 0;
//				valid_out_shift <=0;
//				pref_1st_diff <=0;

				diff_out<=0;
				prev_in<=0;
        end
		  else begin
			if (clk_en) begin
			
//				valid_out_shift[0] <= valid_in;
//            for (int i = 1; i <= block_delay-1; i++) begin
//                valid_out_shift[i] <= valid_out_shift[i-1];
//            end
//				valid_out <= valid_out_shift[block_delay-1];
		  
			  valid_out <= valid_in;
			  if(valid_in) begin
					diff_out <= y_in-prev_in;
					prev_in <= y_in;
							
					
			  end
					// Shift incoming data into the shift register
//					shift_reg[0] <= y_in;
////					for (int i = 1; i <= N; i++) begin
////						 shift_reg[i] <= shift_reg[i-1];
////					end
//				end
//					
//					// Compute first-order difference
////					if(valid_in) begin
//					differences[0] <= y_in- shift_reg[0];
////					end
//					
//					// Compute higher-order differences iteratively
//				  // for (int j = 1; j < N; j++) begin
//					differences[1] <= differences[0] - pref_1st_diff;
//					if (valid_out_shift[0]) begin
//					pref_1st_diff <= differences[0];
//					end
//				  // end

					// Output the final N-th order difference
//					if (valid_out_shift[block_delay]) begin
//					
//					
//					end
	//            valid_out <= 1;
//			  end
		  end
		  end
    end

endmodule
