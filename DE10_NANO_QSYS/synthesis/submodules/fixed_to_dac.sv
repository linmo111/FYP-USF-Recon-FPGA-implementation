module fixed_to_dac #(
	parameter WIDTH =24, 
	parameter block_delay=2,
//	parameter FRACTIONAL_BITS =16,
	parameter dac_max_over_vref=24'h003300, // Q 15,8
	parameter offset=24'h0000200,// Q 15,8
	parameter vref=24'h0000500 //Q 15.8s
//	parameter dac_max=255 //
)(
	 input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  valid_in,      // Valid input signal
	 input	logic 						clk_en,
	 output logic                  valid_out,     // Output valid signal


    input  logic signed [23:0] signal_in,  // 24-bit fixed point (1 sign, 7 int, 16 frac)
    output logic [7:0] dac_value               // 8-bit DAC output
);

    // Internal signals
    logic signed [23:0] signal_with_offset;   // Signal after applying offset
   
		logic [block_delay-1:0] valid_out_shift;
    // Apply OFFSET
	 
	 logic signed [23:0] in_shifted; // shift Q7,16 to Q 15,8 for precision
	 assign in_shifted=signal_in >>>8;
	 logic signed [47:0] scaled_signal; // Q 31,16
    always_ff @(posedge clk or posedge reset)  begin
	 
		if (reset) begin
			valid_out_shift<=0;
			valid_out<=0;
			signal_with_offset<=0;
			scaled_signal<=0;
			dac_value<=0;
		
		end
	 
	 
		else begin
	 
	 
		if (clk_en) begin 
			
			valid_out_shift[0] <= valid_in;
			
			for (int i = 1; i <= block_delay-1; i++) begin
				 valid_out_shift[i] <= valid_out_shift[i-1];
			end

			valid_out <= valid_out_shift[block_delay-1];
						
	 
			if (valid_in) begin
			
			  if (in_shifted+offset < 0) begin
					signal_with_offset <= 0;
		
			  end else if (in_shifted+offset > vref) begin
					signal_with_offset <= vref;  // V_REF shifted to Q7.16 format
			  end else begin
			  signal_with_offset <= (in_shifted + offset) ; // right 
			  end
				
			end
	 
	 
        if (valid_out_shift[0]) begin
				scaled_signal <= signal_with_offset*dac_max_over_vref;

		  end
        // Convert to DAC value

        // Assign the scaled signal to the DAC output
		  if (valid_out_shift[block_delay-1]) begin
				dac_value <= scaled_signal[23:16];
		  end 
    end
	 end
	end
endmodule
