module adc_to_fixed #(
	parameter WIDTH =24, 
	parameter block_delay=2,
//	parameter FRACTIONAL_BITS =16,
	parameter vref_over_adc_max=24'h000002, // Q 12,11
	parameter offset=0,// Q 12,11
	parameter vref=24'h0020c5 //Q 12.11
//	parameter dac_max=255 //
)(
	 input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  valid_in,      // Valid input signal
	 input	logic 						clk_en,
	 output logic                  valid_out,     // Output valid signal


    input  logic signed [11:0] adc_in,  // 12-bit DAC output
    output logic [WIDTH-1:0] fixed_in               // 24-bit fixed point (1 sign, 7 int, 16 frac)
);

    // Internal signals
    logic signed [23:0] signal_with_offset, before_con;   // Signal after applying offset
    
		logic [block_delay-1:0] valid_out_shift;
    // Apply OFFSET
	 
	 logic signed [23:0] in_shifted; // shift Q12 to Q 12,11 for same formatting
	 assign in_shifted={1'b0,adc_in} <<<11; 
	 logic signed [47:0] scaled_signal; // Q 31,16
    always_ff @(posedge clk or posedge reset)  begin
	 
	 
		if (reset) begin
			valid_out_shift<=0;
			valid_out<=0;
			signal_with_offset<=0;
			scaled_signal<=0;
		
		
		
		end
	 
		else begin
	 
		if (clk_en) begin 
			
			valid_out_shift[0] <= valid_in;
			
			for (int i = 1; i <= block_delay-1; i++) begin
				 valid_out_shift[i] <= valid_out_shift[i-1];
			end

			valid_out <= valid_out_shift[block_delay-1];
						
	 
			if (valid_in) begin
			
			  signal_with_offset <= (in_shifted + offset) ; // right 
			 end
			
			
				
	 
	 
        if (valid_out_shift[0]) begin //scaled signal is Q 25 22
				scaled_signal <= signal_with_offset*vref_over_adc_max;
		  
		  

		  end
        // Convert to DAC value

        // Assign the scaled signal to the DAC output
		  if (valid_out_shift[block_delay-1]) begin 
				fixed_in <= {scaled_signal[47],scaled_signal[28:6]} ;
		  end 
    end
	 end
	end
endmodule
