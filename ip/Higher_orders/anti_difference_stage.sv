module anti_difference_stage #(
    parameter WIDTH = 24,  // Data width
//	 parameter pipe_delay =45, // 8 clock cycle delay, not used, just to keep track
	 parameter LAMBDA =24'h00C000, //0.75
	 parameter FRACTIONAL_BITS =16,
	 parameter WINDOW_SIZE=100
//	 parameter Block_delay=45
) (
    input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  clk_en,      // Valid input signal
    input  logic  [WIDTH-1:0] residue_in, // y[k] (Modulo ADC output) 
	 input logic valid_in,
    output logic                  valid_out,     // Output valid signal
    output logic  [WIDTH-1:0] anti_diff_out // g[k] (Final recovered sample)
);  

//	 localparam adc_pipe_len = 30;
	 localparam filter_pipe_len=28;// length of the moving average filter

    logic valid_diff, valid_residual, valid_reconstruction, valid_avg, valid_adc,valid_dac;
    logic signed [WIDTH-1:0] diff_out,ftf_out,hod_in,stored_in;
	 logic signed [WIDTH-1:0] reconstructed_sample;
    logic signed [WIDTH-1:0] filter_piped [0:filter_pipe_len-1];
//	 logic signed [WIDTH-1:0] adc_out_piped [0:adc_pipe_len-1];
	
    logic signed [WIDTH-1:0] modulo_residual_out;
	 logic signed [WIDTH-1:0] residual_out,avg_out,adc_out;// ε_y[k] (Recovered residual)
//	 logic signed  [WIDTH-1:0] modulo_in_shift [pipe_delay:0];
//	 logic valid_in;     // Valid input signal
	 logic signed [WIDTH-1:0] max_in;
	 


    // Module 3: Anti-Difference and Rounding
    anti_difference_rounding #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS)
    ) anti_diff (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_in),
        .residual_diff_in(residue_in),
        .valid_out(valid_reconstruction),
        .residual_out(residual_out)
    );
//	 logic signed  reconstructed;
	 logic valid_avg_in;
	 always_ff @(negedge clk or posedge reset) begin
		if(reset) begin
		for (int i = 0; i < filter_pipe_len; i++) begin
				 filter_piped[i] <= 0;
			end;
		
//		reconstructed_piped[0]<=0;
		valid_avg_in<=0;
		end
		else begin
		
		if(clk_en) begin
		
			filter_piped[0]<=residual_out;
			
			for (int i = 1; i < filter_pipe_len; i++) begin
				 filter_piped[i] <= filter_piped[i-1];
			end;
			valid_avg_in<=valid_reconstruction;
		end
		end
	end
	
	
	
	
	 moving_average #(
		.DATA_WIDTH(WIDTH),
		.WINDOW_SIZE(WINDOW_SIZE),
		.FRACTIONAL_BITS(FRACTIONAL_BITS)
		
	 )mov_avg(
	     .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
		  .valid_in(valid_avg_in),
        .in_sample(filter_piped[0]),
        .valid_out(valid_avg),
        .out_avg(avg_out)
		  
		
	 );
	 
	 
	 
	 

	 

//	 assign valid_in = start & clk_en;
//	 logic second_cycle;
    always_ff @(negedge clk or posedge reset) begin
			
//		  second_cycle <= clk_en;
		  
        if (reset) begin
            anti_diff_out <= 0;
            valid_out <= 0;
//				dac_out<=0;
//				second_cycle <=0;
//				max_in=0;

        end 
		  else begin
//				if (start && clk_en) begin
//					 stored_in <= hod_in;
//s
//				end
				
				if (clk_en) begin
				
					
					anti_diff_out <=  filter_piped[filter_pipe_len-1]-avg_out ;
//					second_cycle<=valid_in;
//				
//					valid_out <=second_cycle;

					valid_out <= valid_avg;

				end
				else begin
//					valid_out<=0;
//					dac_out <=0;
				end
		  end
    end
	 
	 

	 

endmodule
