module reconstruction_top_pipelined_n #(
    parameter WIDTH = 24,  // Data width
//	 parameter pipe_delay =45, // 8 clock cycle delay, not used, just to keep track
	 parameter LAMBDA =24'h00C000, //0.75
	 parameter FRACTIONAL_BITS =16,
	 parameter WINDOW_SIZE=100,
	 parameter signed one_over_a=24'h010000 //scaling parameter
) (
    input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  clk_en,      // Valid input signal
	 input logic 						 start,
    input  logic  [11:0] adc_in, // y[k] (Modulo ADC output) 
	 input logic [1:0] n, // number of difference order, max3
    output logic                  valid_out,     // Output valid signal
    output logic  [7:0] dac_out // g[k] (Final recovered sample)
);  

	 localparam adc_pipe_len = 150;
	 localparam reconstructed_pipe_len=28;// length of the moving average filter
	 localparam stage_delay=44; 
	 localparam first_two_delay=16;

    logic valid_diff, valid_residual, valid_reconstruction, valid_avg, valid_adc,valid_dac;
	 logic stage1_valid,stage2_valid,stage3_valid;
    logic signed [WIDTH-1:0] diff_out,ftf_out,hod_in,stored_in;
	 logic signed [WIDTH-1:0] reconstructed_sample;
    logic signed [WIDTH-1:0] reconstructed_piped [0:reconstructed_pipe_len-1];
	 logic signed [WIDTH-1:0] adc_out_piped [0:adc_pipe_len-1];
	
    logic signed [WIDTH-1:0] modulo_residual_out,stage1_out,stage2_out,stage3_out;
	 logic signed [WIDTH-1:0] residual_out,avg_out,adc_out;// ε_y[k] (Recovered residual)
//	 logic signed  [WIDTH-1:0] modulo_in_shift [pipe_delay:0];
	 logic valid_in;     // Valid input signal
	 logic signed [WIDTH-1:0] max_in;
	 
	 


// convert adc value to desired format

	adc_to_fixed #(
	.WIDTH(WIDTH)
	
	)adc(
		  .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_in),
        .adc_in(adc_in),
        .valid_out(valid_adc),
        .fixed_in(adc_out) 
	
	);
	always_ff @(negedge clk or posedge reset) begin
		if(reset) begin
//		adc_out_piped[0]<=0;
		
		end
		else begin
		  if (clk_en) begin
			adc_out_piped[0]<=adc_out;
				for (int i = 1; i < adc_pipe_len; i++) begin
					 adc_out_piped[i] <= adc_out_piped[i-1];
				end;
			end
		end
	end
	 



    // Module 1: Higher-Order Difference Calculator (2nd order)
    higher_order_difference_n #(
        .WIDTH(WIDTH)
    ) diff_calc (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_adc),
        .y_in(adc_out),
		  .n(n),
        .valid_out(valid_diff),
        .diff_out(diff_out) // Second-order difference output
    );

    // Module 2: Modulo Residual Calculator
    modulo_residual_piped #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS)
    ) mod_residual (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),		  
        .valid_in(valid_diff),
        .diff_in(diff_out), 
        .valid_out(valid_residual),
        .residual_out(modulo_residual_out)
    );
	 
	 anti_difference_stage #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS),
		  .WINDOW_SIZE(WINDOW_SIZE)
		  
    ) stage_1 (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_residual),
        .residue_in(modulo_residual_out),
        .valid_out(stage1_valid),
        .anti_diff_out(stage1_out)
    );
	 
	 	 anti_difference_stage #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS),
		  .WINDOW_SIZE(WINDOW_SIZE)
    ) stage_2 (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(stage1_valid),
        .residue_in(stage1_out),
        .valid_out(stage2_valid),
        .anti_diff_out(stage2_out)
    );
	 
	 	 anti_difference_stage #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS),
		  .WINDOW_SIZE(WINDOW_SIZE)
    ) stage_3 (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(stage2_valid),
        .residue_in(stage2_out),
        .valid_out(stage3_valid),
        .anti_diff_out(stage3_out)
    );
	 



    // Module 3: Anti-Difference and Rounding
//    anti_difference_rounding #(
//        .WIDTH(WIDTH),
//        .LAMBDA(LAMBDA),
//		  .FRACTIONAL_BITS(FRACTIONAL_BITS)
//    ) anti_diff (
//        .clk(clk),
//		  .clk_en(clk_en),
//        .reset(reset),
//        .valid_in(valid_residual),
//        .residual_diff_in(modulo_residual_out),
//        .valid_out(valid_reconstruction),
//        .residual_out(residual_out)
//    );
//	 logic signed  reconstructed;
//	 logic valid_avg_in;
//	 always_ff @(negedge clk or posedge reset) begin
//		if(reset) begin
////		reconstructed_piped[0]<=0;
//		valid_avg_in<=0;
//		end
//		else begin
//		
//		if(clk_en) begin
//		
//			reconstructed_piped[0]<=residual_out+adc_out_piped[adc_pipe_len-1];
//			
//			for (int i = 1; i < reconstructed_pipe_len; i++) begin
//				 reconstructed_piped[i] <= reconstructed_piped[i-1];
//			end;
//			valid_avg_in<=valid_reconstruction;
//		end
//		end
//	end
//	
//	
//	
//	
//	 moving_average #(
//		.DATA_WIDTH(WIDTH),
//		.WINDOW_SIZE(WINDOW_SIZE),
//		.FRACTIONAL_BITS(FRACTIONAL_BITS)
//		
//	 )mov_avg(
//	     .clk(clk),
//		  .clk_en(clk_en),
//        .reset(reset),
//		  .valid_in(valid_avg_in),
//        .in_sample(reconstructed_piped[0]),
//        .valid_out(valid_avg),
//        .out_avg(avg_out)
//		  
//		
//	 );
//	 
//	 
//	 
//	 
//	 
	 logic valid_convert, valid_convert_intermediate;
	 logic signed [WIDTH-1:0] conversion_in_intermediate;
	 logic signed [WIDTH*2-1:0] conversion_in;
//	 assign conversion_in=residual_out;
	 
	 
   always_ff @(negedge clk) begin
		if (reset) begin
			conversion_in<=0;
			valid_convert_intermediate<=0;
			conversion_in_intermediate<=0;
			valid_convert<=0;
		end
		else begin
			if (clk_en) begin
			
			if(n==2'd1) begin
				conversion_in_intermediate<=adc_out_piped[first_two_delay+stage_delay]+stage1_out; // needs shift reg for pipeline
				valid_convert_intermediate<=stage1_valid;
			end
			else if(n==2'd2) begin
				conversion_in_intermediate<=adc_out_piped[first_two_delay+stage_delay*2]+stage2_out; // needs shift reg for pipeline
				valid_convert_intermediate<=stage2_valid;
			
			end
			else if(n==2'd3) begin
				conversion_in_intermediate<=adc_out_piped[first_two_delay+stage_delay*3]+stage3_out; // needs shift reg for pipeline
				valid_convert_intermediate<=stage3_valid;
			
			end
			else begin
				conversion_in_intermediate<=adc_out_piped[adc_pipe_len-1]+stage1_out; // needs shift reg for pipeline
				valid_convert_intermediate<=stage1_valid;
				
			
			end
			
			conversion_in<=conversion_in_intermediate*one_over_a;
			valid_convert<=valid_convert_intermediate;
//				conversion_in<=avg_out;
				
			end
		end
	end
	 
//	 
	 reg  [7:0] converted_residual;
	 
	 
	 
//	 fixed_to_float #(
//		.TOTAL_BITS(WIDTH),
//		.FRACTIONAL_BITS(FRACTIONAL_BITS)
//	 )outconv(
//		
//		.fixed_in(conversion_in),
//		.float_out(converted_residual)
//	 );
//	 
	 
	 
	 fixed_to_dac #(
	 .WIDTH(WIDTH)
	 
	 )dac(
	 
	 	   .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
		  .valid_in(valid_convert),
        .signal_in({conversion_in[47],conversion_in[38:16]}),
        .valid_out(valid_dac),
        .dac_value(converted_residual)
		  
		  
	 
	 );
	 

	 assign valid_in = start & clk_en;
//	 logic second_cycle;
    always_ff @(negedge clk or posedge reset) begin
			
//		  second_cycle <= clk_en;
		  
        if (reset) begin
            reconstructed_sample <= 0;
            valid_out <= 0;
				dac_out<=0;
				//second_cycle <=0;
				max_in=0;

        end 
		  else begin
//				if (start && clk_en) begin
//					 stored_in <= hod_in;
//s
//				end
				
				if (clk_en) begin
				

					dac_out <=  converted_residual ;
//					second_cycle<=valid_in;
//				
//					valid_out <=second_cycle;

					valid_out <= valid_in;

				end
				else begin
//					valid_out<=0;
//					dac_out <=0;
				end
		  end
    end
	 
	 

	 

endmodule
