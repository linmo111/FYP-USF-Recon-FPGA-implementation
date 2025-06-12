module reconstruction_top #(
    parameter WIDTH = 24,  // Data width
	 parameter pipe_delay =45, // 8 clock cycle delay, not used, just to keep track
	 parameter LAMBDA =24'h00C000, //0.75
	 parameter FRACTIONAL_BITS =16,
	 parameter WINDOW_SIZE=100
) (
    input  logic                  clk,           // Clock signal
    input  logic                  reset,         // Reset signal
    input  logic                  clk_en,      // Valid input signal
	 input logic 						 start,
    input  logic  [11:0] adc_in, // y[k] (Modulo ADC output) 
    output logic                  valid_out,     // Output valid signal
    output logic  [7:0] dac_out // g[k] (Final recovered sample)
);  

	

    logic valid_diff, valid_residual, valid_reconstruction, valid_avg, valid_adc,valid_dac;
    logic signed [WIDTH-1:0] diff_out,ftf_out,hod_in,stored_in;
	 logic signed [WIDTH-1:0] reconstructed_sample;
    
	
    logic signed [WIDTH-1:0] modulo_residual_out;
	 logic signed [WIDTH-1:0] residual_out,avg_out,adc_out;// ε_y[k] (Recovered residual)
//	 logic signed  [WIDTH-1:0] modulo_in_shift [pipe_delay:0];
	 logic valid_in;     // Valid input signal
//	 logic second_cycle;

//	 logic [WIDTH-1:0] diff_ou

	// floating point input 
	 
//initial begin
//
//
//
//
//end	 
//	 
//	 float_to_fixed #(
//		.TOTAL_BITS(WIDTH),
//		.FRACTIONAL_BITS(FRACTIONAL_BITS)
//	 )inconv(
//		.float_in(adc_in),
//		.fixed_out(ftf_out)
//	 );
//	 logic ftf_valid;
//	 always_ff @(posedge clk) begin
//		hod_in<=ftf_out;
//		ftf_valid <=valid_in;
//	end
//	 

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
        .fixed_in(adc_out) // Second-order difference output
	
	);



    // Module 1: Higher-Order Difference Calculator (2nd order)
    higher_order_difference #(
        .N(2),
        .WIDTH(WIDTH)
    ) diff_calc (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_adc),
        .y_in(adc_out),
        .valid_out(valid_diff),
        .diff_out(diff_out) // Second-order difference output
    );

    // Module 2: Modulo Residual Calculator
    modulo_residual #(
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

    // Module 3: Anti-Difference and Rounding
    anti_difference_rounding #(
        .WIDTH(WIDTH),
        .LAMBDA(LAMBDA),
		  .FRACTIONAL_BITS(FRACTIONAL_BITS)
    ) anti_diff (
        .clk(clk),
		  .clk_en(clk_en),
        .reset(reset),
        .valid_in(valid_residual),
        .residual_diff_in(modulo_residual_out),
        .valid_out(valid_reconstruction),
        .residual_out(residual_out)
    );
	 logic signed [WIDTH-1:0] reconstructed;
	 logic valid_avg_in;
	 always_ff @(negedge clk) begin
		reconstructed<=residual_out+adc_out;
		valid_avg_in<=valid_reconstruction;
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
        .in_sample(reconstructed),
        .valid_out(valid_avg),
        .out_avg(avg_out)
		  
		  
		  
	 
		
	 );
	 
	 
	 
	 
	 
	 logic valid_convert;
	 logic signed [WIDTH-1:0] conversion_in;
//	 assign conversion_in=residual_out;
	 
	 always_ff @(negedge clk) begin
		conversion_in<=reconstructed-avg_out; // needs shift reg for pipeline
		valid_convert<=valid_avg;
	end
	 
	 
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
        .signal_in(conversion_in),
        .valid_out(valid_dac),
        .dac_value(converted_residual)
		  
		  
	 
	 );
	 
	 
	 

	 assign valid_in = start & clk_en;
    always_ff @(negedge clk or posedge reset) begin
			
//		  second_cycle <= clk_en;
		  
        if (reset) begin
            reconstructed_sample <= 0;
            valid_out <= 0;
//				valid_convert<=0;
//				modulo_in_shift [0] <= 0;
//				stored_in <=0;
				
//				modulo_residual_out <=0;
//				valid_residual <=0;
//				residual_out <=0;
//				second_cycle <=0;
        end 
		  else begin
//				if (start && clk_en) begin
//					 stored_in <= hod_in;
//s
//				end
		  
				if (clk_en) begin
					valid_out <= valid_dac;
		  // the following code is for pipelining, but now pipelining is gone 
//					modulo_in_shift [0] <= adc_in;
//					for (int i = 1; i <= pipe_delay; i++) begin
//						 modulo_in_shift[i] <= modulo_in_shift[i-1];
//					end
					dac_out <=  (valid_dac) ? (converted_residual) : 0;

//					reconstructed_sample <=  (valid_reconstruction) ? (hod_in+residual_out) : 0;
//					reconstructed_sample <=  (hod_in+residual_out) ;
				end
				else begin
					valid_out<=0;
					dac_out <=0;
				end
		  end
    end
	 
	 

	 

endmodule
