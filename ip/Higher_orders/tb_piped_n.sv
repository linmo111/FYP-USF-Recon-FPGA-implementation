	 `timescale 1ns/1ps

module aa_tb_top_pipe_n();
    // Testbench signals
    logic clk, reset,clk_en;
    logic [11:0] adc_input;
    logic valid_out;
    logic signed [31:0] reconstructed_sample;
	 logic [1:0] n;
	 logic  [11:0] test_in;
	 logic start;
	 logic [15:0] delay;
	 

	 
	 
	 assign test_in=1000;
	 assign delay=200;
//	 assign n=3;

    // Instantiate the top-level module
    reconstruction_top_pipelined_n #(
        .WIDTH(24),
        .LAMBDA(24'h00c000),
		  .FRACTIONAL_BITS(16)
    ) uut (
        .clk(clk),
		  .start(start),
        .reset(reset),
		  .clk_en(clk_en),
		  .n(n),
        .adc_in(adc_input),
        .valid_out(valid_out),
        .dac_out(reconstructed_sample)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 10 ns clock period
    end

    // Stimulus
    initial begin
        reset = 1;
        clk_en = 0;
        adc_input = 0;
		  start=0;
		  n=2;
        #20 reset = 0;

        // Apply test vectors (ADC inputs)

			clk_en=1;
			
			start=1;
			
		 adc_input = 0; // Should unwrap
		 	#10;
			start=0;
//			clk_en=0;
			#2000;
			clk_en=0;
			#10;
			start=1;
			clk_en=1;
				

		 adc_input = 0; // Should unwrap
			#10;	
			start=0;
//			clk_en=0;
			#2000;
			clk_en=0;
			#10;
			start=1;
			clk_en=1;
			
		  adc_input = test_in; // First input
		  		 	#10;
			start=0;
//			clk_en=0;
			#2000;
						clk_en=0;
			#10;
			start=1;
			clk_en=1;


		 adc_input = 0; // Should unwrap
		 		 	#10;
			start=0;
//			clk_en=0;
			#2000;
						clk_en=0;
			#10;
			start=1;
			clk_en=1;
			adc_input = test_in;  // Should unwrap
					 	#10;
			start=0;
//			clk_en=0;
			#2000;
						clk_en=0;
			#10;
			start=1;
			clk_en=1;

		  adc_input = test_in;    // Neutral
		  		 	#10;
			start=0;
//			clk_en=0;
			#2000;
						clk_en=0;
			#10;
			start=1;
			clk_en=1;

		  
		  adc_input = 0; // Large negative
		  #10;
		  
			start=0;
			#2000 
			clk_en=0;
			#10;
		start=1;
		clk_en=1;			
		adc_input = 0;    // Neutral
		  		 	#10;
			start=0;
//			clk_en=0;
			#2000;
			clk_en=1;
			#10;

		 // clk_en=0;  // Stop
        
        #1000 $stop;
    end
endmodule
