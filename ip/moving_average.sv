module moving_average #(
    parameter WINDOW_SIZE = 100,
    parameter DATA_WIDTH = 24,// 1 sign bit, 7 int, 16 frac
	 parameter FRACTIONAL_BITS=16,
    parameter SUM_WIDTH = 32,   // 1 sign bit, 15 int, 16 frac
	 parameter BLOCK_DELAY=28
) (
    input logic clk,
	 input logic clk_en,
    input logic reset,
	 input logic valid_in,
	 output logic valid_out,
    input logic signed [DATA_WIDTH-1:0] in_sample,
    output logic signed [DATA_WIDTH-1:0] out_avg
);

//	localparam [DATAWIDTH-1:0] window_size_shifted= WINDOW_SIZE <<< sFRACTIONAL_BITS

    // Buffer to store the last WINDOW_SIZE samples
    logic signed [SUM_WIDTH-1:0] buffer [0:WINDOW_SIZE-1];
    logic [$clog2(WINDOW_SIZE)-1:0] index;
    logic signed [7:0] sample_count, temp_sample_count;  // Track number of samples received
//	 logic [$clog2(BLOCK_DELAY)-1:0] ready_index; we don't use this because indexing logic complicated

    // Sum accumulator with extra width to prevent overflow
    logic signed [SUM_WIDTH-1:0] sum, div_denom, div_res,div_remain,in_extended;
	 logic signed [47:0] div_numer;
	 logic [BLOCK_DELAY-1:0] ready_shifted ;
	 logic signed [DATA_WIDTH-1:0] prev_in_sample;
	 
	 
	 
	 
	 
	 avg_div	avg_div_inst (
	.aclr ( reset ),
	.clken ( clk_en ),
	.clock ( clk ),
	.denom ( div_denom ),
	.numer ( div_numer ),
	.quotient ( div_res ),
	.remain ( div_remain )
	);

//		localparam NUM_PARALLEL=2;

    // Initialization
//	 assign in_extended={{9{in_sample[DATA_WIDTH-1]}},in_sample[DATA_WIDTH-2:0]};
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            index <= 0;
            sample_count <= 0;
				temp_sample_count <= 0;
            sum <= 0;
				out_avg<=0;
				div_denom<=0;
				div_numer<=0;
//			for (int j = 0; j < NUM_PARALLEL; j++) begin
            for (int i = 0; i < WINDOW_SIZE; i ++) begin
                buffer[i] <= 0;
            end
//        end
				ready_shifted <=0;
				
        end else begin
		  
				if (clk_en) begin
					// ready logic
					
					ready_shifted[0]<=valid_in;
					for (int i = 1; i < BLOCK_DELAY; i++) begin
                ready_shifted[i] <= ready_shifted[i-1];
					end
		
					valid_out <= ready_shifted[BLOCK_DELAY-1];
					
				
					if (valid_in) begin // first cycle
//						prev_in_sample<=in_sample;
					// Subtract the oldest sample from the sum
					
						 if (sample_count == WINDOW_SIZE) begin
								sum <= sum - buffer[index] + {{9{in_sample[DATA_WIDTH-1]}},in_sample[DATA_WIDTH-2:0]};
						 end
						 else if(sample_count <WINDOW_SIZE) begin
								sum <= sum + {{9{in_sample[DATA_WIDTH-1]}},in_sample[DATA_WIDTH-2:0]};
						 
						end
						
						
					  buffer[index] <= {{9{in_sample[DATA_WIDTH-1]}},in_sample[DATA_WIDTH-2:0]};
					  index <= (index + 1) % WINDOW_SIZE;

						// Increment sample count until full
						if (sample_count < WINDOW_SIZE) begin
							 sample_count <= sample_count + 1;
						end
						
						
						
					end
//					if (ready_shifted[0]) begin
//						// Store the new sample and update the sum
//						
////						sum <= sum + {{9{prev_in_sample[DATA_WIDTH-1]}},prev_in_sample[DATA_WIDTH-2:0]};
////						temp_sample_count <= sample_count;
//
//						// Update the buffer index
//
//						
//					
//					end
					
					if (ready_shifted[0]) begin// second cycle, append into divider result
//						if (sample_count==0) begin
//							div_denom<=1 <<<FRACTIONAL_BITS;
//						end
//						else begin
						div_denom<=sample_count<<< (FRACTIONAL_BITS);
//						end
						
						div_numer<=sum<<< FRACTIONAL_BITS;
						
						
					
					end
					
					
					
					if (ready_shifted[BLOCK_DELAY-1]) begin// last cycle, output result
						out_avg <= {div_res[SUM_WIDTH-1], div_res[DATA_WIDTH-2:0]};
						
					
					end
					
					
					
					
				end
				
        end
    end

    // Output calculation

endmodule
