module tb_modulo_residual();
    // Testbench signals
    logic clk, reset, valid_in;
    logic signed [15:0] diff_in;
    logic valid_out;
    logic signed [15:0] residual_out;

    // Instantiate the modulo residual block
    modulo_residual #(
        .WIDTH(16),
        .LAMBDA(10),
		  .block_delay(0)
    ) uut (
        .clk(clk),
        .reset(reset),
        .valid_in(valid_in),
        .diff_in(diff_in),
        .valid_out(valid_out),
        .residual_out(residual_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 10 ns clock period
    end

    // Stimulus
    initial begin
        reset = 1;
        valid_in = 0;
        diff_in = 0;
        #20 reset = 0;

        // Apply test vectors
        #10 diff_in = 3;  valid_in = 1;  // No wrapping needed
        #10 diff_in = -5; valid_in = 1;  // No wrapping needed
        #10 diff_in = 12; valid_in = 1;  // Wraps to -8
        #10 diff_in = -15; valid_in = 1; // Wraps to 5
        #10 diff_in = 20; valid_in = 1;  // Wraps to 0
        #10 diff_in = -25; valid_in = 1; // Wraps to 15
        #10 valid_in = 0;  // End test
        
        #100 $stop;
    end
endmodule
