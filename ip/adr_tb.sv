module tb_anti_difference_rounding();
    // Testbench signals
    logic clk, reset, valid_in;
    logic signed [15:0] residual_diff_in; // Input from modulo residual block
    logic valid_out;
    logic signed [15:0] residual_out; // Final recovered residuals

    // Instantiate the anti-difference rounding module
    anti_difference_rounding #(
        .WIDTH(16),
        .LAMBDA(10)
    ) uut (
        .clk(clk),
        .reset(reset),
        .valid_in(valid_in),
        .residual_diff_in(residual_diff_in), // Input from modulo residual block
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
        residual_diff_in = 0;
        #20 reset = 0;

        // Apply test vectors (from modulo residual block)
        #10 residual_diff_in = 0;  valid_in = 1;  // No change
        #10 residual_diff_in = 0;  valid_in = 1;  // No change
        #10 residual_diff_in = -20; valid_in = 1; // Cumulative sum decreases
        #10 residual_diff_in = 20; valid_in = 1;  // Cumulative sum returns
        #10 residual_diff_in = -20; valid_in = 1; // Again decreases
        #10 residual_diff_in = 40; valid_in = 1;  // Increases significantly
        #10 valid_in = 0;  // End test
        
        #100 $stop;
    end
endmodule
