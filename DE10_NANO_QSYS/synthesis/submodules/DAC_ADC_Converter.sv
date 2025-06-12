module DAC_ADC_CustomInstruction (
    input  logic        clk,        // Clock input
    input  logic        reset,      // Reset input
    input  logic [31:0] dataa,      // Input data (ADC value)
//    input  logic [31:0] datab,      // Unused second input (if not required)
    output logic [31:0] result      // Output data (DAC value)
);

    // Internal signals
    logic [11:0] ADC_Val_internal;  // 12-bit ADC value
    logic [7:0]  DAC_Val_internal;  // 8-bit DAC value

    // Capture ADC value on change of dataa
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            ADC_Val_internal <= 12'b0; // Reset ADC value
        end else begin
            ADC_Val_internal <= dataa[11:0]; // Capture lower 12 bits of input
        end
    end

    // Perform conversion
    always_comb begin
        DAC_Val_internal = (ADC_Val_internal*51 /1000);
    end

    // Output DAC value
    assign result = {24'b0, DAC_Val_internal}; // Output as 32-bit result
endmodule
