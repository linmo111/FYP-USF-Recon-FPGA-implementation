
module DE10_NANO_QSYS (
	adc_ltc2308_conduit_end_convst,
	adc_ltc2308_conduit_end_sck,
	adc_ltc2308_conduit_end_sdi,
	adc_ltc2308_conduit_end_sdo,
	adc_ltc2308_conduit_end_dec,
	clk_clk,
	i2c_dac_serial_sda_in,
	i2c_dac_serial_scl_in,
	i2c_dac_serial_sda_oe,
	i2c_dac_serial_scl_oe,
	pll_sys_locked_export,
	pll_sys_outclk2_clk,
	reset_reset_n,
	sw_external_connection_export);	

	output		adc_ltc2308_conduit_end_convst;
	output		adc_ltc2308_conduit_end_sck;
	output		adc_ltc2308_conduit_end_sdi;
	input		adc_ltc2308_conduit_end_sdo;
	input		adc_ltc2308_conduit_end_dec;
	input		clk_clk;
	input		i2c_dac_serial_sda_in;
	input		i2c_dac_serial_scl_in;
	output		i2c_dac_serial_sda_oe;
	output		i2c_dac_serial_scl_oe;
	output		pll_sys_locked_export;
	output		pll_sys_outclk2_clk;
	input		reset_reset_n;
	input	[9:0]	sw_external_connection_export;
endmodule
