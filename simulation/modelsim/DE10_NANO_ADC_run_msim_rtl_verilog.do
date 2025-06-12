transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -vlog01compat -work work +incdir+D:/year4/fyp/ADC {D:/year4/fyp/ADC/DE10_NANO_ADC.v}
vlib DE10_NANO_QSYS
vmap DE10_NANO_QSYS DE10_NANO_QSYS
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/DE10_NANO_QSYS.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_reset_controller.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_reset_synchronizer.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_avalon_st_adapter.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_customins_master_translator.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_timer_0.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_sysid_qsys.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_sw.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/anti_diff_div.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/avg_div.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/mod_divide.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_pll_sys.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_onchip_memory2.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_nios2_qsys.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_jtag_uart.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_csr.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_clk_cnt.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_condt_det.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_condt_gen.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_fifo.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_mstfsm.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_rxshifter.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_txshifter.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_spksupp.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_i2c_txout.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/adc_ltc2308_fifo.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/adc_ltc2308.v}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/adc_data_fifo.v}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_irq_mapper.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_avalon_st_adapter_error_adapter_0.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_rsp_mux_001.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_arbitrator.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_rsp_mux.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_rsp_demux_003.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_rsp_demux.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_cmd_mux_003.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_cmd_mux.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_cmd_demux_001.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_cmd_demux.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_traffic_limiter.sv}
vlog -vlog01compat -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_avalon_sc_fifo.v}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_router_005.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_router_002.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_router_001.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_mm_interconnect_0_router.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_slave_agent.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_burst_uncompressor.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_master_agent.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_slave_translator.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_merlin_master_translator.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/altera_customins_slave_translator.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/DE10_NANO_QSYS_nios2_qsys_custom_instruction_master_multi_xconnect.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/HOD_n.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/anti_difference_stage.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/top_piped_n2.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/adc_to_fixed.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/modulo_residual_piped.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/moving_average.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/anti_difference_rounding.sv}
vlog -sv -work DE10_NANO_QSYS +incdir+D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules {D:/year4/fyp/ADC/DE10_NANO_QSYS/synthesis/submodules/fixed_to_dac.sv}

