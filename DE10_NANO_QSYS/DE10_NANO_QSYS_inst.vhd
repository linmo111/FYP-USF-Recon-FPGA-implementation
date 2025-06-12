	component DE10_NANO_QSYS is
		port (
			adc_ltc2308_conduit_end_convst : out std_logic;                                       -- convst
			adc_ltc2308_conduit_end_sck    : out std_logic;                                       -- sck
			adc_ltc2308_conduit_end_sdi    : out std_logic;                                       -- sdi
			adc_ltc2308_conduit_end_sdo    : in  std_logic                    := 'X';             -- sdo
			adc_ltc2308_conduit_end_dec    : in  std_logic                    := 'X';             -- dec
			clk_clk                        : in  std_logic                    := 'X';             -- clk
			i2c_dac_serial_sda_in          : in  std_logic                    := 'X';             -- sda_in
			i2c_dac_serial_scl_in          : in  std_logic                    := 'X';             -- scl_in
			i2c_dac_serial_sda_oe          : out std_logic;                                       -- sda_oe
			i2c_dac_serial_scl_oe          : out std_logic;                                       -- scl_oe
			pll_sys_locked_export          : out std_logic;                                       -- export
			pll_sys_outclk2_clk            : out std_logic;                                       -- clk
			reset_reset_n                  : in  std_logic                    := 'X';             -- reset_n
			sw_external_connection_export  : in  std_logic_vector(9 downto 0) := (others => 'X')  -- export
		);
	end component DE10_NANO_QSYS;

	u0 : component DE10_NANO_QSYS
		port map (
			adc_ltc2308_conduit_end_convst => CONNECTED_TO_adc_ltc2308_conduit_end_convst, -- adc_ltc2308_conduit_end.convst
			adc_ltc2308_conduit_end_sck    => CONNECTED_TO_adc_ltc2308_conduit_end_sck,    --                        .sck
			adc_ltc2308_conduit_end_sdi    => CONNECTED_TO_adc_ltc2308_conduit_end_sdi,    --                        .sdi
			adc_ltc2308_conduit_end_sdo    => CONNECTED_TO_adc_ltc2308_conduit_end_sdo,    --                        .sdo
			adc_ltc2308_conduit_end_dec    => CONNECTED_TO_adc_ltc2308_conduit_end_dec,    --                        .dec
			clk_clk                        => CONNECTED_TO_clk_clk,                        --                     clk.clk
			i2c_dac_serial_sda_in          => CONNECTED_TO_i2c_dac_serial_sda_in,          --          i2c_dac_serial.sda_in
			i2c_dac_serial_scl_in          => CONNECTED_TO_i2c_dac_serial_scl_in,          --                        .scl_in
			i2c_dac_serial_sda_oe          => CONNECTED_TO_i2c_dac_serial_sda_oe,          --                        .sda_oe
			i2c_dac_serial_scl_oe          => CONNECTED_TO_i2c_dac_serial_scl_oe,          --                        .scl_oe
			pll_sys_locked_export          => CONNECTED_TO_pll_sys_locked_export,          --          pll_sys_locked.export
			pll_sys_outclk2_clk            => CONNECTED_TO_pll_sys_outclk2_clk,            --         pll_sys_outclk2.clk
			reset_reset_n                  => CONNECTED_TO_reset_reset_n,                  --                   reset.reset_n
			sw_external_connection_export  => CONNECTED_TO_sw_external_connection_export   --  sw_external_connection.export
		);

