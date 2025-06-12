This is my Final year project repository. On the implementation of Unlimited Sensing Framework reconstruction algorithm on a FPGA.

# User instruction:

1. Connect the hardware components as per the schematic in the figure. You will need:

An Altera DE-10 Nano FPGA

A Picoscope or waveform viewer

A DAC

A modulo ADC

2. Open Quartus and open the project. (Version 18.1 was used, but newer versions should work fine.)

3. Connect the FPGA to your laptop. If the board is not detected, install the relevant drivers.
Open Programmer, click Auto-detect, delete the second file, and add the time-limited .sopc file. Upload it to the FPGA.

4. Open Eclipse, select the software folder as your workspace.
Choose the streaming reconstruct folder, generate BSP, and build the project.
Run the program. If no error messages appear, it should be running successfully.

5. Assuming the input is connected, use the Picoscope to view the output waveform.
The default output without input should be a flat line at 2V.
You should see the reconstructed waveform from the DAC output.

6.You can change the model used by editing the code in the main loop.
The top option is the C-reference model.

# Content Guide

## IP
The IP folder contains all of the hardware blocks. The files that are not in the specific folders are for baseline&pipelined implementation. The higher orders folder contains the higher-order submodules. The ADC_LTC2308_FIFO folder contains ADC code. Any code that has tb means it is a testbench.
## Software
The software folder contains c code to run on Eclipse. The main file to recreate the streaming outcome will be in the streaming_reconstruct folder. The other folders are used for testing, check their comments for specific information.

## Python_model

The python simulation model is in this folder. Use reconv3.py and 2nd_order_simulation.py. Sin200B.txt is ideal data that works, real.txts are real data collected from the fpga, which does not work in the simulation.



# Changing Parameters
Parameters of the system can be changed in Platform Designer once you’ve added the IP to your system.

lambda and one_over_a should be in 24-bit fixed-point format (1 sign bit, 7 integer bits, 16 fractional bits).

All other parameters can be set as integers.

After making changes:
1.Save the system
2.Generate HDL in platform designer
3.Recompile the entire project
