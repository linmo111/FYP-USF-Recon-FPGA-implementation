
# THIS is OUTDATED. use reconv3 or 2nd order simulation

input=[0,0,-0.3,-0.3,0,-0.3,0,0]
# input=[0,0,0.1,0.1,0,0.1,0,0]
import numpy as np
import matplotlib.pyplot as plt
# Read the file and split into float values
with open('sin100.txt', 'r') as f:
    data = f.read().split()

# Convert to floats and take the first 10,000 elements
values = np.array(data[:10000000], dtype=float)

largest=0
smallest=0
offset=0
# for v in values:
#     if v>largest:
#         largest=v
#     if v<smallest:
#         smallest=v
# offset=(largest-smallest)/2-largest
# print(offset)
offset=0.26242613795


import numpy as np

def modulo(x, lam):
    """Centered modulo operation, mapping to [-λ, λ)."""
    return ((x + lam) % (2 * lam)) - lam

def finite_diff(x, order=1):
    """Compute the finite difference of a sequence."""
    for _ in range(order):
        x = np.diff(x, prepend=x[0])
    return x

def anti_diff(x, initial=0):
    """Compute the anti-difference (discrete integration)."""
    return np.cumsum(np.insert(x, 0, initial))[1:]

def round_to_nearest_grid(x, grid_step):
    """Round each element to nearest multiple of grid_step."""
    return grid_step * np.round(x / grid_step)

def estimate_kappa(diff2_seq, lam, J, beta):
    """Estimate the integer constant needed to fix anti-diff ambiguity."""
    slope = 2 * lam
    delta = (diff2_seq[0] - diff2_seq[J]) / (slope * J)
    return int(np.round(delta))

def reconstruct_unlimited_samples(y, lam, Omega, T, beta):
    """Reconstruct the bandlimited signal from modulo samples y."""
    y = np.array(y)
    DR = beta / lam  # dynamic range

    # Step 1: Choose N so that (T * Ω * e)^N * beta < λ
    e = np.e
    N = int(np.ceil((np.log(lam) - np.log(beta)) / np.log(T * Omega * e)))

    # Step 2: Estimate Δ^N ε = Mλ(Δ^N y) - Δ^N y
    diff_N_y = finite_diff(y, order=N)
    mod_diff_N_y = modulo(diff_N_y, lam)
    diff_N_eps = mod_diff_N_y - diff_N_y

    # Step 3: Recursively anti-diff and round to 2λℤ
    s = diff_N_eps.copy()
    for n in range(N - 1):
        s = anti_diff(s)
        s = round_to_nearest_grid(s, 2 * lam)

        # Estimate and correct constant offset using a sliding window
        J = int(np.ceil(6 * beta / lam))
        if len(s) > J + 1:
            kappa = estimate_kappa(s, lam, J, beta)
            s += 2 * lam * kappa

    # Step 4: Recover samples of g[k] = y[k] + ε[k]
    eps_hat = anti_diff(s)
    g_hat = y + eps_hat

    return g_hat  # these are samples of the original bandlimited signal

# Example usage:
# y = np.array([...])  # your modulo samples
# lam = 1.0
# Omega = np.pi
# T = 1 / (2 * Omega * np.e)  # ensure oversampling condition
# beta = 10.0  # estimated max amplitude of the true signal
# g_hat = reconstruct_unlimited_samples(y, lam, Omega, T, beta)


shift_reg=[0]*1
differences=[0]*2

prev_val=0
def higher_order_difference(input):
    global prev_val
    cur_diff=input-prev_val
    prev_val=input
    # prev_diff=differences[0]
    # differences[0]=input-shift_reg[0]
    # differences[1]=differences[0]-prev_diff
    
    # shift_reg[0]=input
    return  cur_diff
    
def modulo_residual(diff_in, L=1):
    residual_out= ((diff_in + L) %( 2*L)+2*L)%(2*L)-L-diff_in
    return residual_out

first_order_diff=0
residual=0
def anti_diff_rounding(residual_diff_in,L=1):
    global first_order_diff,residual
    first_order_diff=first_order_diff+residual_diff_in
    residual=residual+first_order_diff

    residual_out=round(residual/(2*L))*(2*L)
    return residual_out

def top(adc_in,L=1):

    diff_out=higher_order_difference(adc_in)
    modulo_residual_out=modulo_residual(diff_out,L)
    residual_out=anti_diff_rounding(modulo_residual_out,L)
    return adc_in
    return residual_out+adc_in





def print_output_graph(input_array,timestep,offset,downsample_val=200,L=1):
    # processed=unwrap(input_array,L)
    # processed=reconstruct_unlimited_samples(input_array,L,np.pi,T = 1 / (2 * np.pi * np.e),beta=2)
    processed=[0]*len(input_array)
    for i in range(len(input_array)):
        if i % downsample_val==0:
        # processed[i]=higher_order_difference(input_array[i])
        # processed[i]=modulo_residual(processed[i],L)
            processed[i]=top(input_array[i],L)
  

    time = np.arange(len(processed)) * (timestep)

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, processed)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform from output.txt')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# for e in input:
#     print(top(e,L=0.1))


print_output_graph(values,2e-9,offset,L=1)
# timestep = 2e-9
# time = np.arange(len(values)) * timestep

# # Plot the waveform
# plt.figure(figsize=(10, 4))
# plt.plot(time, values)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Waveform from output.txt')
# plt.grid(True)
# plt.tight_layout()
# plt.show()