import numpy as np
# THIS is OUTDATED. use reconv3 or 2nd order simulation

import numpy as np
import matplotlib.pyplot as plt
# Read the file and split into float values
with open('sin100B.txt', 'r') as f:
    data = f.read().split()

# Convert to floats and take the first 10,000 elements
values = np.array(data[:10000000], dtype=float)

values = values - np.mean(values)
class SecondOrderReconstructor:
    def __init__(self, lam, drift_window=5, apply_drift_correction=True):
        self.lam = lam
        self.y_buffer = []        # Store last 3 modulo samples
        self.eps_diff1 = 0.0      # First-order ε[k-1]
        self.eps_k = 0.0          # ε[k-1]
        self.eps_hist = []        # ε history for drift correction
        self.drift_window = drift_window
        self.apply_drift_correction = apply_drift_correction
        self.reconstructed = []

    def modulo(self, x):
        """Centered modulo into [-λ, λ)"""
        return ((x + self.lam) % (2 * self.lam)) - self.lam

    def correct_drift(self):
        """Optional: Estimate and remove drift from ε history."""
        if len(self.eps_hist) < self.drift_window + 1:
            return 0.0  # Not enough data for correction

        i = -self.drift_window - 1
        j = -1
        delta = (self.eps_hist[i] - self.eps_hist[j]) / (2 * self.lam * self.drift_window)
        kappa = round(delta)
        correction = 2 * self.lam * kappa
        return correction

    def update(self, yk):
        self.y_buffer.append(yk)
        if len(self.y_buffer) < 3:
            # Not enough samples yet for Δ²
            # self.reconstructed.append(yk)
            return yk

        # Step 1: Compute second-order finite difference Δ²y[k]
        yk2, yk1, yk0 = self.y_buffer[-3:]
        d2y = yk0 - 2 * yk1 + yk2

        # Step 2: Estimate Δ²ε[k]
        mod_d2y = self.modulo(d2y)
        d2eps = mod_d2y - d2y

        # Step 3: Anti-diff twice without rounding
        self.eps_diff1 += d2eps
        self.eps_k += self.eps_diff1

        # Step 4: Optional drift correction
        if self.apply_drift_correction:
            self.eps_hist.append(self.eps_k)
            correction = self.correct_drift()
        else:
            correction = 0.0

        # Step 5: Final rounding once, after full anti-diff + drift correction
        eps_corrected = self.eps_k + correction
        eps_rounded = 2 * self.lam * np.round(eps_corrected / (2 * self.lam))

        # Step 6: Reconstruct g[k] = y[k] + ε[k]
        gk = yk + eps_rounded


        return gk

mod_samples = [0.5, -0.8, -0.9, 0.2, -0.7, 0.1]  # simulated modulo samples
lam = 1.0



    # print(f"Input: {yk:.2f}, Reconstructed: {gk:.2f}")


def print_output_graph(input_array,timestep,downsample_val=500,L=0.75):
    # processed=unwrap(input_array,L)
    reconstructor = SecondOrderReconstructor(L)
    # processed=reconstruct_unlimited_samples(input_array,L,np.pi,T = 1 / (2 * np.pi * np.e),beta=2)
    processed=[0]*len(input_array)
    avg=0
    for i in range(len(input_array)):

        if i%downsample_val ==0:

        # processed[i]=higher_order_difference(input_array[i])
        # processed[i]=modulo_residual(processed[i],L)
            processed[i]=reconstructor.update(input_array[i])


    time = np.arange(len(processed)) * (timestep*downsample_val)

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


print_output_graph(values,2e-9)