import numpy as np
# THIS is OUTDATED. use reconv3 or 2nd order simulation

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
# Read the file and split into float values
with open('sin200B.txt', 'r') as f:
    data = f.read().split()
f.close()

# Convert to floats and take the first 10,000 elements
values = np.array(data[:5000000], dtype=float)


class MovingAverage:
    def __init__(self,window_size):
        self.window_size=window_size
        self.buffer = [0.0] * window_size
        self.index = 0
        self.count = 0
        self.sum = 0.0

    def update(self, new_sample: float) -> float:
        """
        Add a new sample, drop the oldest, and return the current average.
        Mirrors the C code logic exactly.
        """
        # subtract the value we're about to overwrite
        self.sum -= self.buffer[self.index]

        # insert the new sample
        self.buffer[self.index] = new_sample
        self.sum += new_sample

        # advance the circular index
        self.index = (self.index + 1) % self.window_size

        # keep track of how many valid samples we have (up to MA_WINDOW_SIZE)
        if self.count < self.window_size:
            self.count += 1

        # return average over the actual number of samples so far
        return self.sum / self.count


# values = values - np.mean(values)
class UnlimitedSamplerReconstructor:
    def __init__(self, lam):
        self.lam = lam
        self.prev_y = None
        self.prev_eps = 0  # ε[k-1]
        self.reconstructed = []
        self.order=1
        self.sampling_interval=2e-2
        self.anti_diff_buffer=[0]*2
        self.diff_buffers=[0]*2



    def modulo(self, x):
        """Centered modulo into [-λ, λ)"""
        return ((x + self.lam) % (2 * self.lam)) - self.lam
    
    def anti_diff_2nd(self,diffN,ma1,ma2,use_filter) :  

        # First integration stage
        x = diffN

        x = self.anti_diff_buffer[0] + x
        x = 2.0 * self.lam * round(x / (2.0 * self.lam))
        self.anti_diff_buffer[0] = x

        if use_filter:
            avg1 = ma1.update(x)                          # updateMovingAverage(ma1, x)      
            x = x - avg1

        # Second (final) integration stage
        idx = self.order - 1
        x = self.anti_diff_buffer[idx] + x
        x = 2.0 * self.lam * round(x / (2.0 * self.lam))

        self.anti_diff_buffer[idx] = x

        if use_filter:
            avg2 = ma2.update(x)                          # updateMovingAverage(ma2, x)
            x = x - avg2


        return x
    def nth_order_difference(self,new_sample):
        """
        Compute the N-th order finite difference using binomial coefficients.
        
        Parameters:
        - window: list or array of length N+1 containing the most recent samples [f[k], ..., f[k+N]]

        Returns:
        - N-th order finite difference at position k
        """
        x = new_sample
        for i in range(self.order):

            delta = x - self.diff_buffers[i]
            self.diff_buffers[i] = x
            x = delta
        return x
    
    def anti_diff_nth(self, diffN):
        """
        Feed in a new Δ^N value, return reconstructed f[k] sample.
        Returns None until sufficient initialization is done.
        """
        x = diffN
        for i in range(self.order-1):
            x = self.anti_diff_buffers[i] + x
            x=2 * self.lam * np.round(x / (2 * self.lam))
            self.anti_diff_buffers[i] = x  # update stored value

        x = self.anti_diff_buffers[self.order-1] + x
        self.anti_diff_buffers[self.order-1] = x 

        return self.anti_diff_buffers[self.order-1]  # This is f[k]

    



    def update(self, yk,ma1,ma2,use_filter):

        self.order=2


        # Step 1: Compute Δy[k]

        dy=self.nth_order_difference(yk)

        # Step 2: Apply modulo to Δy[k]
        mod_dy = self.modulo(dy)

        # Step 3: Estimate Δε[k] = mod(Δy[k]) - Δy[k]
        delta_eps = mod_dy - dy

        # Step 4: ε[k] = ε[k-1] + Δε[k], rounded to nearest 2λ

        eps_k = self.anti_diff_2nd(delta_eps,ma1,ma2,use_filter)


        
        # eps_k = 2 * self.lam * np.round(self.prev_eps / (2 * self.lam))

        # Step 5: Recover g[k] = y[k] + ε[k]
        gk = yk + eps_k

   
    
        return gk




def print_output_graph(input_array,timestep=2e-9,downsample_val=10000,L=0.75,use_filter=False):

    reconstructor = UnlimitedSamplerReconstructor(L)
    ma1=MovingAverage(100)
    ma2=MovingAverage(100)
 
    processed=[0]*(len(input_array))
    
    for i in range(len(input_array)):

        if i%downsample_val ==0:

            processed[i]=reconstructor.update(input_array[i],ma1,ma2,use_filter)


    time = np.arange(len(input_array)) * (timestep)

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, processed)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform from output.txt')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


print_output_graph(values)
