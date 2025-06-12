import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
# Read the file and split into float values
with open('sin200B.txt', 'r') as f:
    data = f.read().split()
f.close()

# with open('sin100.txt', 'r') as f:
#     original_data = f.read().split()
# f.close()

test_in=[0,0,1,0,1,1]
# Convert to floats and take the first 10,000 elements
values = np.array(data[:5000000], dtype=float)
# original_values=np.array(original_data[:10000000], dtype=float)

MA_WINDOW_SIZE = 100  # Size of the moving average window

class MovingAverage:
    def __init__(self):
        self.buffer = [0.0] * MA_WINDOW_SIZE
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
        self.index = (self.index + 1) % MA_WINDOW_SIZE

        # keep track of how many valid samples we have (up to MA_WINDOW_SIZE)
        if self.count < MA_WINDOW_SIZE:
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
        # self.max=0.1
        # self.max_order=10
        # self.diff_buffers=[0]*(self.max_order+1)
        # self.anti_diff_buffers=[0]*(self. max_order+1)
        # self.J = 5  # window length, tune this based on λ and dynamic range
        # self.res_buffer = deque(maxlen=self.J)  # buffer of residuals
        # self.cumsum1 = 0.0
        # self.cumsum2 = 0.0
        # self.J = 10  # Window size for κ(n) estimation

        # self.kappa_s1 = [0.0 for _ in range(self.max_order)]
        # self.kappa_s2 = [0.0 for _ in range(self.max_order)]
        # self.kappa_hist = [deque([0.0] * self.J, maxlen=self.J) for _ in range(self.max_order)]




    def calc_order(self):
        N=int(math.ceil((math.log(self.lam)-math.log(self.max))/math.log(self.sampling_interval*math.pi*math.e)))

        return max(N,1)
    def estimate_reset_count(self, new_residual, Bg):
        """
        Estimate reset count using a rolling buffer and double sum.
        This implements Kn_iter = floor((-S2[J] + S2[0]) / (12 * Bg) + 0.5)
        """

        self.res_buffer.append(new_residual)

        # Update running nested sum
        self.cumsum1 += new_residual
        self.cumsum2 += self.cumsum1

        # Need full buffer to do correction
        if len(self.res_buffer) < self.J:
            return 0  # No correction until enough data

        # Approximate S2[0] and S2[J]
        s2_0 = self.cumsum2
        s2_J = 0
        tmp_cumsum1 = 0
        for r in self.res_buffer:
            tmp_cumsum1 += r
            s2_J += tmp_cumsum1

        Kn = int(np.floor((s2_0 - s2_J) / (12 * Bg) + 0.5))
        return Kn


    def modulo(self, x):
        """Centered modulo into [-λ, λ)"""
        return ((x + self.lam) % (2 * self.lam)) - self.lam
    

    def nth_order_difference(self,new_sample):
        """
        Compute the N-th order finite difference using binomial coefficients.
        
        Parameters:
        - window: list or array of length N+1 containing the most recent samples [f[k], ..., f[k+N]]

        Returns:
        - N-th order finite difference at position k
        """
        x = new_sample
        for i in range(self.max_order+1):
            if self.diff_buffers[i] is None:
                self.diff_buffers[i] = x
                return None  # not enough data yet
            else:
                delta = x - self.diff_buffers[i]
                self.diff_buffers[i] = x
                x = delta
        return self.diff_buffers[self.order]
    
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
            # if abs(x) > 10 * self.lam:
            #     x = 0.0
            #     self.anti_diff_buffers[i] = 0.0
            # x+=2*self.lam*(6*self.max/self.lam+1)

        x = self.anti_diff_buffers[self.order-1] + x
        self.anti_diff_buffers[self.order-1] = x 

        return self.anti_diff_buffers[self.order-1]  # This is f[k]
    
    def anti_diff_nth_kn(self, diffN):
        """
        Feed in a new Δ^N value, return reconstructed f[k] sample.
        Returns None until sufficient initialization is done.
        """
        x = diffN
        for i in range(self.order-1):
            x = self.anti_diff_buffers[i] + x
            x=2 * self.lam * np.round(x / (2 * self.lam))
            self.anti_diff_buffers[i] = x  # update stored value
            self.kappa_s1[i] += x
            self.kappa_s2[i] += self.kappa_s1[i]
            self.kappa_hist[i].append(self.kappa_s2[i])

            kappa = 0
            # if len(self.kappa_hist[i]) == self.J:
            s2_now = self.kappa_hist[i][-1]
            s2_past = self.kappa_hist[i][0]
            beta_g = self.max  # Replace with conservative upper bound if needed

            if beta_g > 0:
                denom = 12 * beta_g / self.lam
                raw = (s2_now - s2_past) / denom
                kappa = int(np.floor(raw + 0.5))  # Round to nearest integer

                    # Optional: clamp for robustness
                    # if abs(kappa) > 10:
                    #     kappa = 0

            # Step 8: Apply 2λ * κ(n)
            x += 2 * self.lam * kappa


        x = self.anti_diff_buffers[self.order-1] + x
        self.anti_diff_buffers[self.order-1] = x 

        return self.anti_diff_buffers[self.order-1]  # This is f[k]
    
    



    def update(self, yk):
        """Process one new modulo sample y[k] and return g[k] estimate."""
        if abs(yk)>self.max:
            self.max=2*self.lam*np.round(abs(yk)/(2*self.lam))
            # print(self.max)
            # self.order=self.calc_order()
            # print(self.order)
        self.order=2
        # Bg = max(abs(yk), self.max)
        Bg = np.ceil(self.max / (2 * self.lam)) * 2 * self.lam


        



        # if self.prev_y is None:
        #     # First sample: cannot compute difference, assume ε[0] = 0
        #     gk = yk
        #     self.prev_y = yk
            
        #     return gk

        # Step 1: Compute Δy[k] = y[k] - y[k-1]
        # dy = yk - self.prev_y
        # self.prev_y=yk
        dy=self.nth_order_difference(yk)

        # Step 2: Apply modulo to Δy[k]
        mod_dy = self.modulo(dy)

        # Step 3: Estimate Δε[k] = mod(Δy[k]) - Δy[k]
        delta_eps = mod_dy - dy
        # Kn = self.estimate_reset_count(delta_eps, Bg)
        # delta_eps += 2 * Kn * self.lam  
        # if abs(delta_eps) > 10 * self.lam:
        #     delta_eps = 0.0

        # Step 4: ε[k] = ε[k-1] + Δε[k], rounded to nearest 2λ
        # self.prev_eps = self.prev_eps + delta_eps
        self.prev_eps = self.anti_diff_nth(delta_eps)

        # if abs(self.prev_eps) > 10 * self.lam:
        #     self.prev_eps = 0.0

        
        eps_k = 2 * self.lam * np.round(self.prev_eps / (2 * self.lam))

        # Step 5: Recover g[k] = y[k] + ε[k]
        gk = yk + eps_k

        # Update state
        
        self.prev_eps = eps_k
    
        return gk


    # print(f"Input: {yk:.2f}, Reconstructed: {gk:.2f}")
# def DO_US(T,L, Bg):
#    return np.ceil((np.log(L) - np.log(Bg))/np.log(T*np.pi*np.exp(1)))

def print_output_graph(input_array,timestep,downsample_val=1000,L=0.15):
    # processed=unwrap(input_array,L)5
    reconstructor = UnlimitedSamplerReconstructor(L)
    # processed=reconstruct_unlimited_samples(input_array,L,np.pi,T = 1 / (2 * np.pi * np.e),beta=2)
    processed=[0]*len(input_array)
    avg=0
    for i in range(len(input_array)):

        if i%downsample_val ==0:

        # processed[i]=higher_order_difference(input_array[i])
        # processed[i]=modulo_residual(processed[i],L)
            processed[i]=reconstructor.update(input_array[i])


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

def simple_test(input_array,L):
    reconstructor = UnlimitedSamplerReconstructor(L)
    for i in input_array:
        print(reconstructor.update(i))

print_output_graph(values,2e-9)
# simple_test(test_in,0.75)
# time = np.arange(len(original_data)) * (2e-9)
# plt.figure(figsize=(10, 4))
# plt.plot(time, original_data)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('original data')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
