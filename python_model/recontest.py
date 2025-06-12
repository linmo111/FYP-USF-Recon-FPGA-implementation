# THIS is OUTDATED. use reconv3 or 2nd order simulation


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
N = 2000
LAMBDA = 0.75
tau = 2 * np.pi
T = tau / N
M = 6
n = np.arange(N)
t = T * n

with open('sin100B.txt', 'r') as f:
    data = f.read().split()

yn = np.array(data[:10000], dtype=float)

with open('sin100.txt', 'r') as f:
    data = f.read().split()

gn = np.array(data[:10000], dtype=float)

resn = gn - yn
resn_diff = np.diff(resn)
n_k_gt = np.where(np.abs(resn_diff) > LAMBDA)[0]
a_k_gt = resn_diff[n_k_gt]
K = len(n_k_gt)

M_OFFSET = 3
spect_offset = M + M_OFFSET
yn_diff = np.diff(yn)
yn_d_dft = np.fft.fft(yn_diff)
res_dft = yn_d_dft[spect_offset: -spect_offset]
m = np.arange(spect_offset, N - spect_offset - 1)

toep_mtx = toeplitz(-np.flip(res_dft))
A = toep_mtx[: -K, -K:]
b = -res_dft[K:]
hn = np.linalg.lstsq(A, b, rcond=None)[0]
hn = np.concatenate(([1], hn))
z_roots = np.roots(hn)
log_mod = np.mod(-np.imag(np.log(z_roots)) + 2 * np.pi, 2 * np.pi)
n_k_hat = log_mod * (N - 1) / (2 * np.pi)
exps = np.exp(-1j * 2 * np.pi * m[:, None] * n_k_hat / (N - 1))
a_k_hat = np.linalg.lstsq(exps, -res_dft, rcond=None)[0]

res_diff_rec = np.zeros_like(yn_diff, dtype=complex)
res_diff_rec[np.round(n_k_hat).astype(int)] = a_k_hat
res_rec = np.cumsum(np.concatenate(([0], res_diff_rec)))
n_k_hat_srt = np.sort(n_k_hat)
srt_idx = np.argsort(n_k_hat)
a_k_hat_srt = a_k_hat[srt_idx]


gn_rec = res_rec + yn
# gn_MSE = np.mean(np.abs(gn_rec - gn) ** 2)
# print(f"g_n MSE: {gn_MSE}")

plt.figure()

plt.plot(t, np.real(gn_rec), "--", color="red", label="Reconstruction")
plt.xlim([t.min(), t.max()])
plt.xlabel("Time (s)")
plt.legend()
plt.show()