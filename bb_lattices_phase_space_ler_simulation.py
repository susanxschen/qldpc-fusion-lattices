"""
Script to simulate BB code lattices under entire grid of error/erasure phase space to get logical error rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys
import ctypes
import scipy
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag, csr_matrix

class UFDecoder:
    def __init__(self, h):  # parity-check matrix h can be scipy coo_matrix, csr_matrix, or numpy array
        self.h = h
        self.n_syndr = self.h.shape[0]  # number of parity checks
        self.n_qbt = self.h.shape[1]  # number of data qubits
        if type(h) != np.ndarray and h.getformat() == 'coo':
            cnt = np.zeros(self.n_syndr, dtype=np.uint8)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint8)  # count number of parity checks per qubit
            for i in self.h.row:
                cnt[i] += 1
            for i in self.h.col:
                cnt_qbt[i] += 1
        elif type(h) != np.ndarray and h.getformat() == 'csr':
            cnt = np.zeros(self.n_syndr, dtype=np.uint8)  # count number of qubits per parity check
            cnt_qbt = np.zeros(self.n_qbt, dtype=np.uint8)  # count number of parity checks per qubit
            for row in range(self.h.shape[0]):
                cnt[row] = len(h.getrow(row).indices)
                for c in h.getrow(row).indices:
                    cnt_qbt[c] += 1
        elif type(h) == np.ndarray:
            cnt_qbt = np.sum(h, axis=0, dtype=np.uint8)
            cnt = np.sum(h, axis=1, dtype=np.uint8)
        else:
            print('invalid parity check matrix')
        self.num_nb_max_syndr = cnt.max()  # maximum number of qubits per parity check
        self.num_nb_max_qbt = cnt_qbt.max()  # maximum number of parity checks per qubit
        self.nn_syndr = np.zeros(self.n_syndr * int(self.num_nb_max_syndr), dtype=np.int32)
        self.nn_qbt = np.zeros(self.n_qbt * int(self.num_nb_max_qbt), dtype=np.int32)
        self.len_nb = np.zeros(self.n_syndr + self.n_qbt, dtype=np.uint8)
        self.correction = np.zeros(self.n_qbt, dtype=np.uint8)
        self.h_matrix_to_tanner_graph()

    def add_from_h_row_and_col(self, r, c):
        self.nn_syndr[r * int(self.num_nb_max_syndr) + int(self.len_nb[r + self.n_qbt])] = c
        self.nn_qbt[c * int(self.num_nb_max_qbt) + int(self.len_nb[c])] = r + self.n_qbt
        self.len_nb[r + self.n_qbt] += 1
        self.len_nb[c] += 1

    def h_matrix_to_tanner_graph(self):
        if type(self.h) != np.ndarray and self.h.getformat() == 'coo':
            for i in range(len(self.h.col)):
                c = self.h.col[i]
                r = self.h.row[i]
                self.add_from_h_row_and_col(r, c)
        elif type(self.h) != np.ndarray and self.h.getformat() == 'csr':
            for r in range(self.h.shape[0]):
                for c in self.h.getrow(r).indices:
                    self.add_from_h_row_and_col(r, c)
        elif type(self.h) == np.ndarray:
            for r in range(self.h.shape[0]):
                for c in range(self.h.shape[1]):
                    if self.h[r, c]:
                        self.add_from_h_row_and_col(r, c)

decode_lib = ctypes.cdll.LoadLibrary('../build/libSpeedDecoder.so')



decode_lib.ldpc_collect_graph_and_decode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint8, ctypes.c_uint8, 
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p]



decode_lib.ldpc_collect_graph_and_decode_batch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint8, ctypes.c_uint8, 
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

# Wrapper for LDPC decoder 
def run_ldpc_decoder(decoder, a_syndrome, a_erasure):
    decode_lib.ldpc_collect_graph_and_decode(ctypes.c_int(decoder.n_qbt),
        ctypes.c_int(decoder.n_syndr), ctypes.c_uint8(decoder.num_nb_max_qbt), ctypes.c_uint8(decoder.num_nb_max_syndr),
        ctypes.c_void_p(decoder.nn_qbt.ctypes.data), ctypes.c_void_p(decoder.nn_syndr.ctypes.data), ctypes.c_void_p(decoder.len_nb.ctypes.data),
        ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data),  ctypes.c_void_p(decoder.correction.ctypes.data)
    )


# Wrapper for LDPC decoder (batch decoding)
def run_ldpc_decoder_batch(decoder, a_syndrome, a_erasure):
    decode_lib.ldpc_collect_graph_and_decode_batch(
        ctypes.c_int(decoder.n_qbt), ctypes.c_int(decoder.n_syndr), ctypes.c_uint8(decoder.num_nb_max_qbt), ctypes.c_uint8(decoder.num_nb_max_syndr),
        ctypes.c_void_p(decoder.nn_qbt.ctypes.data), ctypes.c_void_p(decoder.nn_syndr.ctypes.data), ctypes.c_void_p(decoder.len_nb.ctypes.data),
        ctypes.c_void_p(a_syndrome.ctypes.data), ctypes.c_void_p(a_erasure.ctypes.data), ctypes.c_void_p(decoder.correction.ctypes.data),
        ctypes.c_int(nrep)
    )



### LOAD ALL BICYCLE CODE LATTICE MATRICES ### 

hx_72 = np.loadtxt(f"bicycle code lattices/hx_lattice_72.txt").astype(int)
hx_90 = np.loadtxt(f"bicycle code lattices/hx_lattice_90.txt").astype(int)
hx_108 = np.loadtxt(f"bicycle code lattices/hx_lattice_108.txt").astype(int)
hx_144 = np.loadtxt(f"bicycle code lattices/hx_lattice_144.txt").astype(int)

lx_72 = np.loadtxt(f"bicycle code lattices/lx_lattice_72.txt").astype(int)
lx_90 = np.loadtxt(f"bicycle code lattices/lx_lattice_90.txt").astype(int)
lx_108 = np.loadtxt(f"bicycle code lattices/lx_lattice_108.txt").astype(int)
lx_144 = np.loadtxt(f"bicycle code lattices/lx_lattice_144.txt").astype(int)


#############

d_codes = [6,10,10,12]
hx_list = [hx_72, hx_90, hx_108, hx_144]
lx_list = [lx_72, lx_90, lx_108, lx_144]
n_list = [72,90,108,144]
k_list = [12,8,8,12]
######


if __name__ == "__main__":

    def num_decoding_failures(decoder, hx, logicals, p, num_rep):
        p_err = p[0]
        p_erase = p[1]
        H = hx
        n_err = 0
        n_qbt = H.shape[1]

        for i in range(num_rep):

            erasure = np.random.binomial(1, p_erase, n_qbt).astype(np.uint8)
            error_pauli = np.random.binomial(1, p_err, n_qbt).astype(np.uint8)

            noise = np.logical_or(np.logical_and(np.logical_not(erasure), error_pauli), 
                                  np.logical_and(erasure, np.random.binomial(1, 0.5, n_qbt))).astype(np.uint8)

            syndrome = (H @ noise % 2).astype(np.uint8)
    
            run_ldpc_decoder(decoder, syndrome, erasure)

            # residual error
            error = np.logical_xor(noise, decoder.correction)
            if np.any(error @ logicals.T % 2):
                n_err += 1

        return n_err


    num_trials = 100
    print("Number of trials:", num_trials)

    p_err_list = np.linspace(0.000001,0.003,15)
    p_erasure_list = np.linspace(0.001,0.1,15) 
    
    t0 = time.perf_counter()
    a_error = np.zeros((len(d_codes), len(p_err_list), len(p_erasure_list)), dtype=np.double)
    for i in range(len(d_codes)):
        print(datetime.now().time())
        Hx = hx_list[i]
        lx = lx_list[i]     

        decoder = UFDecoder(Hx)
        print("n =", n_list[i])
    
        for i_p_err, p_err in enumerate(p_err_list):
            for i_p_erasure, p_erasure in enumerate(p_erasure_list):
                p = [p_err,p_erasure]
                num_errors = num_decoding_failures(decoder, Hx, lx, p, num_trials)
                a_error[i,i_p_err,i_p_erasure] = num_errors # for HPC sims, make this just num_errors

    t1 = time.perf_counter()
    print('time (mins): ', (t1 - t0)/60)

    data72 = a_error[0, :, :]
    data90 = a_error[1, :, :]
    data108 = a_error[2, :, :]
    data144 = a_error[3, :, :]

    # save data
    slurm_array_id = sys.argv[1]
#     np.savetxt(f"2d_grid_data/l_errs_72code_{slurm_array_id}.txt", data72)
#     np.savetxt(f"2d_grid_data/l_errs_90code_{slurm_array_id}.txt", data90)
#     np.savetxt(f"2d_grid_data/l_errs_108code_{slurm_array_id}.txt", data108)
#     np.savetxt(f"2d_grid_data/l_errs_144code_{slurm_array_id}.txt", data144)