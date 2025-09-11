"""
Script to get logical error rates for BB code fusion lattices under i.i.d. fusion errors and erasures with modified UF decoder.
Sweep line of constant polar angle theta.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import multiprocessing
from itertools import repeat, product
import sys
import ctypes
import scipy
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag, csr_matrix
from some_codes import toric_code


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

hx_72 = np.loadtxt(f"bicycle code dual lattices/hz_lattice_72.txt").astype(int)
hx_90 = np.loadtxt(f"bicycle code dual lattices/hz_lattice_90.txt").astype(int)
hx_108 = np.loadtxt(f"bicycle code dual lattices/hz_lattice_108.txt").astype(int)
hx_144 = np.loadtxt(f"bicycle code dual lattices/hz_lattice_144.txt").astype(int)

lx_72 = np.loadtxt(f"bicycle code dual lattices/lz_lattice_72.txt").astype(int)
lx_90 = np.loadtxt(f"bicycle code dual lattices/lz_lattice_90.txt").astype(int)
lx_108 = np.loadtxt(f"bicycle code dual lattices/lz_lattice_108.txt").astype(int)
lx_144 = np.loadtxt(f"bicycle code dual lattices/lz_lattice_144.txt").astype(int)


#############

d_codes = [6,10,10,12]
hx_list = [hx_72, hx_90, hx_108, hx_144]
lx_list = [lx_72, lx_90, lx_108, lx_144]
n_list = [72,90,108,144]
k_list = [12,8,8,12]
######

if __name__ == "__main__":
    """ Scan 2D range via angle theta, 
    set maximum and minimum error and loss probabilities, 
    and create the error and loss probability combinations per angle:"""

    x = np.linspace(0,1,20) 
    min_err, max_err = 0.00005, 0.003
    min_loss, max_loss = 0.01, 0.1

    theta = np.float64(sys.argv[2]) * np.pi/2
    p_err_list, p_erasure_list = (x*(max_err-min_err) + min_err)*np.cos(theta), (x*(max_loss-min_loss) + min_loss)*np.sin(theta)
    p_list = [[i,j] for i,j in zip(p_err_list, p_erasure_list)]
    
    num_trials = 1000
    print("Number of trials:", num_trials)

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
    
    def get_logerrs_parallel(decoder, hx, logicals, p_list, num_rep):
    
        with multiprocessing.Pool() as pool:
            logerrs = pool.starmap(num_decoding_failures,
                                        [(decoder, hx, logicals, p, num_rep) for p in p_list])
        return np.array(logerrs)

    t0 = time.perf_counter()
    l_errs_all_codes = []
    for i in range(len(d_codes)):
        Hx = hx_list[i]
        lx = lx_list[i]     
 
        decoder = UFDecoder(Hx)
        print("n =", n_list[i])
    
        l_errs = get_logerrs_parallel(decoder, Hx, lx, p_list, num_trials)
        l_errs_all_codes.append(l_errs)

        print(l_errs)
    t1 = time.perf_counter()
    print('time (mins): ', (t1 - t0)/60)

#     # save data
#     slurm_array_id = sys.argv[1]
#     np.savetxt(f"bb_data_20points/angle_{np.float64(sys.argv[2])}_l_errs_{slurm_array_id}.txt", l_errs_all_codes)

