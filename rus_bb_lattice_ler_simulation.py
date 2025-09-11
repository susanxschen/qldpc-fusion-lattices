"""
Script to get logical error rates with respect to photon loss for BB code fusion lattices with modified repeat-until-success strategy.
"""


import numpy as np
from Fusion_lattice import *
from numpy import random
import matplotlib.pyplot as plt
from linear_algebra_inZ2 import * 
import multiprocessing
from itertools import repeat, product
import random
import sys

hx_72=np.loadtxt(f"bicycle code examples/n_72_lm_{6,6}_hx.txt").astype(int)
lx_72=np.loadtxt(f"bicycle code examples/n_72_lm_{6,6}_lx.txt").astype(int)
hz_72=np.loadtxt(f"bicycle code examples/n_72_lm_{6,6}_hz.txt").astype(int)
lz_72=np.loadtxt(f"bicycle code examples/n_72_lm_{6,6}_lz.txt").astype(int)

hx_90=np.loadtxt(f"bicycle code examples/n_90_lm_{15,3}_hx.txt").astype(int)
lx_90=np.loadtxt(f"bicycle code examples/n_90_lm_{15,3}_lx.txt").astype(int)
hz_90=np.loadtxt(f"bicycle code examples/n_90_lm_{15,3}_hz.txt").astype(int)
lz_90=np.loadtxt(f"bicycle code examples/n_90_lm_{15,3}_lz.txt").astype(int)

hx_108=np.loadtxt(f"bicycle code examples/n_108_lm_{9,6}_hx.txt").astype(int)
lx_108=np.loadtxt(f"bicycle code examples/n_108_lm_{9,6}_lx.txt").astype(int)
hz_108=np.loadtxt(f"bicycle code examples/n_108_lm_{9,6}_hz.txt").astype(int)
lz_108=np.loadtxt(f"bicycle code examples/n_108_lm_{9,6}_lz.txt").astype(int)

hx_144=np.loadtxt(f"bicycle code examples/n_144_lm_{12,6}_hx.txt").astype(int)
lx_144=np.loadtxt(f"bicycle code examples/n_144_lm_{12,6}_lx.txt").astype(int)
hz_144=np.loadtxt(f"bicycle code examples/n_144_lm_{12,6}_hz.txt").astype(int)
lz_144=np.loadtxt(f"bicycle code examples/n_144_lm_{12,6}_lz.txt").astype(int)

d_codes = [6,10,10,12]
hx_codes_list = [hx_72, hx_90, hx_108, hx_144]
hz_codes_list = [hz_72, hz_90, hz_108, hz_144]
lx_codes_list = [lx_72, lx_90, lx_108, lx_144]
lz_codes_list = [lz_72, lz_90, lz_108, lz_144]
n_list = [72,90,108,144]

lattice_list = []
for i in range(len(d_codes)):
    lattice = Fusion_lattice(hx_codes_list[i], hz_codes_list[i], 6,6, d_codes[i], lx_codes_list[i], lz_codes_list[i])
    lattice_list.append(lattice)

hx_lattice_list = []
lx_lattice_list = []
hz_lattice_list = []
lz_lattice_list = []
for i in range(len(lattice_list)):
    hx_lattice = lattice_list[i].get_matching_matrix()
    lx_lattice = lattice_list[i].get_logical_x()
    hx_lattice_list.append(hx_lattice)
    lx_lattice_list.append(lx_lattice)

    hz_lattice = lattice_list[i].get_dual_matching_matrix()
    lz_lattice = lattice_list[i].get_logical_z()
    hz_lattice_list.append(hz_lattice)
    lz_lattice_list.append(lz_lattice)


outcomes = [[0,0],[1,1],[0,1],[1,0]]

# define fusion probabilities in terms of physical loss 
def p_fusion_succ(p_loss):
    eta = 1-p_loss
    return 0.5*(eta**2)
def p_fusion_fail(p_loss):
    eta = 1-p_loss
    return 0.5*(eta**2)
def p_fusion_loss(p_loss):
    eta = 1-p_loss
    return 1-(eta**2)

# define encoded fusion probabilities (from which we do first stage sampling)

def p_logical_succ(p_loss, N):
    return p_fusion_succ(p_loss) + p_fusion_succ(p_loss)*sum((p_fusion_fail(p_loss)**i) for i in range(1, N))

def std_log_loss(p_loss, N):
    return p_fusion_loss(p_loss)*sum((p_fusion_fail(p_loss)**i) for i in range(N))
def std_log_xz_only(p_loss,N):
    return 1 - (p_logical_succ(p_loss,N)+std_log_loss(p_loss,N)) 
def std_log_zx_only(p_loss, N):
    return 0

def probabilities_std(p_loss, N): # standard RUS strategy probabilities
    prob = [p_logical_succ(p_loss,N), std_log_loss(p_loss,N), 
                     std_log_xz_only(p_loss,N), std_log_zx_only(p_loss,N)] # with these probabilities you get above outcomes
    return prob

# defining what will be primal and dual outcomes
xz_in_primal_list = []
for k in range(len(d_codes)):
    xz_in_primal = [0] * lattice_list[k].num_total_fusions # if in primal, outcome is XZ then 1, else 0
    for i in range(lattice_list[k].num_ancilla):
        for j in range(lattice_list[k].check_degree):
            for t in range(lattice_list[k].T):
                index_z = lattice_list[k].z_fusion_ind(i,j,t) # all the Z-fusions are XZ in primal
                xz_in_primal[index_z] = 1 
    xz_in_primal_list.append(np.array(xz_in_primal)) # this will give you all x layers are zx in primal and z layers are xz in primal

def sampling(p_loss, H_with_logop, lattice, xz_in_primal, N):
    # N is max tries 
    # lattice is the class instance of the fusion network
    num_fusions = H_with_logop.shape[1]
    fusion_outcomes = random.choices(outcomes, weights=probabilities_std(p_loss,N), k=num_fusions) # sample based on standard RUS model weights
    fusion_outcomes_temp = [[i, j, 0, idx] for idx, (i, j) in enumerate(fusion_outcomes)] # this adds a temporal seen index and labels fusions
    fusions_list = lattice.lattice_fusions # list with detailed info of each fusion
    
    num_layers = 2*lattice.T
    num_fusions_per_layer = int(len(fusion_outcomes)/num_layers)
    # layers creates layer sub-lists within fusion outcomes list 
    layers = [fusion_outcomes_temp[i:i + num_fusions_per_layer] for i in range(0, len(fusion_outcomes_temp), num_fusions_per_layer)]

    # randomly shuffle within each layer and record the permutations within each layer 
    shuffled_layers = []
    permutations = [] 
    
    for layer in layers:
        fusion_indices = list(range(len(layer)))
        random.shuffle(fusion_indices)               
        permutations.append(fusion_indices)
        shuffled_layers.append([layer[i] for i in fusion_indices])

    # go through each layer sub-list and conditionally update fusion outcomes
    for i in range(len(shuffled_layers)):
        for j in range(len(shuffled_layers[i])):

            if i % 2 == 0: # a.k.a. primal layer
                fusion = shuffled_layers[i][j]
                index = fusion[3]
                if fusion[2] == 1: # if this fusion has already been seen
                    pass
                else:
                    # fusion[2] = 1 # have to set it to seen for all cases 
                    outcome = fusion[:2]
                    if outcome == [0,0]:
                        fusion[2] = 1
                    elif outcome == [1,1]: # find unseen fusions that either are connected to data or ancilla and make successful
                        fusion[2] = 1
                        connected_fusions = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if (
                                        (q == fusions_list[index][1] and fusion_type == fusions_list[index][2] and time == fusions_list[index][3])
                                        or (i == fusions_list[index][0] and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]))]
                        for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in connected_fusions: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful 
                                    relevant_fusion[2] = 1
                    else:
                        if random.randint(0, 1) == 0: # we measure ancilla, set fusion to lost, make fusions involving ancilla successful
                            shuffled_layers[i][j][:2] = [1,1]
                            fusion[2] = 1
                            fusions_same_ancilla = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if i == fusions_list[index][0] 
                                                     and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]]
                            for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in fusions_same_ancilla: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful 
                                    relevant_fusion[2] = 1
                        else: # we measure data, preserve primal cells and make all fusions involving data successful
                            shuffled_layers[i][j][:2] = [0,0]
                            fusion[2] = 1
                            fusions_same_data = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if q == fusions_list[index][1] 
                                                     and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]]
                            for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in fusions_same_data: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful 
                                    relevant_fusion[2] = 1
    
        
            elif i % 2 == 1: # a.k.a. dual layer 
                fusion = shuffled_layers[i][j]
                index = fusion[3]
                if fusion[2] == 1: # if this fusion has already been seen
                    pass
                else:
                    outcome = fusion[:2]
                    if outcome == [0,0]:
                        fusion[2] = 1
                    elif outcome == [1,1]: # find unseen fusions that either are connected to data or ancilla and make successful
                        fusion[2] = 1
                        connected_fusions = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if (
                                        (q == fusions_list[index][1] and fusion_type == fusions_list[index][2] and time == fusions_list[index][3])
                                        or (i == fusions_list[index][0] and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]))]
                        for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in connected_fusions: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful
                                    relevant_fusion[2] = 1
    
                    else:
                        if random.randint(0, 1) == 0: # we measure data, set fusion to lost, make fusions involving data successful
                            shuffled_layers[i][j][:2] = [1,1]
                            fusion[2] = 1
                            fusions_same_data = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if q == fusions_list[index][1] 
                                                     and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]]
                            for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in fusions_same_data: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful 
                                    relevant_fusion[2] = 1
                        else: # we measure ancilla, preserve primal cells and make all fusions involving ancilla successful
                            shuffled_layers[i][j][:2] = [0,0]
                            fusion[2] = 1
                            fusions_same_ancilla = [idx for idx, (i, q, fusion_type, time) in enumerate(fusions_list) if i == fusions_list[index][0] 
                                                     and fusion_type == fusions_list[index][2] and time == fusions_list[index][3]]
                            for relevant_fusion in shuffled_layers[i]:
                                if relevant_fusion[2] == 0 and relevant_fusion[3] in fusions_same_ancilla: # this will only consider unseen elements 
                                    relevant_fusion[:2] = [0,0] # make the relevant fusion successful 
                                    relevant_fusion[2] = 1

    # resort shuffled fusions within layers via inverse permutations 
    final_fusion_outcomes_temp = []

    for shuffled_layer, perm_for_layer in zip(shuffled_layers, permutations):
        reordered_layer = [None] * len(shuffled_layer) 
        for i, fusion_index in enumerate(perm_for_layer):
            reordered_layer[fusion_index] = shuffled_layer[i] # reorder by permutation for that layer 
        final_fusion_outcomes_temp.append(reordered_layer)
    
    final_fusion_outcomes_temp = [fusion for layer in final_fusion_outcomes_temp for fusion in layer] # flattens the list 

    # get rid of other information, leave just fusion outcomes of initially sampled form
    improved_fusion_outcomes = [fusion[:2] for fusion in final_fusion_outcomes_temp]
    
    primal_outcomes = []
    dual_outcomes = []
    for i in range(num_fusions):
        if xz_in_primal[i]==0: # ZX is in primal
            primal_outcomes.append(improved_fusion_outcomes[i][1]) # put ZX sampling outcome into primal
            dual_outcomes.append(improved_fusion_outcomes[i][0]) 
        else:
            primal_outcomes.append(improved_fusion_outcomes[i][0])
            dual_outcomes.append(improved_fusion_outcomes[i][1])

    return primal_outcomes, dual_outcomes

def test_log_loss(p_loss, H_with_logop, lattice, num_logops,xz_in_primal, N): 
    # loss p here is photon loss probability, not fusion loss probability
    primal_outcomes = sampling(p_loss, H_with_logop, lattice, xz_in_primal, N)[0]
    # dual_outcomes = sampling(p_loss, H_with_logop, lattice, xz_in_primal, N)[1]

    lost_fusions_ixs = np.where(primal_outcomes)[0].astype(np.int32) # pick primal or dual
    # lost_fusions_ixs = np.where(dual_outcomes)[0].astype(np.int32) # pick primal or dual 

    new_H_with_logop = LossDecoder_GaussElimin_noordered(H_with_logop, lost_fusions_ixs)
    
    if np.any(new_H_with_logop[-num_logops:, lost_fusions_ixs]):
        return 1
    else:
        return 0

def get_logerr_rate(p_loss, H_with_logop, lattice, num_trials, num_logops,xz_in_primal, N):
    
        num_log_errs = 0

        for _ in range(num_trials):
            num_log_errs += test_log_loss(p_loss, H_with_logop, lattice, num_logops, xz_in_primal,N)
            
        return num_log_errs/num_trials
    
    
def get_logerr_rate_parallel(loss_ps, H_with_logop, lattice, num_trials, num_logops,xz_in_primal, N):
    
    pool = multiprocessing.Pool()
    logerr_rates = pool.starmap(get_logerr_rate,
                                 zip(loss_ps, repeat(H_with_logop), repeat(lattice), repeat(num_trials), 
                                     repeat(num_logops), repeat(xz_in_primal), repeat(N)))
    
    return np.array(logerr_rates)
    
################

if __name__ == "__main__":

    p_list = np.linspace(0.0005, 0.06, 15)
    num_trials = 5000
    N = 3 # repeat until success max tries
    fig, ax = plt.subplots()
    
    import time 
    from datetime import datetime
    start_time = time.perf_counter()
    print(datetime.now().time())
    
    l_loss_all_codes = []
    for j in range(len(d_codes)):
        Hx = hx_lattice_list[j]
        lx = lx_lattice_list[j]
        lattice = lattice_list[j]
        # Hz = hz_lattice_list[j]
        # lz = lz_lattice_list[j]
    
        print("n = ", n_list[j])
        
        num_logops = lx.shape[0]
        # num_logops = lz.shape[0]
    
        l_loss = []
    
        H_with_logop = np.vstack([Hx, lx]).astype(np.uint8)
        # H_with_logop = np.vstack([Hz, lz]).astype(np.uint8)
    
        xz_in_primal = xz_in_primal_list[j]
        
        # l_loss = get_logerrs_parallel(p_list, H_with_logop, num_trials, num_logops,xz_in_primal, N) # not the rate
        l_loss = get_logerr_rate_parallel(p_list, H_with_logop, lattice, num_trials, num_logops, xz_in_primal, N)
    
        l_loss_all_codes.append(l_loss) 
        print(l_loss)
        
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("Total time taken (minutes): ", total_time/60)

    #save data
    # slurm_array_id = sys.argv[1]
    # np.savetxt(f"rus_bb_data_MDF/N_{N}_l_loss_{slurm_array_id}.txt", l_loss_all_codes)