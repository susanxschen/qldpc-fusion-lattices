"""
Class for generating fusion lattices from CSS code input
Input: 
- parity checks Hx, Hz
- tanner graph vertex degrees 
- logical operators lx, lz
- number of "rounds" of syndrome measurement
"""


import numpy as np

class Fusion_lattice(object):
    def __init__(self, Hx, Hz, check_degree, qubit_degree, T, lx, lz): # T = the number of "time intervals"
        self.Hx, self.Hz = Hx, Hz 
        self.T = T
        self.check_degree = check_degree
        self.qubit_degree = qubit_degree
        self.lx = lx # logicals of the code 
        self.lz = lz
        
        self.num_ancilla = self.Hx.shape[0] # assume there are same number of x and z checks 
        self.num_qubits = self.Hx.shape[1]

        ### some lattice numbers ###
        self.num_total_cells = self.num_ancilla * self.T # total no. x checks * no. intervals
        self.num_total_data_rs = self.num_qubits # total no. data resource states
        self.num_total_ancilla_rs = 2 * self.num_ancilla * self.T 
        self.num_fusions_per_data_rs_per_layer = int(self.qubit_degree / 2) # since split between sublayers of x and z
        self.num_fusions_per_ancilla_rs = self.check_degree
        self.num_fusions_per_cell = self.num_fusions_per_ancilla_rs * (2 + self.num_fusions_per_data_rs_per_layer)
        self.num_total_fusions = 2 * self.num_ancilla * self.check_degree * self.T
        
        self.data_rs_dict = {}
        self.x_ancilla_rs_dict = {}
        self.z_ancilla_rs_dict = {}
        self.lattice_fusions = []
        
        self.fill_dictionaries()
        self.fill_lattice_fusions()
        
    ### label cells, resource states, and fusions ###
    def cell_ind(self, i, t):
        """
        C(S_i^x, t) = t * num_ancilla + i 
        - i labels base x check
        - t labels time interval
        """
        return t * self.num_ancilla + i 
    
    def data_rs_ind(self, j): # associated qubit number j
        return j
    def x_ancilla_rs_ind(self, i, t):
        return self.num_qubits + 2 * t * self.num_ancilla + i 
    def z_ancilla_rs_ind(self, i, t):
        return self.num_qubits + (2 * t + 1) * self.num_ancilla + i
    
    def x_fusion_ind(self, i, j, t): # fuses an x check with data resource state j at time t
        # here j is within the range, 0 -> check degree (if searching via check, else must search through lattice_fusions)
        return (2 * t * self.num_ancilla + i) * self.check_degree + j
    def z_fusion_ind(self, i, j, t):
        return ((2 * t + 1) * self.num_ancilla + i) * self.check_degree + j
    
    ### fill above dictionaries with indices as keys ###
    
    def fill_dictionaries(self):
        for t in range(self.T):
            for i in range(self.num_ancilla):
                x_rs_index = self.x_ancilla_rs_ind(i, t)
                z_rs_index = self.z_ancilla_rs_ind(i, t)
                
                self.x_ancilla_rs_dict[x_rs_index] = (i,t)
                self.z_ancilla_rs_dict[z_rs_index] = (i,t)
        
                for j in range(self.num_qubits):
                        data_index = self.data_rs_ind(j)
                        self.data_rs_dict[data_index] = j
            
    def fill_lattice_fusions(self):
        # put all the involved fusions into a list 
        x_fusion_ind_list = np.argwhere(self.Hx == 1)
        z_fusion_ind_list = np.argwhere(self.Hz == 1)

        x_fusions_layer = [(i, j, "X") for i, j in x_fusion_ind_list]
        z_fusions_layer = [(i, j, "Z") for i, j in z_fusion_ind_list]

        fusions_layer = np.concatenate((x_fusions_layer, z_fusions_layer))
        self.lattice_fusions = [(int(i),int(j),k,int(t)) for t in range(self.T) for i,j,k in fusions_layer]
    
    ### defining lattice structure ###
        
    def get_matching_matrix(self):
        matching_matrix = np.zeros((self.num_total_cells, self.num_total_fusions))
        for t in range(self.T):
            for i in range(self.num_ancilla): # go through each cell in layer t
                cell_index = self.cell_ind(i, t)

                # layer tx and (t+1)x fusions 
                data_qubits = np.where(self.Hx[i] == 1)[0] # the j's 

                for j in range(self.check_degree):
                    fusion_index = self.x_fusion_ind(i, j, t)
                    matching_matrix[cell_index, fusion_index] = 1
                    fusion_index_top = self.x_fusion_ind(i, j, (t+1) % self.T) 
                    matching_matrix[cell_index, fusion_index_top] = 1

                    # layer tz fusions 
                    # find attached z checks
                    connected_z_checks = np.where(self.Hz.T[data_qubits[j]] == 1)[0]
                    for k in connected_z_checks:
                        # search the list for index with matching values, if searching via data qubit
                        fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                             if i == k and q == data_qubits[j] and fusion_type == 'Z' and time == t), None)
                        matching_matrix[cell_index, fusion_index] = 1

        return matching_matrix
        
    def get_dual_matching_matrix(self):
        matching_matrix = np.zeros((self.num_total_cells, self.num_total_fusions))
        for t in range(self.T):
            for i in range(self.num_ancilla): # go through each cell in layer t
                cell_index = self.cell_ind(i, t)

                # layer tz and (t+1)z fusions 
                data_qubits = np.where(self.Hz[i] == 1)[0] # the j's 

                for j in range(self.check_degree):
                    fusion_index = self.z_fusion_ind(i, j, t)
                    matching_matrix[cell_index, fusion_index] = 1
                    fusion_index_top = self.z_fusion_ind(i, j, (t+1) % self.T) 
                    matching_matrix[cell_index, fusion_index_top] = 1

                    # layer tz fusions 
                    # find attached z checks
                    connected_x_checks = np.where(self.Hx.T[data_qubits[j]] == 1)[0]
                    for k in connected_x_checks:
                        # search the list for index with matching values, if searching via data qubit
                        fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                             if i == k and q == data_qubits[j] and fusion_type == 'X' and time == t), None)
                        matching_matrix[cell_index, fusion_index] = 1

        return matching_matrix
                
    def get_cells_and_rs_matrix(self): # for primal
        cells_and_rs_matrix = np.zeros((self.num_total_cells, self.num_total_data_rs + self.num_total_ancilla_rs))
        for t in range(self.T):
            for i in range(self.num_ancilla): # go through each check in layer t
                cell_index = self.cell_ind(i, t)

                # top and bottom face x ancilla rs
                rs_index_bottom = self.x_ancilla_rs_ind(i, t)
                rs_index_top = self.x_ancilla_rs_ind(i, (t+1) % self.T)
                cells_and_rs_matrix[cell_index, rs_index_bottom] = 1
                cells_and_rs_matrix[cell_index, rs_index_top] = 1

                connected_data_qubits = np.where(self.Hx[i] == 1)[0]
                # data rs 
                for j in connected_data_qubits:
                    data_rs_index =  self.data_rs_ind(j)
                    cells_and_rs_matrix[cell_index, data_rs_index] = 1

                # z ancilla rs 
                for j in connected_data_qubits:
                    connected_z_checks = np.where(self.Hz.T[j] == 1)[0]
                    for k in connected_z_checks:
                        rs_index = self.z_ancilla_rs_ind(k, t)
                        cells_and_rs_matrix[cell_index, rs_index] = 1
                        
        return cells_and_rs_matrix

    def get_rs_and_fusions_matrix(self):
        rs_and_fusions_matrix = np.zeros((self.num_total_data_rs + self.num_total_ancilla_rs, 
                                          self.num_total_fusions))
        for t in range(self.T):

            # fusions associated with data rs 
            for j in range(self.num_total_data_rs):
                # find attached x and z checks 
                data_rs_index =  self.data_rs_ind(j)
                connected_x_checks = np.where(self.Hx.T[j] == 1)[0]
                connected_z_checks = np.where(self.Hz.T[j] == 1)[0]
                for k in connected_x_checks:
                    fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                         if i == k and q == j and fusion_type == 'X' and time == t), None)
                    rs_and_fusions_matrix[data_rs_index, fusion_index] = 1
                for k in connected_z_checks:
                    fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                         if i == k and q == j and fusion_type == 'Z' and time == t), None)
                    rs_and_fusions_matrix[data_rs_index, fusion_index] = 1

            # fusions assoicated with ancilla rs 
            for i in range(self.num_ancilla):
                x_ancilla_index = self.x_ancilla_rs_ind(i, t)
                z_ancilla_index = self.z_ancilla_rs_ind(i, t)

                for j in range(self.check_degree):
                    x_fusion_index = self.x_fusion_ind(i, j, t)
                    rs_and_fusions_matrix[x_ancilla_index, x_fusion_index] = 1
                    z_fusion_index = self.z_fusion_ind(i, j, t)
                    rs_and_fusions_matrix[z_ancilla_index, z_fusion_index] = 1

        return rs_and_fusions_matrix

    def get_logical_x(self):
        num_log_x = self.lx.shape[0]
        log_x_lattice = np.zeros((self.lx.shape[0], self.num_total_fusions))

        for n in range(num_log_x):
            qubits = np.where(self.lx[n]==1)[0]
            for t in range(self.T):
                for j in qubits:
                    connected_z_checks = np.where(self.Hz.T[j]==1)[0]
                    for k in connected_z_checks: # x logical will not include the fusions with x checks 
                        fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                             if i == k and q == j and fusion_type == 'Z' and time == t), None)
                        log_x_lattice[n, fusion_index] = 1
    
        return log_x_lattice

    def get_logical_z(self):
        num_log_z = self.lz.shape[0]
        log_z_lattice = np.zeros((self.lz.shape[0], self.num_total_fusions))

        for n in range(num_log_z):
            qubits = np.where(self.lz[n]==1)[0]
            for t in range(self.T):
                for j in qubits:
                    connected_x_checks = np.where(self.Hx.T[j]==1)[0]
                    for k in connected_x_checks: # z logical will not include the fusions with z checks 
                        fusion_index = next((idx for idx, (i, q, fusion_type, time) in enumerate(self.lattice_fusions) 
                                             if i == k and q == j and fusion_type == 'X' and time == t), None)
                        log_z_lattice[n, fusion_index] = 1
    
        return log_z_lattice