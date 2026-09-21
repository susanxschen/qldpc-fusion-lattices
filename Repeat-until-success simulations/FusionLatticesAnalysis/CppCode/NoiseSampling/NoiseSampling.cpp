# include "NoiseSampling.h"

# include <iostream>
# include <stdlib.h>
# include <math.h>
# include <random>
# include <functional>
#include <stdexcept>



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////// Function to sample lost fusions, with no adaptiveness

void SampleFusionLoss_passive(int n_qbts, int n_fusions, int* qbts_in_fusions, bool* fusion_biases, bool* lost_fusions, float p_loss, float p_fail, float p_fail_biased, bool fusion_is_physical) {
	int qbt_ix0, qbt_ix1, fus_ix, n_ancillas;
	bool *lost_qubits, *lost_ancillas; 

	/////////////// Sample lost photons

	lost_qubits =  (bool*) malloc(n_qbts * sizeof(bool));
	for (qbt_ix0 = 0; qbt_ix0 < n_qbts; ++qbt_ix0) {
		*(lost_qubits+qbt_ix0) = randomBoolean(p_loss);
	};

	/////////////// Sample lost ancillas
	
	if (p_fail != 0 && fusion_is_physical){
		n_ancillas = int(1/p_fail - 2); // Assumes boosting scheme from W. P. Grice Phys. Rev. A 84, 04233 (2011).
	
	} else if (!fusion_is_physical) {
		n_ancillas = 0;
	} else {
		throw std::invalid_argument( "p_fail" );
	}
		

	if (n_ancillas == 0) {
		lost_ancillas = (bool*) calloc(n_fusions, sizeof(bool));
	} else {
		lost_ancillas =  (bool*) malloc(n_fusions * sizeof(bool));
		for (fus_ix = 0; fus_ix < n_fusions; ++fus_ix) {
			*(lost_ancillas+fus_ix) = randomBoolean(1 - pow(1-p_loss, n_ancillas));
		};
	}



	/////////////// Sample lost fusions, no adaptiveness

	for (fus_ix = 0; fus_ix < n_fusions; ++fus_ix) {
		qbt_ix0 = *(qbts_in_fusions+2*fus_ix);
		qbt_ix1 = *(qbts_in_fusions+2*fus_ix + 1);

		if (*(lost_qubits+qbt_ix0) || *(lost_qubits+qbt_ix1) || *(lost_ancillas+fus_ix)){
			*(lost_fusions + 2*fus_ix) = *(lost_fusions + 2*fus_ix + 1) = 1;
		}
		else {
			if (*(fusion_biases+fus_ix)) {
				*(lost_fusions + 2*fus_ix) = randomBoolean(p_fail_biased);
				*(lost_fusions + 2*fus_ix + 1) = randomBoolean(p_fail);
			} else {
				*(lost_fusions + 2*fus_ix) = randomBoolean(p_fail);
				*(lost_fusions + 2*fus_ix + 1) = randomBoolean(p_fail_biased);
			}
		}
	};

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////// Function to sample lost fusions, simple adaptiveness


////////// Old version of the comment: 
////////// The adaptiveness here is implemented by, upon erasure of a fusion outcome, biasing the paired fusion which shares 
////////// the erased syndrom with the erased outcome, in the opposite type (i.e. dual if primal is erased and vice-versa) so that it is more likely to be safe.
////////// QUESTION: instead of swapping biasing one could just measure with single-qubit measurements in the opposite basis, 
////////// so that lost of a photon doesn't affect the other one and no ancillas are involved. 
////////// Would that make anything better? YES! no ancillas means that failure is less likely due to losses, and also has less noise, so let's put that in. [This started the new version]

void SampleFusionLoss_simpleactive(int n_qbts, int n_fusions, int* qbts_in_fusions, int* fusions_timeorder, int* time_of_fusions, 
								   bool* fusion_biases, bool* single_pauli_meas_list, int* paired_fusions,
								   bool* lost_fusions, float p_loss, float p_fail, float p_fail_biased, bool fusion_is_physical) {
	int qbt_ix0, qbt_ix1, fus_ix, time_ix, n_ancillas;
	bool fus_bias, fusion_failed, fusion_failed_biased, *lost_qubits, *lost_ancillas; 

	// std::cout << "\n\n ###########  Starting sampling losses and adaptive biasing ###########\n\n"<< std::endl;

	// std::cout << "\n\n ###########  First ten read elements of paired_fusions ###########\n\n"<< std::endl;
	// for (fus_ix=0; fus_ix<10; ++fus_ix) {
	// 		std::cout << *(paired_fusions+fus_ix)  << std::endl;
	// }

	// std::cout << "\n\n ###########  First ten read elements of qbts_in_fusions ###########\n\n"<< std::endl;
	// for (fus_ix=0; fus_ix<10; ++fus_ix) {
	// 		std::cout << *(qbts_in_fusions+fus_ix)  << std::endl;
	// }


	/////////////// Sample lost photons

	lost_qubits =  (bool*) malloc(n_qbts * sizeof(bool));
	for (qbt_ix0 = 0; qbt_ix0 < n_qbts; ++qbt_ix0) {
		*(lost_qubits+qbt_ix0) = randomBoolean(p_loss);
	};

	/////////////// Sample lost ancillas
	
	if (p_fail != 0 && fusion_is_physical){
		n_ancillas = int(1/p_fail - 2); // Assumes boosting scheme from W. P. Grice Phys. Rev. A 84, 04233 (2011).
	
	} else if (!fusion_is_physical) {
		n_ancillas = 0;
	} else {
		throw std::invalid_argument( "p_fail" );
	}
		

	if (n_ancillas == 0) {
		lost_ancillas = (bool*) calloc(n_fusions, sizeof(bool));
	} else {
		lost_ancillas =  (bool*) malloc(n_fusions * sizeof(bool));
		for (fus_ix = 0; fus_ix < n_fusions; ++fus_ix) {
			*(lost_ancillas+fus_ix) = randomBoolean(1 - pow(1-p_loss, n_ancillas));
		};
	}

	/////////////// Sample lost fusions, no adaptiveness

	for (time_ix = 0; time_ix < n_fusions; ++time_ix) {
		fus_ix = *(fusions_timeorder + time_ix);

		qbt_ix0 = *(qbts_in_fusions+2*fus_ix);
		qbt_ix1 = *(qbts_in_fusions+2*fus_ix + 1);

			// std::cout << "\n\nAttempting fusion " << fus_ix << " at time " << time_ix  << std::endl;
			// std::cout << "Biased in " << *(fusion_biases + fus_ix) << ". Paired with " << *(paired_fusions + 2*fus_ix + *(fusion_biases + fus_ix)) << std::endl;
			// std::cout << "Is single q meas: " << *(single_pauli_meas_list + fus_ix)<< std::endl;
			// std::cout << "fusion_biases:";
			// PrintMatrix_toTerminal(fusion_biases, 1, n_fusions);
			// std::cout << "single_pauli_meas_list:";
			// PrintMatrix_toTerminal(single_pauli_meas_list, 1, n_fusions);
			// std::cout << std::endl;	


		if (!*(single_pauli_meas_list + fus_ix)){  	/// a fusion is performed
			// std::cout << "   A 2-qubit fusion is attempted" << std::endl;
			if (*(lost_qubits+qbt_ix0) || *(lost_qubits+qbt_ix1) || *(lost_ancillas+fus_ix)){
				*(lost_fusions + 2*fus_ix) = *(lost_fusions + 2*fus_ix + 1) = 1;

					// std::cout << "\n\n Loss found when attempting 2-qubit fusion " << fus_ix << " at time " << time_ix  << std::endl;
					// std::cout << "Biased in " << *(fusion_biases + fus_ix) << ". Paired with " << *(paired_fusions + 2*fus_ix + *(fusion_biases + fus_ix)) << std::endl;
					// std::cout << "Is single q meas: " << *(single_pauli_meas_list + fus_ix)<< std::endl;

					// std::cout << "   Photon is lost, both outcomes erased" << std::endl;
				simple_update_bias_and_singles_after_fail(fus_ix, 0, fusion_biases, single_pauli_meas_list, paired_fusions, time_of_fusions);
				simple_update_bias_and_singles_after_fail(fus_ix, 1, fusion_biases, single_pauli_meas_list, paired_fusions, time_of_fusions);

			}
			else {
				fus_bias = *(fusion_biases+fus_ix);
				fusion_failed = randomBoolean(p_fail);
				fusion_failed_biased = randomBoolean(p_fail_biased);

				if (*(fusion_biases+fus_ix)) {
					*(lost_fusions + 2*fus_ix) = fusion_failed;
					*(lost_fusions + 2*fus_ix + 1) = randomBoolean(p_fail_biased);
					// *(lost_fusions + 2*fus_ix) = 0;
					// *(lost_fusions + 2*fus_ix + 1) = fusion_failed;
				} else {
					*(lost_fusions + 2*fus_ix) = randomBoolean(p_fail_biased);
					*(lost_fusions + 2*fus_ix + 1) = fusion_failed;
					// *(lost_fusions + 2*fus_ix) = fusion_failed;
					// *(lost_fusions + 2*fus_ix + 1) = 0;
				}

				if (fusion_failed) {

					// std::cout << "\n\n Loss found when attempting 2-qubit fusion " << fus_ix << " at time " << time_ix  << std::endl;
					// std::cout << "Biased in " << *(fusion_biases + fus_ix) << ". Paired with " << *(paired_fusions + 2*fus_ix + *(fusion_biases + fus_ix)) << std::endl;
					// std::cout << "Is single q meas: " << *(single_pauli_meas_list + fus_ix)<< std::endl;

					// std::cout << "   Fusion " << fus_ix <<  " failed, outcome  " << !fus_bias << " erased" << std::endl;
					simple_update_bias_and_singles_after_fail(fus_ix, !fus_bias, fusion_biases, single_pauli_meas_list, paired_fusions, time_of_fusions);					
				}
				// } else {
				// 	std::cout << "   Success! Both outcomes are obtained" << std::endl;
				// }

				if (fusion_failed_biased) {
				simple_update_bias_and_singles_after_fail(fus_ix, fus_bias, fusion_biases, single_pauli_meas_list, paired_fusions, time_of_fusions);					
				}
			}
		} else {      								/// single-qubit measurements are performed on the biased basis
			// std::cout << "   Single-qubit measurements are attempted for fusion"  << fus_ix << " at time " << time_ix  << std::endl;
			*(lost_fusions + 2*fus_ix + !(*(fusion_biases+fus_ix))) = 1;
			if (*(lost_qubits+qbt_ix0) || *(lost_qubits+qbt_ix1)){  /// When single-qubit measurements are performed, no ancillas are considered
				// std::cout << "   Photon is lost with single-qubit measurement, both outcomes erased" << std::endl;
				*(lost_fusions + 2*fus_ix + *(fusion_biases+fus_ix)) = 1;
			}
			// } else {
			// std::cout << "   Success! Oucome with bias" << *(fusion_biases+fus_ix) << " is obtained, the other is lost" << std::endl;
			// }
		}
	};

}

void simple_update_bias_and_singles_after_fail(int lost_fus_ix, bool lost_fus_bias, bool* fusion_biases, bool* single_pauli_meas_list, int* paired_fusions, int* time_of_fusions) {
	int paired_fus_ix;
	paired_fus_ix = *(paired_fusions + 2*lost_fus_ix + lost_fus_bias);

	// std::cout << "      Updating lost fusion " << lost_fus_ix << ", biased in " << lost_fus_bias << ", paired with fusion " << paired_fus_ix << std::endl;

	if (*(time_of_fusions+paired_fus_ix)>*(time_of_fusions+lost_fus_ix)) {   /// Update only if the paired fusion will happen at later times compared to the lost one
		// std::cout << "           Update of fusion bias and single-pauli on paired fusion completed." << std::endl;
		*(fusion_biases + paired_fus_ix) = !lost_fus_bias;
		*(single_pauli_meas_list + paired_fus_ix) = 1;
	}
	// } else {
	// 	std::cout << "           Paired qubit was precedent, no update performed." << std::endl;
	// }
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////// Function to sample qubit errors - assuming no correlated errors

void SampleFusionErrors_uncorrelated(int num_alive_primal_fusions, int* alive_primal_fusions, 
									 int num_alive_dual_fusions, int* alive_dual_fusions,  
									 bool* primal_errors,   bool* dual_errors, 
									 bool* fusion_biases, bool* single_pauli_meas_list, 
									 float p_err_biased, float p_err_unbiased) {
	int alive_ix, fus_ix;

	/////////////// Sample errors in primal fusions, no correlations

	for (alive_ix = 0; alive_ix < num_alive_primal_fusions; ++alive_ix) {
		fus_ix = *(alive_primal_fusions + alive_ix);

		if (!*(single_pauli_meas_list + fus_ix)) {
			if (*(fusion_biases + fus_ix)) {
				*(primal_errors + fus_ix) = randomBoolean(p_err_unbiased);
			} else {
				*(primal_errors + fus_ix) = randomBoolean(p_err_biased);
			}
		}
	};

	/////////////// Sample errors in dual fusions, no correlations

	for (alive_ix = 0; alive_ix < num_alive_dual_fusions; ++alive_ix) {
		fus_ix = *(alive_dual_fusions + alive_ix);

		if (!*(single_pauli_meas_list + fus_ix)) {
			if (*(fusion_biases + fus_ix)) {
				*(dual_errors + fus_ix) = randomBoolean(p_err_biased);
			} else {
				*(dual_errors + fus_ix) = randomBoolean(p_err_unbiased);
			}
		}
	};
}



////////// Function to calculate the error probability for a multiedge - assuming no correlated errors. 

void multiedge_errorprob_uncorrelated(int num_fusions, int num_multiedges, int* multiedge_of_fusion, bool is_primal,
									 float* multiedge_errorprobs, bool* fusion_biases, bool* single_pauli_meas_list, 
									 float p_err_biased, float p_err_unbiased) {
	int fus_ix, multiedge_ix;
	float p, temp_p;

	for (fus_ix = 0; fus_ix < num_fusions; ++fus_ix) {
		// std::cout << "fus_ix:" << fus_ix << ",  which single pauli" << *(single_pauli_meas_list + fus_ix)<< std::endl;

		if (!*(single_pauli_meas_list + fus_ix)) {
			// std::cout << "   Is not single Pauli: it's a fusion" << std::endl;
			multiedge_ix = *(multiedge_of_fusion + fus_ix);
			if (0 <= multiedge_ix &&  multiedge_ix < num_multiedges) {
				temp_p = *(multiedge_errorprobs + multiedge_ix);
				// std::cout << "   Multiedge is " << multiedge_ix << "And has current error_prob:" << temp_p << std::endl;

				if (*(fusion_biases + fus_ix) == is_primal){
					p = p_err_biased;
					// std::cout << "   Fusion is biased, error prob:" << p << std::endl;
				} else {
					p = p_err_unbiased;
					// std::cout << "   Fusion is unbiased, error prob:" << p << std::endl;

				}
				*(multiedge_errorprobs + multiedge_ix) = p*(1-temp_p) + temp_p*(1-p);
				// std::cout << "  New error prob on multiedge is :" << *(multiedge_errorprobs + multiedge_ix) << std::endl;
			}
		}
	};
}


////////// Function to propagate fusion errors in multi-edged syndrom graphs

void propagate_errors_multiedge(int num_fusions, int num_multiedges, int* syndromes_of_multiedge, int* multiedge_of_fusion, bool* errors_in_fusions, bool* fired_syndromes, bool* errors_in_multiedges) {
	int fus_ix, multiedge_ix;
	bool *syndrome_pointer;

	// std::cout << "      Physical fusion errors:" << std::endl;
	// PrintMatrix_toTerminal(errors_in_fusions, 1, num_fusions); 

	// std::cout << "      Initial syndrome errors:" << std::endl;
	// PrintMatrix_toTerminal(fired_syndromes, num_fusions, 1); 


	for (fus_ix = 0; fus_ix < num_fusions; ++fus_ix) {
		// std::cout << "fus_ix:" << fus_ix << "  with error" << *(errors_in_fusions + fus_ix) << std::endl;

		if (*(errors_in_fusions + fus_ix)) {
			multiedge_ix = *(multiedge_of_fusion + fus_ix);
			if (0 <= multiedge_ix &&  multiedge_ix < num_multiedges) {
				// std::cout << "  Error found on fus_ix:" << fus_ix << "  Associated multiedge:" << multiedge_ix << " In between 0 and" <<  num_multiedges << std::endl;
				// std::cout << "      Associated syndromes are" << *(syndromes_of_multiedge+2*multiedge_ix) << " and " << *(syndromes_of_multiedge+2*multiedge_ix + 1) << std::endl;
				syndrome_pointer = fired_syndromes  + *(syndromes_of_multiedge+2*multiedge_ix);
				// std::cout << "      syndrome_pointer" << syndrome_pointer << " with value " << *(syndrome_pointer) << std::endl;
				*(syndrome_pointer) = !*(syndrome_pointer);
				syndrome_pointer = fired_syndromes  + *(syndromes_of_multiedge+2*multiedge_ix + 1);
				// ++ syndrome_pointer;
				*(syndrome_pointer) = !*(syndrome_pointer);
		// 		// std::cout << "      New syndrome errors:" << std::endl;
		// 		// PrintMatrix_toTerminal(fired_syndromes, num_fusions, 1); 
				*(errors_in_multiedges+multiedge_ix) = !(*(errors_in_multiedges+multiedge_ix));
			}
		}
	};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void PrintMatrix_toTerminal(data_type* mat, int n_rows, int n_cols) {
	data_type* m;
	int ix_r, ix_c;

	m = mat;

	for (ix_r = 0; ix_r < n_rows; ix_r++)
	{
		std::cout << "\n";
		for (ix_c = 0; ix_c < n_cols; ix_c++)
		{
			std::cout << *m << " ";
			++m;
		}
	}
	std::cout << std::endl;
}


void PrintMatrix_int_toTerminal(int* mat, int n_rows, int n_cols) {
	int* m;
	int ix_r, ix_c;

	m = mat;

	for (ix_r = 0; ix_r < n_rows; ix_r++)
	{
		std::cout << "\n";
		for (ix_c = 0; ix_c < n_cols; ix_c++)
		{
			std::cout << *m << " ";
			++m;
		}
	}
	std::cout << std::endl;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool randomBoolean(const float p = 0.5) {
    static auto dev = std::random_device();
    static auto gen = std::mt19937{dev()};
    static auto dist = std::uniform_real_distribution<>(0,1);
    return (dist(gen) < p);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
    int n_qbts, n_fusions, qbt_ix0, qbt_ix1, fus_ix, time_ix, pair_fus_ix, pair_fus_time, *fusion_for_qubits, *qbts_in_fusions, *fusions_timeorder, *time_of_fusions, *paired_fusions;
	bool *lost_qubits, *fusion_biases, *lost_primal_fusions, *lost_dual_fusions, *single_pauli_meas_list; 
	float p_loss, p_fail, p_fail_biased;

	n_fusions = 40;
	n_qbts = 2*n_fusions;
	
	p_loss = 0.;
	p_fail = 0.25;
	p_fail_biased = 0;

	/////////////// Mock up fusions structure
	qbts_in_fusions =  (int*) malloc(n_qbts * sizeof(int));
	fusion_for_qubits = (int*) malloc(n_qbts * sizeof(int));
	fusion_biases = (bool*) malloc(n_fusions * sizeof(bool));
	fusions_timeorder = (int*) malloc(n_fusions * sizeof(int));
	time_of_fusions = (int*) malloc(n_fusions * sizeof(int));
	paired_fusions = (int*) malloc(2*n_fusions * sizeof(int));
	single_pauli_meas_list = (bool*) calloc(n_fusions, sizeof(bool));

	time_ix = 0;

	for (fus_ix = 0; fus_ix < n_fusions; ++fus_ix) {
		*(fusion_biases + fus_ix) = (int(fus_ix/2) % 2);

		qbt_ix0 = 2 * fus_ix;
		qbt_ix1 = 2 * fus_ix + 1;

		// Add qubit0 to fusion
		*(qbts_in_fusions + 2*fus_ix) = qbt_ix0;
		*(fusion_for_qubits + qbt_ix0) = fus_ix;

		// Add qubit1 to fusion

		*(qbts_in_fusions + 2*fus_ix + 1) = qbt_ix1;
		*(fusion_for_qubits + qbt_ix1) = fus_ix;

		// Update timeorder
		*(fusions_timeorder + time_ix) = fus_ix;
		*(time_of_fusions +fus_ix) = time_ix;

		if (fus_ix%2 ==0){
			*(paired_fusions + 2*fus_ix + 0) = (fus_ix+1) % n_fusions;
			*(paired_fusions + 2*fus_ix + 1) = (fus_ix+n_fusions-1) % n_fusions;			
		} else {
			*(paired_fusions + 2*fus_ix + 0) = (fus_ix+n_fusions-1) % n_fusions;
			*(paired_fusions + 2*fus_ix + 1) = (fus_ix+1) % n_fusions;			
		}

		++time_ix;

	};


	std::cout << "qbts_in_fusions:";
	PrintMatrix_int_toTerminal(qbts_in_fusions, 1, n_qbts);
	std::cout << std::endl;

	std::cout << "fusion_for_qubits:";
	PrintMatrix_int_toTerminal(fusion_for_qubits, 1, n_qbts);
	std::cout << std::endl;

	std::cout << "fusion_biases:";
	PrintMatrix_toTerminal(fusion_biases, 1, n_fusions);
	std::cout << std::endl;

	std::cout << "fusions_timeorder:";
	PrintMatrix_int_toTerminal(fusions_timeorder, 1, n_fusions);
	std::cout << std::endl;

	std::cout << "time_of_fusions:";
	PrintMatrix_int_toTerminal(time_of_fusions, 1, n_fusions);
	std::cout << std::endl;

	std::cout << "paired_fusions:";
	PrintMatrix_int_toTerminal(paired_fusions, 1, 2*n_fusions);
	std::cout << std::endl;

	std::cout << "single_pauli_meas_list:";
	PrintMatrix_toTerminal(single_pauli_meas_list, 1, n_fusions);
	std::cout << std::endl;	

	///////////////



	/////////////// Sample lost photons


	lost_qubits =  (bool*) malloc(n_qbts * sizeof(bool));
	for (qbt_ix0 = 0; qbt_ix0 < n_qbts; ++qbt_ix0) {
		*(lost_qubits+qbt_ix0) = randomBoolean(p_loss);
	};


	// /////////////// Sample lost fusions, no adaptiveness

	// lost_primal_fusions =  (bool*) calloc(n_fusions, sizeof(bool));
	// lost_dual_fusions =  (bool*) calloc(n_fusions, sizeof(bool));

	// for (fus_ix = 0; fus_ix < n_fusions; ++fus_ix) {
	// 	qbt_ix0 = *(qbts_in_fusions+2*fus_ix);
	// 	qbt_ix1 = *(qbts_in_fusions+2*fus_ix + 1);

	// 	if (*(lost_qubits+qbt_ix0) || *(lost_qubits+qbt_ix1)){
	// 		*(lost_primal_fusions + fus_ix) = *(lost_dual_fusions + fus_ix) = 1;
	// 	}
	// 	else {
	// 		if (*(fusion_biases+fus_ix)) {
	// 			*(lost_primal_fusions + fus_ix) = 0;
	// 			*(lost_dual_fusions + fus_ix) = randomBoolean(p_fail);
	// 		} else {
	// 			*(lost_primal_fusions + fus_ix) = randomBoolean(p_fail);
	// 			*(lost_dual_fusions + fus_ix) = 0;
	// 		}
	// 	}
	// };

	/////////////// Sample lost fusions, simple adaptive

	// lost_primal_fusions =  (bool*) calloc(n_fusions, sizeof(bool));
	// lost_dual_fusions =  (bool*) calloc(n_fusions, sizeof(bool));


	bool* lost_fusions = (bool*) calloc(2*n_fusions, sizeof(bool));

	SampleFusionLoss_simpleactive(n_qbts, n_fusions, qbts_in_fusions, fusions_timeorder, time_of_fusions, 
									fusion_biases, single_pauli_meas_list, paired_fusions,
									lost_fusions, p_loss, p_fail, p_fail_biased, true);


	/////////////// Check that, with no loss, all the single qubit measurements succeed


	/////////////// 

	// PrintMatrix_int_toTerminal(qbts_in_fusions, 1, n_qbts);
	// std::cout << std::endl;
	// PrintMatrix_int_toTerminal(fusion_for_qubits, 1, n_qbts);
	// std::cout << std::endl;
	// PrintMatrix_toTerminal(fusion_biases, 1, n_fusions);
	// std::cout << std::endl;
	// PrintMatrix_toTerminal(lost_qubits, 1, n_qbts);
	// std::cout << std::endl;
	// PrintMatrix_toTerminal(lost_primal_fusions, 1, n_fusions);
	// std::cout << std::endl;
	PrintMatrix_toTerminal(lost_fusions, n_fusions, 2);
	std::cout << std::endl;

	int* all_good_biased_outcomes_primal = (int*) calloc(n_fusions, sizeof(int));
	int* all_good_biased_outcomes_dual = (int*) calloc(n_fusions, sizeof(int));

	for (time_ix = 0; time_ix < n_fusions; ++time_ix) {
		fus_ix = *(fusions_timeorder  + time_ix);
		
		if (*(lost_fusions + 2*fus_ix + 0)){
			pair_fus_ix = *(paired_fusions  + 2*fus_ix + 0);
			pair_fus_time = *(time_of_fusions + pair_fus_ix);	
			if (pair_fus_time>time_ix && *(lost_fusions + 2*fus_ix + 1)) {
				std::cout << "Found a bad primal case!" << std::endl;
				*(all_good_biased_outcomes_primal) = pair_fus_ix;
				++all_good_biased_outcomes_primal;
			}
		}

		if (*(lost_fusions + 2*fus_ix + 1)){
			pair_fus_ix = *(paired_fusions  + 2*fus_ix + 1);
			pair_fus_time = *(time_of_fusions + pair_fus_ix);	
			if (pair_fus_time>time_ix && *(lost_fusions + 2*fus_ix + 0)) {
				std::cout << "Found a bad dual case!" << std::endl;
				*(all_good_biased_outcomes_dual) = pair_fus_ix;
				++all_good_biased_outcomes_dual;
			}
		}

		};

	PrintMatrix_int_toTerminal(all_good_biased_outcomes_primal, 1, n_fusions);
	std::cout << std::endl;
	PrintMatrix_int_toTerminal(all_good_biased_outcomes_primal, 1, n_fusions);
	std::cout << std::endl;

    return 0;
}