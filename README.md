# qldpc-fusion-lattices

This repository contains all code, simulations and data/figures generated for the article **Fusion-based implementation of qLDPC codes with quantum emitters** (npj Quantum Inf 12, 90 (2026), https://doi.org/10.1038/s41534-026-01233-y).  It features a class which generates fusion-based quantum computing lattices to implement quantum low-density parity check (qLDPC) codes, and simulation scripts to test small examples of Bivariate Bicycle (and Toric) code lattices with respect to phenomenological and physical noise models employing repeat-until-success strategies. 

## Usage
### Compiling the UF Decoder for phenomenological simulations

The following [variant of the union-find decoder](https://arxiv.org/pdf/2407.15988) was used in the phenomenological simulations, refer to [this repository](https://github.com/nbi-hyq/uf_decoder) to find set-up instructions.  
Linux required (tested on Ubuntu 22.04.2 LTS). 

### Compiling the Loss Decoder for RUS scripts

#### Requirements
- Python 3
- `g++` (Linux/WSL) or MinGW-w64 (Windows)

#### Linux or Windows subsystem for Linux 

From the project root (`check-RUS`), run:

```bash
cd FusionLatticesAnalysis/CppCode/LossDecoder

g++ -O2 -fPIC -c LossDec_funcs.cpp -o LossDec_funcs.o
g++ -shared -o libLossDec.so LossDec_funcs.o

cp libLossDec.so ../../libLossDec.so
```

Run the simulation script from the root:

```bash
cd ../../..
python3 rus_bb_lattice_ler_simulation.py
```

#### Windows

If you are using a Windows-compatible compiler such as MinGW-w64:

```bash
cd FusionLatticesAnalysis/CppCode/LossDecoder

g++ -O2 -shared -o LossDec.dll LossDec_funcs.cpp
```
