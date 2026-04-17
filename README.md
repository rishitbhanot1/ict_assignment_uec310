# UEC-310 Information and Communication Theory Assignment

This repository contains the solution for the UEC-310 assignment on **Reliable Data Transmission in a Satellite Communication System**. The work studies the transmission of environmental sensor data from remote Earth-based sensors to a ground station over a noisy communication channel.

The assignment is divided into two parts:

- **Part A — Analytical Tasks**
  - Source entropy calculation
  - Coding efficiency evaluation
  - Binary Symmetric Channel (BSC) capacity analysis

- **Part B — Simulation-Based Tasks**
  - BSC transmission simulation for 10,000 bits
  - Bit Error Rate (BER) computation
  - Entropy vs probability plot
  - Channel capacity vs noise plot

## Problem Context

A satellite communication company is transmitting environmental data from remote sensors on Earth to a ground station. Due to long-distance propagation and atmospheric disturbances, the communication channel is noisy, which can cause bit errors during transmission.

To study this system, the assignment analyzes:
- the **compressibility of the source**
- the **efficiency of the coding scheme**
- the **capacity of the noisy channel**
- the **practical behavior of BER and channel performance through simulation**

## Repository Contents

- `ict_asgn_script.py` — Python implementation for the simulation-based tasks
- `ict_asgn_script.ipynb` — Jupyter Notebook Equivalent for the simulation-based tasks
- `report.tex` — LaTeX source of the report
- `part_a_q1.png` , `part_a_q2.png` , `part_a_q3.png` - Numerical Analysis for Part A
- `b_plots.png` — Simulation Plots for Part B

## Tools Used

- Python
- NumPy
- Matplotlib
- Seaborn
- LaTeX

## Key Results

- **Source Entropy:** 0.8813 bits/symbol
- **Coding Efficiency:** 73.44%
- **BSC Capacity at p = 0.1:** 0.5310 bits/channel use
- **Simulated BER:** close to the theoretical error probability of 0.1
