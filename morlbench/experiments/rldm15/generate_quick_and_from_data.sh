#!/bin/bash

# regenerate the plots, that can be calculated quickly
# regenerate the rest from the pickled data

python figure_1_a_one_vs_twoboxing.py
python figure_1_b_c_combined_newcomb_sarsa_avgq_eu.py
python figure_2_a_b_defect_vs_cooperate.py
python figure_2_c_d_pd_avgq_sarsa_payouts.py

./cropfigures.sh
