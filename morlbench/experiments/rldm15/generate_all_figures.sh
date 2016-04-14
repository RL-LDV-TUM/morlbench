#!/bin/bash

# regenerate all plots
# delete all .pickle files beforehand


python figure_1_a_one_vs_twoboxing.py
python nc_avgq_prediction_sweep.py && python nc_eu_prediction_sweep.py && python nc_sarsa_prediction_sweep.py && python figure_1_b_c_combined_newcomb_sarsa_avgq_eu.py
python figure_2_a_b_defect_vs_cooperate.py
python ppd_avgq_prediction_sweep.py && python ppd_sarsa_prediction_sweep.py && python figure_2_c_d_pd_avgq_sarsa_payouts.py

./cropfigures.sh
