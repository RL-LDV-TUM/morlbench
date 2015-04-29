'''
Created on Apr 29, 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

import numpy as np

# general experiment parameters
expparams_sarsa_sarsa_individual = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': False,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }
expparams_sarsa_sarsa_sum = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': True,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }
expparams_sarsa_avgq_individual = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': False,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }
expparams_sarsa_avgq_sum = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': True,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }
expparams_sarsa_static_individual = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': False,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }
expparams_sarsa_static_sum = {
    'independent_runs': 50,
    'interactions': 10000,
    'use_sum_of_payouts': True,
    'paramspace': np.linspace(0.5, 0.99, 25),
    # the number determined by the paramspace, which
    # gets looped over can be copied to either agent1
    # parameter, agent1 parameter or problem parameter
    # the original paramter in the supplied dictionary
    # will get overwritten. If no parameter should
    # get modified then use None
    'paramspace_to_agent1_parameter': 'epsilon',
    'paramspace_to_agent2_parameter': 'epsilon',
    'paramspace_to_problem_parameter': None
    }


# parameters for agent
aparams_sarsa = {
    '_agent_class': "SARSAPrisonerAgent",
    'alpha': 0.1,
    'gamma': 0.2,
    'epsilon': 0.9
    }
aparams_avgq = {
    '_agent_class': "AVGQPrisonerAgent",
    }
aparams_static_defect = {
    '_agent_class': "DefectPrisonerAgent"
    }
aparams_static_cooperate = {
    '_agent_class': "CooperatePrisonerAgent"
    }
aparams_stateful_sarsa = {
    '_agent_class': "SARSALastActionsPrisonerAgent",
    'alpha': 0.1,
    'gamma': 0.2,
    'epsilon': 0.9
    }


def gen_xdata(results, e):
    return [e['expparams']['paramspace'] for _ in xrange(len(results[0]))]


def gen_payout_ydata(results, e):
    paramspace = e['expparams']['paramspace']
    avg_payout1 = np.zeros((len(results), len(paramspace)))
    avg_payout2 = np.zeros((len(results), len(paramspace)))
    for r in xrange(len(results)):
        avg_payout1[r, :] = results[r][0][0]
        avg_payout2[r, :] = results[r][1][0]
    avg_payout1 = avg_payout1.mean(axis=0)
    avg_payout2 = avg_payout2.mean(axis=0)
    return [avg_payout1, avg_payout2]


def gen_learned_action_ydata(results, e):
    paramspace = e['expparams']['paramspace']
    learned_action1 = np.zeros((len(results), len(paramspace)))
    learned_action2 = np.zeros((len(results), len(paramspace)))
    for r in xrange(len(results)):
        learned_action1[r, :] = results[r][0][1]
        learned_action2[r, :] = results[r][1][1]
    learned_action1 = learned_action1.mean(axis=0)
    learned_action2 = learned_action2.mean(axis=0)
    return [learned_action1, learned_action2]

experiments = [
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_sarsa,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_sarsa_individual,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA 1", "SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_sarsa_payout_individual.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA 1", "SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_sarsa_learned_action_individual.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_sarsa_individual_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_sarsa,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_sarsa_sum,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA 1", "SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_sarsa_payout_sum.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA 1", "SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_sarsa_learned_action_sum.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_sarsa_sum_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_avgq,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_avgq_individual,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "AVGQ"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_avgq_payout_individual.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "AVGQ"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_avgq_learned_action_individual.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_avgq_individual_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_avgq,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_avgq_sum,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "AVGQ"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_avgq_payout_sum.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "AVGQ"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_avgq_learned_action_sum.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_avgq_sum_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_static_defect,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_static_individual,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "Defect"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_defect_payout_individual.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "Defect"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_defect_learned_action_individual.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_defect_individual_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_static_defect,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_static_sum,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "Defect"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_defect_payout_sum.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "Defect"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_defect_learned_action_sum.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_defect_sum_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_static_cooperate,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_static_individual,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "Cooperate"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_cooperate_payout_individual.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "Cooperate"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_cooperate_learned_action_individual.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_cooperate_individual_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_sarsa,
         'aparams2': aparams_static_cooperate,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_static_sum,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["SARSA", "Cooperate"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_sarsa_cooperate_payout_sum.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["SARSA", "Cooperate"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_sarsa_cooperate_learned_action_sum.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_sarsa_cooperate_sum_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_stateful_sarsa,
         'aparams2': aparams_stateful_sarsa,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_sarsa_individual,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["S-SARSA 1", "S-SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_stateful_sarsa_stateful_sarsa_payout_individual.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["S-SARSA 1", "S-SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_stateful_sarsa_stateful_sarsa_learned_action_individual.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_stateful_sarsa_stateful_sarsa_individual_epsilon_sweep.pickle"
         },
        {'aparams1': aparams_stateful_sarsa,
         'aparams2': aparams_stateful_sarsa,
         'pparams': {'_problem_class': "PrisonersDilemma",
                     'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
         'expparams': expparams_sarsa_sarsa_sum,
         'plotparams': [{'xdata': gen_xdata,
                         'ydata': gen_payout_ydata,
                         'labels': ["S-SARSA 1", "S-SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Payout",
                         'y_range': (0, 7, 1),
                         'output_filename': 'pd_normal_stateful_sarsa_stateful_sarsa_payout_sum.pdf',
                         'custom_yticks': [""] + ["%i" % (int(x)) for x in
                                                  np.arange(*(1, 7, 1))],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 6),
                         'label_offsets': [-0.1, -0.4]},
                        {'xdata': gen_xdata,
                         'ydata': gen_learned_action_ydata,
                         'labels': ["S-SARSA 1", "S-SARSA 2"],
                         'xlabel': r"$\epsilon$",
                         'x_range': (0.4, 1.1, 0.2),
                         'ylabel': "Learned Action",
                         'y_range': (0, 1.1, 0.2),
                         'output_filename': 'pd_normal_stateful_sarsa_stateful_sarsa_learned_action_sum.pdf',
                         'custom_yticks': ["Cooperate", "0.2\%", "0.4\%",
                                            "0.6\%", "0.8\%", "Defect"],
                         'fontsize': 25,
                         'label_fontsize': 25,
                         'y_lim': (0, 1),
                         'label_offsets': [-0.1, -0.4]},
                        ],
         'picklefile': "pd_stateful_sarsa_stateful_sarsa_sum_epsilon_sweep.pickle"
         },
        ]
