
from SOT.Exact_posterior.exact_examples import *
from SOT.Nearest_neighbour.NN_examples import *
from SOT.Probabilistic_data_assosiation.PDA_examples import *
from SOT.Gaussian_sum_filter.GSF_examples import *
from nOT.Gobal_nearest_neighbour.GNN_examples import *
from nOT.Multi_hypotesis_tracker.MHT_examples import *


if __name__ == "__main__":
    
    # exact_linear_gaussian_models_single_time_step()
    # exact_linear_gaussian_models_several_time_steps()

    # NN_linear_gaussian_models_simple_scenario()
    # NN_linear_gaussian_models_hard_scenario()

    # PDA_linear_gaussian_models_simple_scenario()
    # PDA_linear_gaussian_models_hard_scenario()

    # GSF_linear_gaussian_models_simple_scenario()
    # GSF_linear_gaussian_models_hard_scenario()

    # GNN_LG_models_2_obj_1D()
    # GNN_LG_models_2_obj_sequence_1D()

    MHT_LG_models_2_obj_1D()

    input('Press any button to quit.')
