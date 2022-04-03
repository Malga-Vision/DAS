import argparse

def get_args():

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Data generation arguments
    parser.add_argument('--graph_type', type=int, default='ER', help="Acceptedvalues: ER, SF")
    parser.add_argument('-d', type=int, default=10, help="Number of causal variables")
    parser.add_argument('-s0', type=int, default=10, help="Number of expected edges")
    parser.add_argument('-N', type=int, default=1000, help="Sample size")

    # Hyperparameters
    parser.add_argument('--eta_G', type=float, default=0.001, help="Regularization coefficient 1st order")
    parser.add_argument('--eta_H', type=float, default=0.001, help="Regularization coefficient 2nd order")
    parser.add_argument('--cam_cutoff', type=float, default=0.001, help="CAM pruning hyperparameter")

    # Others
    parser.add_argument('--tf', type=float, default=0.2, help="Threshold for fast pruning (mine)")
    parser.add_argument('--ts', type=float, default=0.2, help="Threshold for Stein pruning")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (TODO: make results reproducible")


    # Read arguments from command line
    return parser.parse_args()
