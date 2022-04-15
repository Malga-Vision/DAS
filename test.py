import cdt
from experiments import FastExperiment
from datasets import *
from stein import *

def main():
    # Args
    path_to_data = "/data/francescom/dream5/DREAM5_network_inference_challenge/Network1/input_data/net1_expression_data.tsv"
    gold_standard_path = "/data/francescom/dream5/DREAM5_network_inference_challenge/Evaluation_scripts/INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network1.tsv"
    pruning = "Fast"

    # Hyperparams
    threshold = 0.01
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001

    X = dream_data(path_to_data)
    A_truth = dream_adj(gold_standard_path, X.shape)

    A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold = threshold)
    print(f"SCORE execution time: ----- {round(SCORE_time, 2)}s -----")
    print(f"Total execution time: ----- {round(tot_time, 2)}s -----")
    print(f"Pruning: {pruning}, Threshold: {threshold}")
    print("SHD : {}".format(SHD(A_SCORE, A_truth)))
    print("SID: {}".format(int(cdt.metrics.SID(target=A_truth, pred=A_SCORE))))
    print("top order errors: {}".format(num_errors(top_order_SCORE, A_truth)))