from modules.data import *
from modules.stein import *

def main():
    # Args
    path_to_data = "/data/francescom/dream5/DREAM5_network_inference_challenge/Network1/input_data/net1_expression_data.tsv"
    gold_standard_path = "/data/francescom/dream5/DREAM5_network_inference_challenge/Evaluation_scripts/INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network1.tsv"

    # Hyperparams
    threshold = 0.05
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001
    sid=False
    K=15
    pns=None

    # Logs paths
    fast_path = f'../logs/test/dream5_fast_{threshold}.txt'
    fastcam_path = f'../logs/test/dream5_fastcam_{threshold}.txt'

    # Data
    X = dream_data(path_to_data)
    A_truth = dream_adj(gold_standard_path, X.shape)

    A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning="Fast", threshold = threshold, pns=pns, K=K)
    top_order_err = num_errors(top_order_SCORE, A_truth)
    pretty = pretty_evaluate("K-Fast", threshold, A_truth, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
    print(pretty)
    
    # FAST logs
    with open(fast_path, 'w+') as f:
        f.writelines(pretty)


    # FastCAM
    start = time.time()
    A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
    cam_time = time.time() - start
    tot_time += cam_time
    pretty = pretty_evaluate("K-FastCAM", threshold, A_truth, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
    with open(fastcam_path, 'w+') as f:
        f.writelines(pretty)


if __name__ == '__main__':
    main()