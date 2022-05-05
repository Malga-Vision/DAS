from modules.data import *
from modules.stein import *
import cdt
import networkx as nx
import pandas as pd

def main():

    # Hyperparams
    threshold = 1e-5
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001
    sid=True
    K=None
    pns=5

    data = "Sachs"
    pruning = "CAM"

    if data == "Dream":
        path_to_data = "/data/francescom/dream5/DREAM5_network_inference_challenge/Network1/input_data/net1_expression_data.tsv"
        gold_standard_path = "/data/francescom/dream5/DREAM5_network_inference_challenge/Evaluation_scripts/INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network1.tsv"

        # Logs paths
        fast_path = f'../logs/test/dream5_fast.txt'
        fastcam_path = f'../logs/test/dream5_fastcam.txt'

        # Data
        X = dream_data(path_to_data)
        A_truth = dream_adj(gold_standard_path, X.shape)

    elif data == "Sachs":
        path_to_data = ""
        # Logs paths
        fast_path = f'../logs/test/sachs_{pruning.lower()}_{threshold}.txt'
        fastcam_path = f'../logs/test/sachs_fastcam_{threshold}.txt'

        # Data
        _, G_target = cdt.data.load_dataset('sachs')
        X = pd.read_excel()
        columns = list(X.columns)
        X = torch.tensor(X.to_numpy())
        A_truth = nx.adjacency_matrix(G_target, nodelist=columns).toarray()
        sid = True


    A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold = threshold, pns=pns, K=K)
    top_order_err = num_errors(top_order_SCORE, A_truth)
    pretty = pretty_evaluate(pruning, threshold, A_truth, A_SCORE, top_order_err, SCORE_time, tot_time, sid, K=K)
    print(pretty)
    
    # FAST logs
    with open(fast_path, 'a+') as f:
        f.writelines(pretty)


    # FastCAM
    if pruning != "CAM":
        start = time.time()
        A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
        cam_time = time.time() - start
        tot_time += cam_time
        pretty = pretty_evaluate("K-FastCAM", threshold, A_truth, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
        print(pretty)
        with open(fastcam_path, 'a+') as f:
            f.writelines(pretty)


if __name__ == '__main__':
    main()