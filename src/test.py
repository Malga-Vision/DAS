from modules.data import *
from modules.stein import *
from modules.utils import ground_truth
import cdt
import os
import networkx as nx
import pandas as pd
from glob import glob

def main():

    # Hyperparams
    threshold = 0
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001
    sid=False
    K=10
    pns=None

    data = "Sergio"
    pruning = "Fast"

    if data == "Sergio":
        folder_path = "/home/francescom/Research/De-noised_400G_9T_300cPerT_5_DS2"

        # Logs paths
        fast_path = f'../logs/test/sergio_{pruning.lower()}.txt'
        fastcam_path = f'../logs/test/sergio_fastcam.txt'

        # Data
        data_path = os.path.join(folder_path, "simulated_noNoise_8.csv")
        X = pd.read_csv(data_path)
        X = torch.tensor(X.to_numpy()[:, 1:])
        d = X.shape[1]

        # Ground truth
        gt_path = os.path.join(folder_path, "gt_GRN.csv")
        A_truth = ground_truth(d, gt_path)

        custom_d = 20
        X = X[:, :custom_d]
        A_truth = A_truth[:custom_d, :custom_d]
        tp = A_truth.sum()
        print(f"True positives: {tp}")

        sid = False

    elif data == "Sachs":
        path_to_data = ""
        
        # Logs paths
        fast_path = f'../logs/test/sachs_{pruning.lower()}_{threshold}.txt'
        fastcam_path = f'../logs/test/sachs_fastcam_{threshold}.txt'

        # Data
        _, G_target = cdt.data.load_dataset('sachs')
        X = pd.read_excel(path_to_data)
        columns = list(X.columns)
        X = torch.tensor(X.to_numpy())
        A_truth = nx.adjacency_matrix(G_target, nodelist=columns).toarray()
        sid = True


    A_SCORE, top_order_SCORE, SCORE_time =  SCORE(X, eta_G, eta_H, threshold = threshold, K=K)
    top_order_err = num_errors(top_order_SCORE, A_truth)
    pretty = pretty_evaluate(pruning, threshold, A_truth, A_SCORE, top_order_err, SCORE_time, SCORE_time, sid, s0=A_truth.sum(), K=K)
    print(pretty)
    
    # FAST logs
    with open(fast_path, 'a+') as f:
        f.writelines(pretty)


    # FastCAM
    if pruning == "Fast":
        tot_time = SCORE_time
        start = time.time()
        A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
        cam_time = time.time() - start
        tot_time += cam_time
        pretty = pretty_evaluate("K-FastCAM", threshold, A_truth, A_SCORE, top_order_err, SCORE_time, tot_time, sid, s0=A_truth.sum(), K=K)
        print(pretty)
        with open(fastcam_path, 'a+') as f:
            f.writelines(pretty)


if __name__ == '__main__':
    main()