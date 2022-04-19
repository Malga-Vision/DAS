from modules.data import *
from modules.stein import *

def main():
    # Args
    path_to_data = "/data/francescom/dream5/DREAM5_network_inference_challenge/Network1/input_data/net1_expression_data.tsv"
    gold_standard_path = "/data/francescom/dream5/DREAM5_network_inference_challenge/Evaluation_scripts/INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network1.tsv"

    # Hyperparams
    threshold = 0.4
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001

    X = dream_data(path_to_data)
    A_truth = dream_adj(gold_standard_path, X.shape)

    A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning="Fast", threshold = threshold)
    shd = SHD(A_SCORE, A_truth)
    print(f"SCORE execution time: ----- {round(SCORE_time, 2)}s -----")
    print(f"Total execution time: ----- {round(tot_time, 2)}s -----")
    print(f"Pruning: Fast, Threshold: {threshold}")
    print(f"SHD : {shd}")
    # print("SID: {}".format(int(cdt.metrics.SID(target=A_truth, pred=A_SCORE))))
    print("top order errors: {}".format(num_errors(top_order_SCORE, A_truth)))

    fn, fp, rev = edge_errors(A_SCORE, A_truth)

    # FAST logs
    with open(f'../logs/test/dream5_fast_{threshold}.txt', 'a+') as f:
        f.writelines(f'SHD: {shd}\n')
        f.writelines(f'False negative: {fn}\n')
        f.writelines(f'False positive: {fp}\n')
        f.writelines(f'Reversed: {rev}\n')
        f.writelines(f'SCORE time: {round(SCORE_time, 2)}s\n')
        f.writelines(f'Total time: {round(tot_time, 2)}s\n')
        f.writelines(f'Topological ordering errors: {num_errors(top_order_SCORE, A_truth)}\n')


    # CAM
    start = time.time()
    A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
    cam_time = time.time() - start
    tot_time += cam_time
    shd = SHD(A_SCORE, A_truth)
    fn, fp, rev = edge_errors(A_SCORE, A_truth)
    with open(f'../logs/test/dream5_fastcam_{threshold}.txt', 'a+') as f:
        f.writelines(f'SHD: {shd}\n')
        f.writelines(f'False negative: {fn}\n')
        f.writelines(f'False positive: {fp}\n')
        f.writelines(f'Reversed: {rev}\n')
        f.writelines(f'SCORE time: {round(SCORE_time, 2)}s\n')
        f.writelines(f'Total time: {round(tot_time, 2)}s\n')
        f.writelines(f'Topological ordering errors: {num_errors(top_order_SCORE, A_truth)}\n')


if __name__ == '__main__':
    main()