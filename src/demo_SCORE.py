from modules.stein import *
from modules.utils import generate, pretty_evaluate
import cdt


# Data generation paramters
graph_type = 'ER'
d = 10
s0 = d
N = 1000

X, adj = generate(d, s0, N, GP=True)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001
pruning = "Fast"
threshold = 0.05 # None for CAM
sid = bool(d<=200)

# Test: provo a elevare a esponenziale la funzione discriminatoria,e  diminuire accordingly threshold
A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold = threshold)
top_order_err = num_errors(top_order_SCORE, adj)
pretty_evaluate(pruning, threshold, adj, A_SCORE, top_order_err, SCORE_time, tot_time, sid)


fn, fp, rev = edge_errors(A_SCORE, adj)
shd = SHD(A_SCORE, adj)
# FAST logs
with open(f'../logs/test/test_logs_fast.txt', 'w') as f:
    f.writelines(f'SHD: {shd}\n')
    f.writelines(f'False negative: {fn}\n')
    f.writelines(f'False positive: {fp}\n')
    f.writelines(f'Reversed: {rev}\n')
    f.writelines(f'SCORE time: {round(SCORE_time, 2)}s\n')
    f.writelines(f'Total time: {round(tot_time, 2)}s\n')
    f.writelines(f'Topological ordering errors: {num_errors(top_order_SCORE, adj)}\n')

# CAM
start = time.time()
A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
cam_time = time.time() - start
tot_time += cam_time
shd = SHD(A_SCORE, adj)
fn, fp, rev = edge_errors(A_SCORE, adj)
with open(f'../logs/test/test_logs_fastcam.txt', 'w') as f:
    f.writelines(f'SHD: {shd}\n')
    f.writelines(f'False negative: {fn}\n')
    f.writelines(f'False positive: {fp}\n')
    f.writelines(f'Reversed: {rev}\n')
    f.writelines(f'SCORE time: {round(SCORE_time, 2)}s\n')
    f.writelines(f'Total time: {round(tot_time, 2)}s\n')
    f.writelines(f'Topological ordering errors: {num_errors(top_order_SCORE, adj)}\n')