from modules.stein import *
from modules.utils import generate, pretty_evaluate

# Data generation paramters
graph_type = 'ER'
d = 50
s0 = 4*d
N = 1000

X, adj = generate(d, s0, N, GP=True)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001
pruning = "Fast"
threshold = 0.01 # None for CAM
sid = bool(d<=200)
K=1

# Test: provo a elevare a esponenziale la funzione discriminatoria,e  diminuire accordingly threshold
A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold=threshold, K=K)
top_order_err = num_errors(top_order_SCORE, adj)
pretty = pretty_evaluate(pruning, threshold, adj, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
print(pretty)

# CAM logs
if pruning == "Fast":
   with open(f'../logs/test/test_logs_cam.txt', 'a+') as f:
    f.writelines(pretty) 

elif pruning == "FastCAM":
# Fast logs
    with open(f'../logs/test/test_logs_fast.txt', 'a+') as f:
        f.writelines(pretty)

    # FastCAM logs
    start = time.time()
    A_SCORE = cam_pruning(A_SCORE, X, cam_cutoff)
    cam_time = time.time() - start
    tot_time += cam_time
    pretty = pretty_evaluate(pruning + "CAM", threshold, adj, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
    with open(f'../logs/test/test_logs_fastcam.txt', 'a+') as f:
        f.writelines(pretty)