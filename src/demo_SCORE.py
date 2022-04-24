from modules.stein import *
from modules.utils import generate, pretty_evaluate
import cdt


# Data generation paramters
graph_type = 'ER'
d = 1000
s0 = d
N = 1000

X, adj = generate(d, s0, N, GP=True)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001
pruning = "Fast"
threshold = 0.000 # None for CAM
sid = bool(d<=200)
K=5
pns=None

# """
# Keep trehshold very low, such that it looks like zero. And select K according to how much you
# are willing to wait. 
# The key point is: I know exactly how much I am going to wait, because for d nodes and K param, I have 
# d*K edges to check with CAM, that's it.
# This way the threshold loses its improtance, K is a much more meaningful parameter + it is already in CAM
# and finally, the trick I use retains it value, become the crucial point. 
# Could I do same with pns setting it to 5? Yes, but it is way slower

# plus, I have apriori control on running time
# If CAM runs with O(.) complexity, than I can get an expected running time. 
# """

# Test: provo a elevare a esponenziale la funzione discriminatoria,e  diminuire accordingly threshold
A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold=threshold, pns=pns, K=K)
top_order_err = num_errors(top_order_SCORE, adj)
pretty = pretty_evaluate(pruning, threshold, adj, A_SCORE, top_order_err, SCORE_time, tot_time, sid)
print(pretty)

# CAM logs
if pruning == "CAM":
   with open(f'../logs/test/test_logs_cam.txt', 'a+') as f:
    f.writelines(pretty) 

else:
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