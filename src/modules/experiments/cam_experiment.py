import numpy as np

from sklearn.model_selection import ParameterGrid

from modules.utils import generate, pretty_evaluate
from modules.stein import SCORE
from modules.experiments.experiment import Experiment


class CAMExperiment(Experiment):
    """
    Score with CAM pruning
    """
    def __init__(self, d_values, num_tests, s0, data_type, cam_cutoff):
        self.d_values = d_values
        self.num_tests = num_tests
        self.s0 = s0
        self.data_type = data_type
        self.output_file = f"../logs/exp/cam_{s0}_median_{d_values[-1]}.csv"
        self.logs = []
        self.cam_cutoff = cam_cutoff
        self.columns = [
            'V', 'E', 'N', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]


    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'cutoff': [self.cam_cutoff]}))


    def config_logs(self, run_logs, compute_SID):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif not compute_SID and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        self.logs.append(logs)
        

    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        cam_cutoff = params['cutoff']
        s0 = self.set_s0(d)

        compute_SID = True
        if d > 200:
            compute_SID = False
        
        run_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cutoff=cam_cutoff, pruning="CAM")
            fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, compute_SID)
            pretty_evaluate("CAM", None, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, compute_SID)
            run_logs.append([d, s0, N, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        self.config_logs(run_logs, compute_SID)
        self.save_logs()