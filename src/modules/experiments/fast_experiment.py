import numpy as np

from sklearn.model_selection import ParameterGrid

from modules.utils import generate, pretty_evaluate, precision, recall
from modules.stein import SCORE
from modules.experiments.experiment import Experiment

class FastExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, data_type, thresholds, k):
        self.d_values = d_values
        self.num_tests = num_tests
        self.s0 = s0
        self.data_type = data_type
        self.thresholds = thresholds
        self.k = k
        self.output_file = f"../logs/exp/fast_{s0}_{d_values[-1]}.csv"
        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'threshold', 'precision', 'recall', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'threshold': self.thresholds, 'k': [self.k]}))

    def config_logs(self, run_logs, sid):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] == 'threshold':
                logs.append(round(m, 5))
            elif not sid and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        self.logs.append(logs)


    def fast(self, X, adj, eta_G, eta_H, threshold, d, s0, N, sid, run_logs):
        """
        Run SCORE with Fast as pruning algorithm. Update logs
        """
        A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, pruning="Fast", threshold=threshold, K=self.k)
        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, sid)
        precision_metric = precision(s0, fn, fp)
        recall_metric = recall(s0, fn)
        print(pretty_evaluate("Fast", threshold, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, sid, s0))
        run_logs.append([d, s0, N, threshold, precision_metric, recall_metric, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        return A_SCORE, top_order_SCORE, SCORE_time, tot_time


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        threshold = params['threshold']
        s0 = self.set_s0(d)
        sid = self.compute_SID(d)

        run_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            self.fast(X, adj, eta_G, eta_H, threshold, d, s0, N, sid, run_logs)

        self.config_logs(run_logs, sid)
        self.save_logs()
