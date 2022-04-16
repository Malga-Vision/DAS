import abc
import cdt
import time
import pandas as pd

from modules.utils import edge_errors
from modules.stein import num_errors

class Experiment(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run_config(self, params, N, eta_G, eta_H):
        """Run self.num_tests experiment on a given configuration"""
        raise NotImplementedError

    @abc.abstractmethod
    def run_experiment(num_tests, N, eta_G, eta_H):
        """Run experiments on all configurations for a given pruning algorithm"""
        raise NotImplementedError

    @abc.abstractmethod
    def config_logs(self, run_logs, compute_SID):
        """Logs of a tested configuration"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self):
        """Return ParameterGrid with all the combination of testing parameters"""
        raise NotImplementedError

    def run_experiment(self, N, eta_G, eta_H):
        start = time.time()
        param_grid = self.get_params()
        run = 0
        tot_runs = len(param_grid)
        for params in param_grid:
            print(f"Start experiment {run+1}/{tot_runs} with parameters: {params}")
            self.run_config(params, N, eta_G, eta_H)

            run += 1
            print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------ \n")

    def metrics(self, A_SCORE, adj, top_order_SCORE, compute_SID=True):
            fn, fp, rev = edge_errors(A_SCORE, adj)
            SHD = sum((fn, fp, rev))

            SID = -1
            if compute_SID:
                SID = int(cdt.metrics.SID(target=adj, pred=A_SCORE))
            top_order_errors = num_errors(top_order_SCORE, adj)
            return fn, fp, rev, SHD, SID, top_order_errors

    def set_s0(self, d):
        if self.s0 == 'd':
            s0 = d
        elif self.s0 == '4d':
            s0 = 4*d
        else:
            raise ValueError("Forbidden s0 value")
        return s0

    def compute_SID(self, d):
        sid = True
        if d > 200:
            sid = False
        return sid

    def save_logs(self):
        df = pd.DataFrame(self.logs, columns =self.columns)
        df.to_csv(self.output_file)
