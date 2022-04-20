from modules.experiments.cam_experiment import CAMExperiment
from modules.experiments.fast_experiment import FastExperiment
from modules.experiments.fastcam_experiment import FastCAMExperiment


if __name__ == "__main__":
    """
    Run experiments and store logs
    """
    # General
    data_type = 'Gauss'
    num_tests = 10

    # Regression parameters
    N = 1000
    eta_G = 0.001
    eta_H = 0.001

    # Experiments parameters
    pruning = "CAM" # ["Fast", "FastCAM", "CAM"]
    d_values = [200, 500]
    thresholds = [0.05, 0.1]
    cam_cutoff = 0.001

    if pruning == "Fast":
        for s0 in ['d', '4d']:
            experiment = FastExperiment(d_values, num_tests, s0, data_type, thresholds)
            experiment.run_experiment(N, eta_G, eta_H)

    elif pruning == "CAM":
        for s0 in ['d', '4d']:
            experiment = CAMExperiment(d_values, num_tests, s0, data_type, cam_cutoff, pns=20)
            experiment.run_experiment(N, eta_G, eta_H)

    elif pruning == "FastCAM":
        for s0 in ['d']:
            experiment = FastCAMExperiment(d_values, num_tests, s0, data_type, cam_cutoff, thresholds)
            experiment.run_experiment(N, eta_G, eta_H)

    else:
        raise ValueError("Unknown pruning method")