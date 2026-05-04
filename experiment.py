import csv
import time
import random
import logging

from trainer import train_model
from config import generate_experiment_config

def main():
    logging.basicConfig(filename="results/experiment.log",
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting experiments")

    results_file = "results/experiment.csv"
    test_results_file = "results/test.csv"
    header = ["Exp#", "LR", "LR_DECAY", "EPS_DECAY", "PLANNING_STEPS", "GAMMA", "SOFT_TAU",
              "STEP_PENALTY", "IMPROVEMENT_FACTOR", "REVISIT_PENALTY", "N_STEP", "BATCH_SIZE", "REPLAY_CAPACITY", "BestOverallSuccess"]

    with open(results_file, mode="w", newline="") as csv_file, open(test_results_file, mode="w", newline="") as test_csv_file:
        writer_csv = csv.writer(csv_file)
        test_writer = csv.writer(test_csv_file)
        writer_csv.writerow(header)
        test_writer.writerow(["Exp#", "Episode", "Successful_maps", "Total_maps", "Success_rate"])
        csv_file.flush()
        test_csv_file.flush()

        for exp_count in range(1, 31):
            exp_config = generate_experiment_config()
            
            logging.info("Experiment %d with parameters: LR=%.6f, LR_DECAY=%.6f, EPS_DECAY=%.6f, PLANNING_STEPS=%d, GAMMA=%.3f, SOFT_TAU=%.3f, STEP_PENALTY=%.3f, IMPROVEMENT_FACTOR=%.3f, REVISIT_PENALTY=%.3f, N_STEP=%d, BATCH_SIZE=%d, REPLAY_CAPACITY=%d", 
                         exp_count, exp_config["LR"], exp_config["LR_DECAY"], exp_config["EPS_DECAY"], exp_config["PLANNING_STEPS"],
                         exp_config["GAMMA"], exp_config["SOFT_TAU"], exp_config["STEP_PENALTY"], exp_config["IMPROVEMENT_FACTOR"],
                         exp_config["REVISIT_PENALTY"], exp_config["N_STEP"], exp_config["BATCH_SIZE"], exp_config["REPLAY_CAPACITY"])
            
            start_time = time.time()
            best_success = train_model(exp_config, experiment_number=exp_count, test_writer=test_writer)
            elapsed = time.time() - start_time
            writer_csv.writerow([exp_count, exp_config["LR"], exp_config["LR_DECAY"], exp_config["EPS_DECAY"],
                                 exp_config["PLANNING_STEPS"], exp_config["GAMMA"], exp_config["SOFT_TAU"],
                                 exp_config["STEP_PENALTY"], exp_config["IMPROVEMENT_FACTOR"], exp_config["REVISIT_PENALTY"],
                                 exp_config["N_STEP"], exp_config["BATCH_SIZE"], exp_config["REPLAY_CAPACITY"],
                                 best_success])
            csv_file.flush()
        logging.info("All experiments finished")

if __name__ == "__main__":
    main()