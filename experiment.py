import argparse
import csv
import os
import time
import random
import logging

from trainer import train_model
from config import generate_experiment_config, validate_config
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Запуск серии экспериментов DQN на FrozenLake")
    parser.add_argument("--seed", type=int, default=None,
                        help="Глобальный seed воспроизводимости (по умолчанию — случайный)")
    parser.add_argument("--num-experiments", type=int, default=30,
                        help="Количество экспериментов (по умолчанию 30)")
    args = parser.parse_args()

    global_seed = args.seed if args.seed is not None else random.randint(0, 2 ** 31 - 1)
    set_seed(global_seed)

    os.makedirs("results", exist_ok=True)
    logging.basicConfig(filename="results/experiment.log",
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting experiments with global_seed=%d", global_seed)
    print(f"Global seed: {global_seed}")

    results_file = "results/experiment.csv"
    test_results_file = "results/test.csv"
    header = ["Exp#", "Seed", "LR", "LR_DECAY", "EPS_DECAY", "PLANNING_STEPS", "GAMMA",
              "SOFT_TAU", "STEP_PENALTY", "IMPROVEMENT_FACTOR", "REVISIT_PENALTY",
              "N_STEP", "BATCH_SIZE", "REPLAY_CAPACITY", "BestOverallSuccess"]

    with open(results_file, mode="w", newline="") as csv_file, \
         open(test_results_file, mode="w", newline="") as test_csv_file:
        writer_csv = csv.writer(csv_file)
        test_writer = csv.writer(test_csv_file)
        writer_csv.writerow(header)
        test_writer.writerow(["Exp#", "Seed", "Episode", "Successful_maps", "Total_maps", "Success_rate"])
        csv_file.flush()
        test_csv_file.flush()

        for exp_count in range(1, args.num_experiments + 1):
            exp_seed = global_seed + exp_count
            exp_config = generate_experiment_config()
            validate_config(exp_config)

            logging.info(
                "Experiment %d seed=%d: LR=%.6f, LR_DECAY=%.6f, EPS_DECAY=%.6f, "
                "PLANNING_STEPS=%d, GAMMA=%.3f, SOFT_TAU=%.3f, STEP_PENALTY=%.3f, "
                "IMPROVEMENT_FACTOR=%.3f, REVISIT_PENALTY=%.3f, N_STEP=%d, "
                "BATCH_SIZE=%d, REPLAY_CAPACITY=%d",
                exp_count, exp_seed,
                exp_config["LR"], exp_config["LR_DECAY"], exp_config["EPS_DECAY"],
                exp_config["PLANNING_STEPS"], exp_config["GAMMA"], exp_config["SOFT_TAU"],
                exp_config["STEP_PENALTY"], exp_config["IMPROVEMENT_FACTOR"],
                exp_config["REVISIT_PENALTY"], exp_config["N_STEP"],
                exp_config["BATCH_SIZE"], exp_config["REPLAY_CAPACITY"],
            )

            start_time = time.time()
            best_success = train_model(exp_config, experiment_number=exp_count,
                                       test_writer=test_writer, seed=exp_seed)
            elapsed = time.time() - start_time
            logging.info("Experiment %d finished in %.1fs, best_success=%.2f%%",
                         exp_count, elapsed, best_success)

            writer_csv.writerow([
                exp_count, exp_seed,
                exp_config["LR"], exp_config["LR_DECAY"], exp_config["EPS_DECAY"],
                exp_config["PLANNING_STEPS"], exp_config["GAMMA"], exp_config["SOFT_TAU"],
                exp_config["STEP_PENALTY"], exp_config["IMPROVEMENT_FACTOR"],
                exp_config["REVISIT_PENALTY"], exp_config["N_STEP"],
                exp_config["BATCH_SIZE"], exp_config["REPLAY_CAPACITY"],
                best_success,
            ])
            csv_file.flush()

        logging.info("All experiments finished")


if __name__ == "__main__":
    main()