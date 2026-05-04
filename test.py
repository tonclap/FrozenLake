import argparse
import json
import os
import sys

import gym
import torch

from dqn_model import AugmentedObservationWrapperCNN, DQNCNN
from utils import generate_all_valid_maps, map_complexity, set_seed, test_on_map


def cluster_label(complexity: dict) -> str:
    holes = complexity["holes"]
    path = complexity["shortest_path"]
    if holes <= 3:
        difficulty = "easy"
    elif holes <= 6:
        difficulty = "medium"
    else:
        difficulty = "hard"
    if path <= 6:
        route = "short"
    elif path <= 9:
        route = "medium"
    else:
        route = "long"
    return f"{difficulty}_route-{route}"


def print_failure_report(failed_maps: list, top_n: int) -> None:
    if not failed_maps:
        print("\nВсе карты пройдены успешно — нет провалов.")
        return

    enriched = [(m, map_complexity(m)) for m in failed_maps]
    enriched.sort(key=lambda x: (x[1]["holes"], -x[1]["shortest_path"]), reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP-{min(top_n, len(enriched))} сложнейших проваленных карт")
    print(f"{'='*60}")
    for i, (m, c) in enumerate(enriched[:top_n], 1):
        print(f"\n#{i}  дыр={c['holes']}  плотность={c['hole_density']}  "
              f"кратчайший_путь={c['shortest_path']}  кластер={cluster_label(c)}")
        for row in m:
            print("  " + " ".join(row))

    clusters: dict[str, list] = {}
    for m, c in enriched:
        label = cluster_label(c)
        clusters.setdefault(label, []).append(c)

    print(f"\n{'='*60}")
    print("Статистика провалов по кластерам сложности")
    print(f"{'='*60}")
    print(f"{'Кластер':<28} {'Провалов':>8} {'Ср. дыр':>10} {'Ср. путь':>10}")
    print("-" * 60)
    for label in sorted(clusters):
        items = clusters[label]
        avg_holes = sum(x["holes"] for x in items) / len(items)
        avg_path = sum(x["shortest_path"] for x in items) / len(items)
        print(f"{label:<28} {len(items):>8} {avg_holes:>10.1f} {avg_path:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Тестирование чекпоинта DQN на всех валидных картах")
    parser.add_argument("checkpoint", help="Путь к файлу чекпоинта (.pth)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed для воспроизводимости (опционально)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Сколько проваленных карт показать в отчёте (по умолчанию 10)")
    parser.add_argument("--failed-maps-out", default="failed_maps.json",
                        help="Куда сохранить JSON с проваленными картами (по умолчанию failed_maps.json)")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_size = 4
    n_actions = 4

    model = DQNCNN(map_size, n_actions).to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Файл чекпоинта {args.checkpoint} не найден.")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()
    saved_seed = checkpoint.get("seed")
    print(f"Модель загружена из {args.checkpoint}"
          + (f"  (обучена с seed={saved_seed})" if saved_seed is not None else ""))

    valid_maps = generate_all_valid_maps(map_size)
    total_maps = len(valid_maps)
    print(f"Найдено {total_maps} валидных карт для {map_size}x{map_size}.")

    successful = 0
    failed_maps = []
    for map_desc in valid_maps:
        reward = test_on_map(model, map_desc, device, AugmentedObservationWrapperCNN)
        if reward >= 1.0:
            successful += 1
        else:
            failed_maps.append(map_desc)

    success_rate = (successful / total_maps) * 100
    print(f"\nУспешно пройдено: {successful}/{total_maps}  ({success_rate:.2f}%)")
    print(f"Провалено: {len(failed_maps)} карт")

    with open(args.failed_maps_out, "w", encoding="utf-8") as f:
        json.dump(failed_maps, f, ensure_ascii=False, indent=4)
    print(f"Проваленные карты сохранены в {args.failed_maps_out}")

    print_failure_report(failed_maps, args.top_n)


if __name__ == "__main__":
    main()
