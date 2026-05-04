import os
import sys
import gym
import torch
import json

from dqn_model import AugmentedObservationWrapperCNN, DQNCNN
from utils import generate_all_valid_maps, test_on_map

def main():
    if len(sys.argv) != 2:
        print("Использование: python test_checkpoint.py <путь_к_чекпойнту>")
        return

    checkpoint_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_size = 4
    n_actions = 4

    model = DQNCNN(map_size, n_actions).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["policy_net_state_dict"])
        model.eval()
        print(f"Модель загружена из {checkpoint_path}")
    else:
        print(f"Файл модели {checkpoint_path} не найден.")
        return

    valid_maps = generate_all_valid_maps(map_size)
    total_maps = len(valid_maps)
    print(f"Найдено {total_maps} валидных карт для {map_size}x{map_size}.")

    successful = 0
    failed_maps = []  # Список карт, на которых модель ошиблась
    for map_desc in valid_maps:
        reward = test_on_map(model, map_desc, device, AugmentedObservationWrapperCNN)
        if reward >= 1.0:
            successful += 1
        else:
            failed_maps.append(map_desc)

    success_rate = (successful / total_maps) * 100
    print(f"Успешно пройдено: {successful} карт, {success_rate:.2f}%")

    # Сохраняем перечень карт в JSON для последующего анализа
    with open("failed_maps.json", "w", encoding="utf-8") as f:
        json.dump(failed_maps, f, ensure_ascii=False, indent=4)
    print("\nПолный перечень неудачных карт сохранён в failed_maps.json")

if __name__ == "__main__":
    main()