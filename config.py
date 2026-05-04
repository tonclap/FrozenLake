import random

# Базовая конфигурация
default_config = {
    "BATCH_SIZE": 256,          # значение будет заменено случайным выбором
    "GAMMA": 0.925,
    "EPS_START": 1.0,
    "EPS_END": 0.1,
    "EPS_DECAY": 0.9963346347770271,
    "LR": 0.000513,
    "LR_DECAY": 0.9995,
    "SOFT_TAU": 0.01,
    "NUM_EPISODES": 30000,
    "REPLAY_CAPACITY": 50000,   # значение будет заменено случайным выбором
    "N_STEP": 3,                # значение будет заменено случайным выбором
    "MAP_SIZE": 4,
    "PLANNING_STEPS": 5,        # будет заменено случайным выбором из grid
    "STEP_PENALTY": 0.01,
    "IMPROVEMENT_FACTOR": 0.1,
    "REVISIT_PENALTY": 0.05,
    "CURRICULUM_EPISODES": 30000,
    "START_P": 0.95,
    "END_P": 0.7,
    # Диапазон сложности карты после curriculum (ранее был захардкожен в trainer.py)
    "POST_CURRICULUM_P_MIN": 0.78,
    "POST_CURRICULUM_P_MAX": 0.82,
    # Гиперпараметры для специального этапа (после curriculum)
    "NEW_LR": 0.00025,
    "NEW_LR_DECAY": 0.9997,
    "NEW_EPS_DECAY": 0.99997,
    "NEW_EPS_END": 0.05,
    "NEW_GAMMA": 0.95,
    "NEW_PLANNING_STEPS": 10,
}

# Диапазоны для случайного выбора значений
grid = {
    "LR": [0.00015, 0.00018],
    "LR_DECAY": [0.99972, 0.99978],
    "EPS_DECAY": [0.9983, 0.9987],
    "PLANNING_STEPS": [10, 11, 12],
    "GAMMA": [0.933, 0.938],
    "SOFT_TAU": [0.0078, 0.0081],
    "STEP_PENALTY": [0.0093, 0.0115],
    "IMPROVEMENT_FACTOR": [0.010, 0.012],
    "REVISIT_PENALTY": [0.050, 0.053],
}

# Диапазоны для дополнительных параметров, вынесенных сюда:
additional_params = {
    "N_STEP": (2, 4),                   # целое число в диапазоне [2,4]
    "BATCH_SIZE": [256, 512],
    "REPLAY_CAPACITY": [45000, 50000, 55000],
}

def generate_experiment_config() -> dict:
    """
    Генерирует конфигурацию для эксперимента,
    выбирая случайные значения для параметров из grid и additional_params.
    """
    config = default_config.copy()
    
    # Для каждого параметра из grid выбираем случайное значение
    for key, vals in grid.items():
        if key == "PLANNING_STEPS":
            config[key] = random.choice(vals)
        else:
            config[key] = random.uniform(min(vals), max(vals))
    
    # Дополнительные параметры
    config["N_STEP"] = random.randint(*additional_params["N_STEP"])
    config["BATCH_SIZE"] = random.choice(additional_params["BATCH_SIZE"])
    config["REPLAY_CAPACITY"] = random.choice(additional_params["REPLAY_CAPACITY"])

    return config


def validate_config(config: dict) -> None:
    """Проверяет допустимость значений конфига и бросает ValueError при нарушении."""
    rules = [
        ("LR",                  lambda v: 0 < v < 1,       "должен быть в (0, 1)"),
        ("LR_DECAY",            lambda v: 0 < v < 1,       "должен быть в (0, 1)"),
        ("GAMMA",               lambda v: 0 < v < 1,       "должен быть в (0, 1)"),
        ("NEW_GAMMA",           lambda v: 0 < v < 1,       "должен быть в (0, 1)"),
        ("EPS_START",           lambda v: 0 <= v <= 1,     "должен быть в [0, 1]"),
        ("EPS_END",             lambda v: 0 <= v <= 1,     "должен быть в [0, 1]"),
        ("EPS_DECAY",           lambda v: 0 < v < 1,       "должен быть в (0, 1)"),
        ("SOFT_TAU",            lambda v: 0 < v <= 1,      "должен быть в (0, 1]"),
        ("BATCH_SIZE",          lambda v: v > 0,            "должен быть положительным"),
        ("REPLAY_CAPACITY",     lambda v: v >= v,           "должен быть положительным"),
        ("N_STEP",              lambda v: v >= 1,           "должен быть >= 1"),
        ("PLANNING_STEPS",      lambda v: v >= 0,           "должен быть >= 0"),
        ("NUM_EPISODES",        lambda v: v > 0,            "должен быть положительным"),
        ("CURRICULUM_EPISODES", lambda v: 0 < v <= config["NUM_EPISODES"],
                                "должен быть в (0, NUM_EPISODES]"),
        ("START_P",             lambda v: 0 < v <= 1,      "должен быть в (0, 1]"),
        ("END_P",               lambda v: 0 < v <= 1,      "должен быть в (0, 1]"),
        ("POST_CURRICULUM_P_MIN", lambda v: 0 < v < 1,     "должен быть в (0, 1)"),
        ("POST_CURRICULUM_P_MAX", lambda v: config["POST_CURRICULUM_P_MIN"] < v <= 1,
                                "должен быть > POST_CURRICULUM_P_MIN"),
        ("STEP_PENALTY",        lambda v: v >= 0,           "должен быть >= 0"),
        ("IMPROVEMENT_FACTOR",  lambda v: v >= 0,           "должен быть >= 0"),
        ("REVISIT_PENALTY",     lambda v: v >= 0,           "должен быть >= 0"),
    ]
    errors = []
    for key, check, msg in rules:
        if key not in config:
            errors.append(f"  {key}: отсутствует в конфиге")
        elif not check(config[key]):
            errors.append(f"  {key}={config[key]}: {msg}")
    if errors:
        raise ValueError("Недопустимые значения конфига:\n" + "\n".join(errors))