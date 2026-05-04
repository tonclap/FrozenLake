# Рекомендации по улучшению проекта FrozenLake

## 1. Критический баг в `finetune.py`: target-сеть не изолирована

**Файл:** [finetune.py:79](finetune.py#L79)

```python
# Текущий код (неверно)
optimize_model(model, model, memory, optimizer, config)

# Исправление
target_model = type(model)(map_size, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimize_model(model, target_model, memory, optimizer, config)
```

`optimize_model` принимает `policy_net` и `target_net` как отдельные аргументы, но в `finetune.py` оба параметра передаются одним и тем же объектом `model`. Это нарушает стабилизацию обучения: таргет-значения вычисляются той же сетью, которая обновляется в этом же шаге, что приводит к нестабильным градиентам и расходимости Q-значений. Аналогичная проблема — `num_episodes=24` при вызове ([строка 145](finetune.py#L145)), тогда как в сигнатуре функции стоит `num_episodes=2000` — дообучение практически не происходит.

---

## 2. Неверный коэффициент дисконтирования в N-step DQN

**Файл:** [trainer.py:87](trainer.py#L87)

```python
# Текущий код (неверно)
expected_values = (next_state_values * config["GAMMA"] * (1 - done_batch)) + reward_batch

# Исправление
gamma_n = config["GAMMA"] ** config["N_STEP"]
expected_values = (next_state_values * gamma_n * (1 - done_batch)) + reward_batch
```

В N-step DQN накопленная N-шаговая награда уже учтена в `get_n_step_info` со своим дисконтированием. При бутстрапе следующего состояния нужно применять `GAMMA^N`, а не просто `GAMMA`. Сейчас агент недооценивает будущие награды при N > 1 (в конфиге N от 2 до 4), что смещает обучение и замедляет сходимость.

---

## 3. Неэффективная очередь в BFS и повторные вычисления в `utils.py`

**Файл:** [utils.py:17](utils.py#L17) и [utils.py:28](utils.py#L28)

```python
# Текущий код — list.pop(0) это O(n)
queue = [start]
queue.pop(0)

# Исправление — deque.popleft() это O(1)
from collections import deque, lru_cache

queue = deque([start])
queue.popleft()
```

`is_valid_map` вызывается внутри `generate_all_valid_maps` для всех 2^14 = 16 384 комбинаций при каждом запуске тестирования (каждые 3000 эпизодов). Использование `list.pop(0)` даёт O(n) на каждое извлечение из очереди BFS. Замена на `collections.deque` с `popleft()` убирает эту квадратичную сложность. Дополнительно: результат `generate_all_valid_maps` детерминирован для фиксированного `map_size` — его стоит вычислить один раз и кэшировать (например, через `functools.lru_cache` или простую переменную в `Trainer`), а не пересчитывать при каждом тестировании.
