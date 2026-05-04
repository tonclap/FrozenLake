# Стратегия достижения 100% на FrozenLake

## Диагностика текущего потолка (99.79%)

### Паттерн провалов

Все 8 проваленных карт имеют одну общую черту: **кратчайший_путь = 6 шагов** при высокой плотности дыр (4–7 из 14 ячеек). Это не случайность.

Манхэттенское расстояние от S(0,0) до G(3,3) равно 6, значит **кратчайший путь = Манхэттенское расстояние** — нет ни одного «обходного» шага. Любое отклонение от единственного оптимального маршрута немедленно ведёт в дыру или требует возврата. Именно эти карты образуют «слепую зону» текущей модели.

### Почему модель застряла

1. **Недостаточная экспозиция к экстремально трудным картам.** Параметр `p` в конфигурации означает вероятность ячейки быть *замёрзшей* (не дырой). После curriculum обучение идёт при `p ∈ [0.78, 0.82]`, что соответствует 18–22% дыр. Проваленные карты имеют плотность дыр 28–50% — этот диапазон почти не встречается в обучении.

2. **Reward shaping мешает для «narrow corridor» карт.** `improvement_factor * ΔManhattan` хорошо работает для карт с запасом пространства, но на картах с единственным путём (path=6) любое движение в сторону «улучшения» по Манхэттену может вести в тупик.

3. **Observation: Manhattan distance вместо BFS distance.** Канал 2 (манхэттенская дистанция) не отражает реальную сложность маршрута. На карте с 7 дырами Манхэттен до цели = 2, но BFS = 6 — сигнал вводит в заблуждение.

4. **Текущий finetune.py имеет критические недостатки:** нет reward shaping, replay буфер 1000 (слишком мал), target network не обновляется, epsilon фиксирован 0.05 (недостаточно исследует 8 карт).

---

## Решение: двухэтапный подход

### Этап 1 — Переобучение с улучшенным curriculum (→ ~99.9%+)

Взять параметры лучшей модели за основу и внести три изменения:

#### 1.1 Инъекция «трудных» карт в пост-curriculum фазу

В `trainer.py`, функция `train_model`, заменить генерацию карты для эпизодов после `CURRICULUM_EPISODES`:

```python
# Было:
current_p = random.uniform(config["POST_CURRICULUM_P_MIN"], config["POST_CURRICULUM_P_MAX"])

# Стало (добавить hard injection с вероятностью HARD_INJECTION_PROB):
if random.random() < config.get("HARD_INJECTION_PROB", 0.0):
    current_p = random.uniform(0.50, 0.65)  # экстремально трудные карты
else:
    current_p = random.uniform(config["POST_CURRICULUM_P_MIN"], config["POST_CURRICULUM_P_MAX"])
```

В конфиге добавить `"HARD_INJECTION_PROB": 0.25` — каждый 4-й эпизод после curriculum генерировать карту с p=0.50–0.65, что воспроизводит плотности дыр проваленных карт.

#### 1.2 Замена канала 2: BFS-расстояние вместо Манхэттена

В `dqn_model.py`, `AugmentedObservationWrapperCNN.__init__`, вместо манхэттенской карты предвычислять **реальную BFS-дистанцию** от каждой ячейки до G:

```python
# Предвычисление BFS-расстояния от каждой клетки до цели
from collections import deque
self.bfs_map = np.full((self.nrow, self.ncol), fill_value=1.0, dtype=np.float32)
goal_r, goal_c = self.nrow - 1, self.ncol - 1
queue = deque([(goal_r, goal_c, 0)])
visited = {(goal_r, goal_c)}
max_dist = self.nrow + self.ncol - 2  # нормировочный делитель
while queue:
    r, c, d = queue.popleft()
    self.bfs_map[r, c] = d / max_dist
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.nrow and 0 <= nc < self.ncol and (nr,nc) not in visited:
            if self.map[nr][nc] in ['S','F','G']:
                visited.add((nr, nc))
                queue.append((nr, nc, d + 1))
```

Дыры получат значение 1.0 (максимальное расстояние), свободные ячейки — реальное число шагов. Это даёт модели точный «навигационный сигнал» вместо геометрической эвристики.

#### 1.3 Оптимальные параметры

Взять параметры лучшей модели без изменений:

```
LR=0.00015051
LR_DECAY=0.99978
EPS_DECAY=0.99873
PLANNING_STEPS=10
GAMMA=0.9365
SOFT_TAU=0.00813
STEP_PENALTY=0.01115
IMPROVEMENT_FACTOR=0.01143
REVISIT_PENALTY=0.05300
N_STEP=3
BATCH_SIZE=256
REPLAY_CAPACITY=55000
NUM_EPISODES=40000        # увеличить с 30000
CURRICULUM_EPISODES=30000 # оставить 30000
HARD_INJECTION_PROB=0.25  # новый параметр
```

Увеличение NUM_EPISODES до 40000 при 30000 curriculum даёт 10000 эпизодов пост-curriculum фазы (против ~0 в текущей реализации, где CURRICULUM_EPISODES = NUM_EPISODES).

---

### Этап 2 — Целевой fine-tuning на проваленных картах (→ 100%)

После достижения ≥99.8% на Этапе 1 запустить исправленный fine-tuning.

#### Исправления finetune.py

**Проблема 1: Нет reward shaping**
```python
# Добавить reward shaping аналогично trainer.py
improvement_reward = config["IMPROVEMENT_FACTOR"] * max(prev_distance - curr_distance, 0)
revisit_penalty = config["REVISIT_PENALTY"] if curr_pos in visited else 0.0
shaping_reward = improvement_reward - revisit_penalty - config["STEP_PENALTY"]
total_step_reward = reward + shaping_reward
```

**Проблема 2: Replay буфер слишком мал**
```python
memory = ReplayMemory(capacity=20000)  # было 1000
```

**Проблема 3: Target network не обновляется**
```python
# После каждого вызова optimize_model уже вызывается soft_update внутри —
# но нужно передавать TRAINABLE target_model, не frozen deepcopy.
# Изменить: target_model = copy.deepcopy(model); target_model.train()
# и обновлять через soft_update каждый episode
```

**Проблема 4: Epsilon слишком мал для исследования 8 карт**
```python
# Использовать убывающий epsilon
eps_start = 0.20
eps_decay = 0.9995
eps_end = 0.03
eps_threshold = max(eps_end, eps_start * (eps_decay ** episode))
```

**Проблема 5: Отсутствует контроль катастрофического забывания**
```python
# Каждые 500 эпизодов тестировать ВСЕ 3828 карт
# Если success_rate падает ниже 99.5% — остановить и вернуть предыдущий checkpoint
if episode % 500 == 0:
    s, t, rate = test_current_model(model, 4, device)
    if rate < 99.5 and prev_rate >= 99.5:
        logging.warning("Catastrophic forgetting detected! Rate: %.2f%%", rate)
        model.load_state_dict(best_state_dict)
        break
```

#### Параметры fine-tuning

```
num_episodes=10000
max_episode_steps=200    # увеличить с 100: трудным картам нужно больше попыток
fine_tune_lr = config["LR"] * 0.05  # ещё осторожнее, было 0.1
replay_capacity = 20000
epsilon: 0.20 → 0.03 (экспоненциальный decay)
```

---

### Этап 3 — Альтернатива: Prioritized Experience Replay (PER)

Если после этапов 1–2 остаются провалы, внедрить PER в `trainer.py`.

PER заменяет равномерную выборку из `ReplayMemory` на выборку, пропорциональную TD-ошибке. Переходы с высокой ошибкой (сложные карты) воспроизводятся чаще автоматически.

```python
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha   # степень приоритизации (0 = равномерно)
        self.beta = beta     # корректировка важностных весов
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, transition, td_error=None):
        priority = (abs(td_error) + 1e-6) ** self.alpha if td_error else max(self.priorities.max(), 1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.memory)]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.memory[i] for i in indices], weights, indices
```

---

## Сводка рекомендаций по приоритету

| Приоритет | Действие | Ожидаемый эффект | Сложность |
|-----------|----------|-----------------|-----------|
| 1 (быстрый выигрыш) | Исправить finetune.py (reward shaping + большой буфер + epsilon decay + anti-forgetting) | 99.79% → 100% | Низкая |
| 2 (основное) | Переобучить с `HARD_INJECTION_PROB=0.25` + 40k эпизодов | 99.79% → ~99.9% без finetune | Средняя |
| 3 (улучшение observation) | BFS-расстояние вместо Манхэттена в канале 2 | +0.1–0.2% на hard maps | Средняя |
| 4 (долгосрочное) | Prioritized Experience Replay | Устойчивое улучшение на редких паттернах | Высокая |

---

## Конкретный план действий

```bash
# Шаг 1: Исправить и запустить finetune на текущем лучшем чекпойнте
python finetune.py models/<best_checkpoint>.pth models/0998_failed_maps.json \
    --num-episodes 10000

# Шаг 2: Проверить результат
python test.py models/<best_checkpoint>_finetuned.pth

# Если не 100% → Шаг 3: Переобучение с новыми параметрами
python experiment.py --num-experiments 5  # с HARD_INJECTION_PROB=0.25

# Шаг 4: Fine-tune лучшего из новых экспериментов
python finetune.py <new_best>.pth failed_maps.json --num-episodes 10000
```

---

## Почему именно эти карты сложны: теоретическое обоснование

На карте 4×4 с path_len=6 = Manhattan(S,G) = 6 выполняется:
- **Нет «slack»**: каждый шаг должен быть в правильном направлении
- **Reward shaping по Манхэттену обманчив**: на узких коридорах улучшение по Манхэттену (−1) часто означает шаг в дыру
- **Revisit penalty контрпродуктивен**: в некоторых картах нужно пройти через одну и ту же ячейку дважды (когда единственный путь образует «Z»-паттерн)

Проверить гипотезу о revisit penalty: для провалившихся карт вычислить, требует ли оптимальный путь повторного посещения ячеек. Если да — **отключить revisit_penalty для карт с path_len=6** во время fine-tuning.
