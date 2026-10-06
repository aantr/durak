# Behaviour cloning для «Дурака»

`behaviour_cloning.py` обучает policy на JSON-партиях в `games_dataset`.
`generate_games_dataset.py` собирает эти партии существующим C++ симулятором
`game_engine._native`. Правила и формат наблюдения взяты из текущего движка.
Нужны PyTorch из зависимостей проекта и собранный движок:

```bash
venv/bin/python -m pip install --no-build-isolation ./game_engine
```

## Генерация

```bash
venv/bin/python generate_games_dataset.py --games 1000 --workers 4 \
  --strategies random:1 heuristic:2 greedy:1 mcts:4 \
  --iterations 256 --rollout-depth 256 --output games_dataset
```

Все перечисленные стратегии оценивают **каждое состояние одновременно в составе
ансамбля**. Это не случайное назначение одного бота на всю партию. `--workers`
запускает несколько независимых партий в разных процессах.

| Стратегия | Оценка |
| --- | --- |
| `random` | Равномерные вероятности всех легальных ходов |
| `heuristic` | Дешёвые карты, сохранение козырей, атака парными рангами |
| `greedy` | Предпочтение сыграть старшие карты, меньшее предпочтение козырям |
| `mcts` | Оценки существующего ISMCTS на наблюдаемом состоянии |

Сначала оценки каждой стратегии переводятся в вероятности через softmax.
Температура для MCTS — 0.15, для остальных — 1. Затем вероятности усредняются
с заданными весами. MCTS использует `time_limit_ms=0` и фиксированный seed;
число итераций не меньше количества доступных действий, чтобы оценить каждое.
Сырые оценки MCTS — средние баллы симуляций, а не измеренная вероятность победы.

По умолчанию ход семплируется из итогового распределения. `--selection argmax`
выбирает максимум. Для быстрого датасета можно исключить MCTS:

```bash
venv/bin/python generate_games_dataset.py --games 100 --seed 10000 \
  --strategies random:1 heuristic:3 greedy:1
```

Один файл — одна партия: `games_dataset/game_00000000000000000042.json`.
Повторный запуск с теми же seed не перезаписывает файлы; для пополнения укажите
новый диапазон через `--seed`. `--max-steps` ограничивает длину партии (2000 по
умолчанию). Незавершённая партия помечается `truncated`, победитель ей не назначается.

## Обучение и выбор хода

Обычный BC минимизирует `-log π(action | state)` для реально выбранного хода:

```bash
venv/bin/python behaviour_cloning.py train --dataset games_dataset \
  --epochs 20 --batch-size 128 --output runs/bc
```

Чтобы использовать оценки всех ходов, задайте `--target scores`. В этом режиме
цель — полное распределение `action_probabilities`, loss — soft cross entropy.
Это дистилляция ансамбля, отдельная от обычного BC по выбранным действиям:

```bash
venv/bin/python behaviour_cloning.py train --dataset games_dataset \
  --target scores --device cuda --epochs 20 --output runs/bc_scores
```

По умолчанию обучение идёт на CPU, `--device cuda` включает GPU.
`--threads` задаёт CPU-потоки PyTorch, `--workers` — процессы загрузчика.
Датасет загружается по одной партии на worker. Партии и их ходы перемешиваются
на каждой эпохе. Разделение train/validation воспроизводимо по seed; одинаковый
`game_id` всегда попадает в одну выборку. Для validation нужны хотя бы две партии;
отключить её можно через `--validation-fraction 0`.

В выходной директории сохраняются:

- `best.pt` — веса с минимальным validation loss (train loss без validation);
- `last.pt` — веса последней эпохи;
- `metrics.jsonl` — loss, совпадение с действием игрока (`accuracy`) и максимумом
  целевого распределения (`target_accuracy`);
- `split.json` — параметры и списки файлов обеих выборок.

Повторное обучение в той же выходной директории заменяет эти артефакты.
Эти метрики измеряют имитацию, а не winrate; для оценки силы нужны отдельные матчи.

```python
from behaviour_cloning import load_policy
from game_engine import observation_from_state

policy = load_policy("runs/bc/best.pt")
observation = observation_from_state(existing_state, trump="S")
result = policy.suggest(observation)
action = result["action"]  # тот же type/card/target, что в C++ движке

# Можно передать текущий DurakGameState или снимок напрямую:
result = policy.suggest(existing_state, trump="S")
```

`suggest()` предназначен для момента, когда ходит наблюдатель (`turn=0`).
Модель получает актуальный список легальных действий от движка. Скрытый расклад
для вычисления этого списка не меняет допустимых ходов нашей руки и не передаётся
нейросети.

### Live-бот с BC

```bash
venv/bin/python -m game_state.bot --bc-checkpoint runs/bc/best.pt --draw-detections
```

Настройки подключения задаются существующими `--mac-ip`, `--control-port`,
`--video-port`. Например, для локального bridge на портах 12004/12005:

```bash
venv/bin/python -m game_state.bot --mac-ip 127.0.0.1 \
  --control-port 12004 --video-port 12005 \
  --bc-checkpoint runs/bc/best.pt --draw-detections
```

По умолчанию BC работает на CPU; `--bc-device cuda` загружает её на GPU.
`Space` рассчитывает подсказку, `↑` выполняет предложенный ход, `Enter`
включает/выключает автоигру, `R` сбрасывает состояние, `Esc` завершает бота.
Клавиши работают при фокусе окна бота. Без `--bc-checkpoint` используется MCTS;
параметры `--mcts-*` не управляют BC.

Live-снимки пока не содержат истории действий, поэтому модель получает пустую
историю и актуальное состояние с известными картами соперника. Это отличается от
генератора, сохраняющего последние действия, и может влиять на качество игры.
Вероятность действия BC не является оценкой вероятности победы.

Для CLI сохраните наблюдение или один transition в отдельный JSON:

```bash
venv/bin/python behaviour_cloning.py predict --checkpoint runs/bc/best.pt \
  --state observation.json
```

## Модель

Общий CardEncoder: embeddings ранга (16), масти (8), признака козыря (8) и MLP.
DeepSets агрегирует руку, известные карты соперника, биту и открытый нижний козырь.
Пары атака/защита кодируются совместно; для `defend` encoder действия учитывает
и карту из руки, и атакующую карту выбранной цели. Контекст содержит размеры зон,
роль, чей ход, лимит атаки, взятие и завершение подкидывания. GRU обрабатывает
последние 32 публичных действия; отсутствие истории допустимо.

MLP формирует state embedding (256 по умолчанию). Policy head оценивает каждое
легальное действие по state/action embeddings. Padding исключается из softmax
и loss. Порядок карт внутри наборов не влияет на результат.
Это первый этап BC; Q/value heads и IQL пока не реализованы.

## JSON schema v1

В репозитории на момент реализации готовых JSON-партий не было, поэтому генератор
и загрузчик используют следующий общий контракт. Существующие сторонние JSON
нужно привести к нему; произвольные схемы автоматически не угадываются.

```text
{
  schema_version: 1,
  game_id: "game-42",
  seed: 42,
  winner: 0 | 1 | null,
  truncated: bool,
  config: {strategies, selection, iterations, rollout_depth, history_size, max_steps},
  transitions: [{
    player_id: 0 | 1,            # абсолютный номер участника партии
    turn_id: int,               # индекс глобального действия
    state: Observation,
    legal_actions: [{type, card, target}, ...],
    action: {type, card, target},
    action_index: int,
    action_probabilities: [float, ...],
    strategy_evaluations: {
      strategy_name: {scores: [...], probabilities: [...], value_kind, ...}
    },
    policy_id: "ensemble",
    reward: -1 | 0 | 1,
    next_state: Observation,
    next_legal_actions: [...],
    steps_to_next: int,
    done: bool,
    truncated: bool
  }, ...]
}
```

Все массивы оценок соответствуют порядку `legal_actions`. Загрузчику BC нужны
`schema_version`, `game_id`, непустой `transitions`, а в каждом transition —
`state`, `legal_actions`, `action`; для `--target scores` также нужны
`action_probabilities`. При наличии `action_index` проверяется его согласованность.

`Observation` совпадает с `game_engine.observation_from_state()`:
`hand`, `known_opponent`, `discard`, `table` (пары `attack`/`defense`),
`deck_count`, `opponent_count`, `trump`, `bottom_trump`, `attacker`, `turn`,
`attack_limit`, `taking`, `attacker_passed`, `simultaneous_winner`.
Дополнительно разрешено `history`: события с `type`, `card`, `target_card`,
`target`, относительным `player_id`. Карты — `6C` … `AS`, масти — `CDHS`.

Каждое наблюдение записано с точки зрения автора хода: он всегда игрок 0,
соперник — 1. Настоящие скрытые руки и порядок колоды в JSON не сохраняются.
Известные карты отслеживаются по публичным взятиям, сыгранным картам и добору
открытого нижнего козыря. Внешний `player_id` transition остаётся абсолютным.

`next_state` — следующее решение **того же игрока**, а не состояние перед ходом
противника. Поэтому `steps_to_next` может быть больше 1. Последний переход каждого
игрока получает награду +1/-1 при завершении партии; остальные получают 0.
Терминальные `next_legal_actions` пусты. При обрезании последние переходы игроков
имеют `truncated=true`, `done=false`, награду 0 и наблюдение на момент остановки;
если ходит другой игрок, список следующих действий пуст. Такие границы нельзя
считать обычными следующими решениями при будущем RL. BC не использует next/reward.

Проверки:

```bash
venv/bin/python -B -m unittest discover -s tests -p test_behaviour_cloning.py -v
```
