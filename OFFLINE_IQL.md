# Offline IQL для «Дурака»

`offline_iql.py` обучает модель на существующих JSON-партиях `games_dataset`.
Во время обучения партии не генерируются, поиск MCTS не вызывается. Модель
использует реальные выбранные действия, награды и переходы из датасета.
`action_probabilities` и оценки стратегий не служат целями IQL.

Реализация следует [статье IQL](https://arxiv.org/abs/2110.06169) и
[реализации авторов](https://github.com/ikostrikov/implicit_q_learning),
с дискретной policy и softmax по легальным действиям «Дурака».

## Обучение

Из корня проекта, со стартом из обученной BC-модели:

```bash
venv/bin/python -B offline_iql.py train \
  --dataset games_dataset \
  --init-bc runs/bc_scores/best.pt \
  --output runs/iql \
  --device cuda \
  --epochs 20 --batch-size 128
```

Проверка силы игры без телефона:

```bash
venv/bin/python -B offline_iql.py evaluate   --checkpoint runs/iql/best.pt   --games 100 --opponent heuristic
```

Для CPU замените `--device cuda` на `--device cpu` (значение по умолчанию).
Без `--init-bc` модель обучается с нуля. Размер encoder берётся из BC-checkpoint;
без BC используется `hidden_size=256`, его можно задать через `--hidden-size`.
В actor копируются все BC-веса, в Q/V — только encoders; Q/V heads обучаются с нуля.
У actor, каждой Q-сети и V-сети свои параметры encoder, градиенты между ними не смешиваются.
Старые BC-checkpoint совместимы с вынесенным методом `state_features`.

Если рядом с BC-checkpoint есть `split.json`, его разделение train/validation
используется автоматически. Список JSON в `--dataset` должен точно совпадать с ним:
это предотвращает случайную проверку на партиях, уже изученных BC. Явный split
можно передать через `--split path/to/split.json`; он также проверяется на
совпадение файлов и отсутствие общих `game_id` между train и validation.
При самостоятельном обучении без split работает `--validation-fraction 0.15`.
При переносе проекта пути в старом manifest нужно обновить.

Параметры:

| Флаг | По умолчанию | Назначение |
| --- | --- | --- |
| `--gamma` | 0.99 | Дисконт между решениями одного игрока |
| `--expectile` | 0.7 | Верхний expectile для V, допустимо 0.5 < τ < 1 |
| `--beta` | 3 | Множитель advantage в `exp(beta * advantage)` |
| `--max-weight` | 100 | Максимальный вес примера при обучении actor |
| `--tau` | 0.005 | Доля новых Q-весов при EMA target-сетей |
| `--actor-lr` | 0.0003 | Скорость обучения policy |
| `--critic-lr` | 0.0003 | Скорость обучения двух Q-сетей |
| `--value-lr` | 0.0003 | Скорость обучения V |
| `--history-mode` | none | `none`: без истории; `dataset`: использовать историю из JSON |
| `--threads` | 2 | CPU-потоки PyTorch |
| `--workers` | 0 | Процессы загрузчика, партии читаются по одной на worker |
| `--seed` | 42 | Инициализация и перемешивание |

`history-mode=none` соответствует текущему live-боту: он передаёт актуальное
состояние, но не надёжную историю действий. Режим сохраняется в checkpoint и
применяется также при predict/evaluate. С `history-mode=dataset` live-бот всё равно
не располагает историей; для него такой режим создаёт различие с обучением.

Для быстрой проверки всего цикла можно добавить `--max-batches 2 --epochs 2`
и сохранить результат в `--output runs/iql_smoke`. Это ограничивает и train,
и validation на каждой эпохе; такие метрики и веса служат только проверке запуска.
Для полноценного обучения уберите `--max-batches`.

Артефакты:

- `last.pt` — последний checkpoint: actor, Q1/Q2, V, target Q, состояния optimizers;
- `best.pt` — checkpoint с наименьшим validation `critic_loss` (train без validation);
- `metrics.jsonl` — метрики каждой эпохи;
- `split.json` — параметры запуска и списки партий.

`best.pt` выбран по диагностической ошибке Q, **не по winrate**. Targets Q меняются
в ходе обучения, поэтому уменьшение `critic_loss` само по себе не доказывает рост
силы игры. Для запуска разумно сравнить `last.pt` и `best.pt` в матчах.
Повторный запуск в той же output-директории заменяет артефакты. Автоматическое
продолжение прерванного обучения в этой версии не реализовано.

## Что именно оптимизируется

Для состояния `s` и сыгранного действия `a`:

```text
q_target = min(Q1_target(s, a), Q2_target(s, a))
L_V = mean(weight * (q_target - V(s))²)
weight = expectile, если q_target > V(s), иначе 1 - expectile

y = reward + gamma * V(next_state)          # только для nonterminal
y = reward                                # terminal
L_Q = mean((Q1(s,a) - y)² + (Q2(s,a) - y)²)

advantage = q_target - V(s)
actor_weight = min(exp(beta * advantage), max_weight)
L_actor = mean(-actor_weight * log policy(a|s))
```

Порядок шага: обновить V; вычислить advantage с обновлённой V; обновить actor;
обновить Q; плавно обновить target Q. Targets и actor weights вычисляются без
градиентов. Q оценивает только действие из датасета. Actor нормирует вероятности
по всему списку legal_actions; padding исключён. Во время игры выбирается
`argmax policy`, Q-поиск по действиям не выполняется.

В отличие от весов стратегий генератора, **веса примеров IQL меняются в процессе
обучения**: выше вес у действий с положительным advantage. Качество исходного
датасета и результаты партий остаются существенными.

## Переходы и награды

Используется schema v1 из `generate_games_dataset.py`. Обязательны `state`,
`action`, `legal_actions`, `reward`, `next_state`, `next_legal_actions`, `done`,
`truncated`, `steps_to_next`. Игрок-наблюдатель всегда 0.

`next_state` — следующее решение **того же игрока**, между ними соперник мог
сделать несколько действий. Поэтому знак bootstrap не меняется. `gamma`
применяется один раз на переход; `steps_to_next` — число глобальных действий
для диагностики, степень gamma по нему не вычисляется.

Награда ±1 назначена последнему решению победителя/проигравшего, остальные — 0.
Терминальные next-state не передаются encoder, и bootstrap для них равен нулю.
Обрезанные граничные переходы (`truncated=true`) пропускаются, поскольку следующий
ход того же игрока в них ещё может отсутствовать. Предыдущие корректные переходы
таких партий используются. Обрезание не превращается в поражение или ничью.
Некорректные награды, terminal-флаги, нелегальные действия и `turn!=0`
в nonterminal next-state отклоняются с именем файла и номером хода.

## Проверка силы игры без iPhone

```bash
venv/bin/python -B offline_iql.py evaluate \
  --checkpoint runs/iql/last.pt --games 100 --opponent heuristic --seed 100000
```

Соперники: `random`, `heuristic`, `greedy`, `mcts` (128 итераций, глубина 128).
Соперник семплирует действия из распределения выбранной стратегии, как генератор.
IQL выбирает максимум policy. Участник IQL чередуется между 0 и 1, первоначальный
атакующий определяется правилами движка. Seed следует выбирать вне обучающего
набора. Выводятся победы, поражения, обрезанные партии и доля побед среди завершённых.
Большая доля обрезаний делает winrate среди завершённых нерепрезентативным.

## Live-бот

```bash
venv/bin/python -B -m game_state.bot \
  --mac-ip 127.0.0.1 --control-port 12004 --video-port 12005 \
  --iql-checkpoint runs/iql/last.pt --draw-detections
```

Укажите свои параметры bridge. По умолчанию IQL работает на CPU, для GPU добавьте
`--iql-device cuda`. `--bc-checkpoint` и `--iql-checkpoint` взаимоисключающие.
`Space` — подсказка, `↑` — выполнить ход, `Enter` — автоигра, `R` — сброс,
`Esc` — выход. IQL загружается один раз в фоновом процессе. В отображаемом
результате используется вероятность выбора действия policy, а не вероятность победы.

Предсказание по JSON-наблюдению/transition:

```bash
venv/bin/python -B offline_iql.py predict \
  --checkpoint runs/iql/last.pt --state observation.json
```

Проверки формул, переходов, обучения и интеграции с ботом:

```bash
venv/bin/python -B -m unittest discover -s tests -p test_offline_iql.py -v
```
