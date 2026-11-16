# rl-optimization (v0.1.0)

Исследовательская Python-библиотека для адаптивного управления дифференциальной эволюцией с использованием обучения с подкреплением.

Текущая версия фокусируется на воспроизводимой экспериментальной базе для сравнения пяти вариантов DE:

1. `rand_1` (`DE/rand/1/bin`)
2. `best_1` (`DE/best/1/bin`)
3. `current_to_best_1` (`DE/current-to-best/1/bin`)
4. `random_de` (случайный выбор стратегии по поколениям)
5. `rl_de_q_learning` (Q-learning выбирает стратегию по состоянию поиска)
6. `rl_de_q_learning_stagnation` (Q-learning с состоянием "стадия + стагнация")

## Важная зависимость: thefittest

Базовая реализация DE построена на библиотеке [`thefittest`](https://github.com/sherstpasha/thefittest), версия зафиксирована в проекте:

- `thefittest==0.2.7`

Проект не содержит самостоятельной «с нуля» реализации DE; используются и расширяются механизмы `DifferentialEvolution` из `thefittest`.

## Возможности версии v0.1.0

- DE с поддержкой трёх стратегий мутации.
- Random-DE с отдельным RNG селектора стратегий.
- RL-DE с табличным Q-learning:
  - состояния: `early`, `middle`, `late` (по доле использованного бюджета),
  - действия: `rand_1`, `best_1`, `current_to_best_1`,
  - награда: доля строгих улучшений в поколении.
- RL-DE v2 (экспериментальный вариант):
  - состояния: `early/middle/late` × `improving/stagnating`,
  - настраиваемый порог стагнации `stagnation_threshold` в YAML.
- Набор тестовых функций: Sphere, Ackley, Rastrigin, Rosenbrock, Schwefel.
- Инфраструктура воспроизводимых сравнительных экспериментов:
  - YAML-конфигурации,
  - одиночный и пакетный запуск,
  - сохранение run-by-run результатов,
  - агрегированная статистика и ранжирование.
- Автоматические тесты (`pytest`).

## Требования

- Python `>=3.11`

## Установка

### Библиотека

```bash
python -m pip install -e .
```

### Библиотека + инструменты разработки/тестирования

```bash
python -m pip install -e .[dev]
```

## Быстрые примеры запуска

### 1) Обычный DE на Sphere

```bash
python examples/sphere_de.py --mode normal --strategy rand/1 --output-dir results_example_rand1
```

Другие стратегии:

```bash
python examples/sphere_de.py --mode normal --strategy best/1 --output-dir results_example_best1
python examples/sphere_de.py --mode normal --strategy current-to-best/1 --output-dir results_example_ctb1
```

### 2) Random-DE на Sphere

```bash
python examples/sphere_de.py --mode normal --strategy random-de --output-dir results_example_random_de
```

### 3) RL-DE (Q-learning) на Sphere

```bash
python examples/rl_de_sphere.py --mode normal --output-dir results_example_rl_de
```

### 4) Учебный режим Q-learning

```bash
python examples/q_learning_demo.py --episodes 60
```

## Сравнительные эксперименты

### Запуск одного сценария

```bash
python experiments/run_comparison.py --config experiments/configs/rl_de_comparison_check.yaml --algorithm de_rand_1 --function sphere --repeats 1 --output-dir results_comparison
```

### Пакетный запуск (все 5 алгоритмов × все 5 функций)

```bash
python experiments/batch_run.py --config experiments/configs/rl_de_comparison_check.yaml --repeats 1 --output-dir results_comparison_batch

# RL-DE v2 (30 запусков × 5 функций, подготовка конфигурации)
python experiments/batch_run.py --config experiments/configs/rl_de_stagnation_v2_10d_30runs.yaml --algorithms rl_de_q_learning_stagnation --output-dir results_rl_de_v2
```

### Анализ результатов пакетного запуска

```bash
python experiments/analyze_batch.py --batch-dir results_comparison_batch/<batch_id>
```

## Тестирование

Полный набор тестов:

```bash
pytest -q
```

## Структура проекта

```text
rl-optimization/
├─ src/rl_optimization/
│  ├─ algorithms/          # DE, Random-DE, RL-DE
│  ├─ agents/              # QLearningAgent
│  ├─ benchmarks/          # Sphere/Ackley/Rastrigin/Rosenbrock/Schwefel
│  ├─ experiments/         # загрузка конфигов, batch-run, статистика
│  └─ monitoring/          # сохранение JSON/CSV/графиков
├─ examples/               # примеры запуска
├─ experiments/
│  ├─ configs/             # YAML-конфигурации экспериментов
│  └─ EXPERIMENT_PROTOCOL.md
└─ tests/                  # pytest
```

## Статус разработки

- Текущая публикационная версия: **v0.1.0**
- Фокус: воспроизводимая исследовательская инфраструктура для DE и первой версии RL-DE.
- Ограничение: в этой версии нет DQN/SARSA/PPO и нет адаптации `F/CR`.

## Лицензирование

Внешняя зависимость `thefittest` распространяется под MIT-лицензией (см. репозиторий проекта).

Лицензия **этого** репозитория должна быть выбрана автором отдельно перед публикацией.
