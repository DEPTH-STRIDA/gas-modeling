# Моделирование турбулентного потока вокруг препятствий
# Turbulent Flow Simulation around Obstacles

## Описание | Description

[RU]
Проект представляет собой программу для моделирования турбулентного потока газа вокруг различных препятствий (прямоугольных и круглых) с использованием уравнений Навье-Стокса. Программа включает графический интерфейс пользователя, разработанный с помощью Tkinter, который позволяет настраивать параметры симуляции и визуализировать результаты в 2D и 3D форматах.

[EN]
This project simulates turbulent gas flow around various obstacles (rectangular and circular) using the Navier-Stokes equations. The program features a Tkinter-based graphical user interface for configuring simulation parameters and visualizing results in both 2D and 3D formats.

## Пример/ Example
![2025-01-23_05-03-03](https://github.com/user-attachments/assets/94c40b0a-19c1-4d21-a86e-d7368b41bd7f)
![2025-01-23_05-03-30](https://github.com/user-attachments/assets/f73d3878-f5ff-45e0-810e-a36f22e846c6)

## Основные возможности | Features

- Численное решение уравнений Навье-Стокса
- Явная и неявная численные схемы
- 2D и 3D визуализация результатов моделирования
- Настройка параметров сетки и начальных условий
- Выбор формы препятствия (прямоугольник/круг)
- Настройка параметров потока (вязкость, плотность, скорость)
- Визуализация линий тока (опционально)

## Установка | Installation

### Требования | Prerequisites

- Python 3.8 или выше
- Необходимые библиотеки:
  - numpy
  - matplotlib
  - tkinter (обычно входит в стандартную библиотеку Python)

### Установка зависимостей | Install Dependencies

```bash
pip install -r requirements.txt
```

### Запуск | Running

```bash
python gui.py
```

## Параметры симуляции | Simulation Parameters

- Размер сетки (X, Y, Z)
- Количество временных шагов
- Физические параметры:
  - Плотность (ρ)
  - Кинематическая вязкость (ν)
  - Начальная скорость (U)
  - Число Рейнольдса (Re)
- Параметры препятствия:
  - Форма (прямоугольник/круг)
  - Положение
  - Размеры
- Схема расчёта (явная/неявная)

## Примечания | Notes

Проект находится в активной разработке.

## Features

- Numerical solution of Navier-Stokes equations
- Explicit and implicit numerical schemes
- 2D and 3D visualization of simulation results
- User-friendly GUI for setting simulation parameters
- Adjustable mesh size and initial conditions

## Installation

To get started with this project, follow the steps below:

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or later
- Git

### Clone the Repository

Clone the repository from GitHub:

```sh
git clone https://github.com/username/project-repo.git
cd project-repo


