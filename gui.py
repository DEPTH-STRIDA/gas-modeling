import tkinter as tk
from tkinter import ttk, messagebox
from logic import run_simulation
X, Y, Z = 30, 30, 30
T = 1000
F = 1
rho = 1
dt = 0.0001
nu = 0.1
U = 1
Re = 10
dimensionality = "2D"
obstacle_shape = "rectangle"
obstacle_center = (15, 15, 15)
obstacle_radius = 3
obstacle_start = 8
obstacle_end = 19
vector_range_start = 1
vector_range_end = 2
show_streamlines = False
scheme = "implicit"
time_step = 200


def start():
    try:
        X = int(entry_X.get())
        Y = int(entry_Y.get())
        Z = int(entry_Z.get())
        T = int(entry_T.get())
        F = float(entry_F.get())
        rho = float(entry_rho.get())
        dt = float(entry_dt.get())
        nu = float(entry_nu.get())
        U = float(entry_U.get())
        Re = float(entry_Re.get())
        dimensionality = dim_var.get()
        obstacle_shape = obstacle_var.get()
        obstacle_center = (int(entry_obstacle_center_x.get()), int(entry_obstacle_center_y.get()), int(entry_obstacle_center_z.get()))
        obstacle_radius = int(entry_obstacle_radius.get())
        obstacle_start = int(entry_obstacle_start.get())
        obstacle_end = int(entry_obstacle_end.get())
        vector_range_start = int(entry_vector_range_start.get())
        vector_range_end = int(entry_vector_range_end.get())
        show_streamlines = bool(show_streamlines_var.get())
        scheme = scheme_var.get()
        time_step = int(entry_time_step.get())

        run_simulation(X, Y, Z, T, F, rho, dt, nu, U, Re, dimensionality, 
                       obstacle_shape, obstacle_center, obstacle_radius, 
                       obstacle_start, obstacle_end, vector_range_start, 
                       vector_range_end, show_streamlines, scheme, time_step)

    except ValueError:
        messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные значения параметров.")


root = tk.Tk()
root.title("Параметры симуляции")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="Количество узлов по X:").grid(column=1, row=1, sticky=tk.W)
entry_X = ttk.Entry(mainframe, width=7)
entry_X.grid(column=2, row=1, sticky=(tk.W, tk.E))
entry_X.insert(0, str(X))

ttk.Label(mainframe, text="Количество узлов по Y:").grid(column=1, row=2, sticky=tk.W)
entry_Y = ttk.Entry(mainframe, width=7)
entry_Y.grid(column=2, row=2, sticky=(tk.W, tk.E))
entry_Y.insert(0, str(Y))

ttk.Label(mainframe, text="Количество узлов по Z:").grid(column=1, row=3, sticky=tk.W)
entry_Z = ttk.Entry(mainframe, width=7)
entry_Z.grid(column=2, row=3, sticky=(tk.W, tk.E))
entry_Z.insert(0, str(Z))

ttk.Label(mainframe, text="Количество шагов по времени:").grid(column=1, row=4, sticky=tk.W)
entry_T = ttk.Entry(mainframe, width=7)
entry_T.grid(column=2, row=4, sticky=(tk.W, tk.E))
entry_T.insert(0, str(T))

ttk.Label(mainframe, text="Сила (F):").grid(column=1, row=5, sticky=tk.W)
entry_F = ttk.Entry(mainframe, width=7)
entry_F.grid(column=2, row=5, sticky=(tk.W, tk.E))
entry_F.insert(0, str(F))

ttk.Label(mainframe, text="Плотность (ρ):").grid(column=1, row=6, sticky=tk.W)
entry_rho = ttk.Entry(mainframe, width=7)
entry_rho.grid(column=2, row=6, sticky=(tk.W, tk.E))
entry_rho.insert(0, str(rho))

ttk.Label(mainframe, text="Шаг по времени (dt):").grid(column=1, row=7, sticky=tk.W)
entry_dt = ttk.Entry(mainframe, width=7)
entry_dt.grid(column=2, row=7, sticky=(tk.W, tk.E))
entry_dt.insert(0, str(dt))

ttk.Label(mainframe, text="Кинематическая вязкость (ν):").grid(column=1, row=8, sticky=tk.W)
entry_nu = ttk.Entry(mainframe, width=7)
entry_nu.grid(column=2, row=8, sticky=(tk.W, tk.E))
entry_nu.insert(0, str(nu))

ttk.Label(mainframe, text="Начальная скорость (U):").grid(column=1, row=9, sticky=tk.W)
entry_U = ttk.Entry(mainframe, width=7)
entry_U.grid(column=2, row=9, sticky=(tk.W, tk.E))
entry_U.insert(0, str(U))

ttk.Label(mainframe, text="Число Рейнольдса (Re):").grid(column=1, row=10, sticky=tk.W)
entry_Re = ttk.Entry(mainframe, width=7)
entry_Re.grid(column=2, row=10, sticky=(tk.W, tk.E))
entry_Re.insert(0, str(Re))

dim_var = tk.StringVar()
dim_var.set(dimensionality)
ttk.Label(mainframe, text="Размерность:").grid(column=1, row=11, sticky=tk.W)
ttk.Radiobutton(mainframe, text="2D", variable=dim_var, value="2D").grid(column=2, row=11, sticky=tk.W)
ttk.Radiobutton(mainframe, text="3D", variable=dim_var, value="3D").grid(column=3, row=11, sticky=tk.W)

obstacle_var = tk.StringVar()
obstacle_var.set(obstacle_shape)
ttk.Label(mainframe, text="Форма препятствия:").grid(column=1, row=12, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Прямоугольник", variable=obstacle_var, value="rectangle").grid(column=2, row=12, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Круг", variable=obstacle_var, value="circle").grid(column=3, row=12, sticky=tk.W)

ttk.Label(mainframe, text="Центр препятствия X:").grid(column=1, row=13, sticky=tk.W)
entry_obstacle_center_x = ttk.Entry(mainframe, width=7)
entry_obstacle_center_x.grid(column=2, row=13, sticky=(tk.W, tk.E))
entry_obstacle_center_x.insert(0, str(obstacle_center[0]))

ttk.Label(mainframe, text="Центр препятствия Y:").grid(column=1, row=14, sticky=tk.W)
entry_obstacle_center_y = ttk.Entry(mainframe, width=7)
entry_obstacle_center_y.grid(column=2, row=14, sticky=(tk.W, tk.E))
entry_obstacle_center_y.insert(0, str(obstacle_center[1]))

ttk.Label(mainframe, text="Центр препятствия Z:").grid(column=1, row=15, sticky=tk.W)
entry_obstacle_center_z = ttk.Entry(mainframe, width=7)
entry_obstacle_center_z.grid(column=2, row=15, sticky=(tk.W, tk.E))
entry_obstacle_center_z.insert(0, str(obstacle_center[2]))

ttk.Label(mainframe, text="Радиус препятствия:").grid(column=1, row=16, sticky=tk.W)
entry_obstacle_radius = ttk.Entry(mainframe, width=7)
entry_obstacle_radius.grid(column=2, row=16, sticky=(tk.W, tk.E))
entry_obstacle_radius.insert(0, str(obstacle_radius))

ttk.Label(mainframe, text="Начало препятствия:").grid(column=1, row=17, sticky=tk.W)
entry_obstacle_start = ttk.Entry(mainframe, width=7)
entry_obstacle_start.grid(column=2, row=17, sticky=(tk.W, tk.E))
entry_obstacle_start.insert(0, str(obstacle_start))

ttk.Label(mainframe, text="Конец препятствия:").grid(column=1, row=18, sticky=tk.W)
entry_obstacle_end = ttk.Entry(mainframe, width=7)
entry_obstacle_end.grid(column=2, row=18, sticky=(tk.W, tk.E))
entry_obstacle_end.insert(0, str(obstacle_end))

ttk.Label(mainframe, text="Начало диапазона векторов:").grid(column=1, row=19, sticky=tk.W)
entry_vector_range_start = ttk.Entry(mainframe, width=7)
entry_vector_range_start.grid(column=2, row=19, sticky=(tk.W, tk.E))
entry_vector_range_start.insert(0, str(vector_range_start))

ttk.Label(mainframe, text="Конец диапазона векторов:").grid(column=1, row=20, sticky=tk.W)
entry_vector_range_end = ttk.Entry(mainframe, width=7)
entry_vector_range_end.grid(column=2, row=20, sticky=(tk.W, tk.E))
entry_vector_range_end.insert(0, str(vector_range_end))

show_streamlines_var = tk.IntVar()
show_streamlines_var.set(int(show_streamlines))
ttk.Checkbutton(mainframe, text="Показать линии тока", variable=show_streamlines_var).grid(column=1, row=21, sticky=tk.W)

scheme_var = tk.StringVar()
scheme_var.set(scheme)
ttk.Label(mainframe, text="Схема:").grid(column=1, row=22, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Явная", variable=scheme_var, value="explicit").grid(column=2, row=22, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Неявная", variable=scheme_var, value="implicit").grid(column=3, row=22, sticky=tk.W)

ttk.Label(mainframe, text="Шаг времени:").grid(column=1, row=23, sticky=tk.W)
entry_time_step = ttk.Entry(mainframe, width=7)
entry_time_step.grid(column=2, row=23, sticky=(tk.W, tk.E))
entry_time_step.insert(0, str(time_step))

ttk.Button(mainframe, text="Начать симуляцию", command=start).grid(column=2, row=24, sticky=tk.W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)
def runGui():
    root.mainloop()

runGui()
