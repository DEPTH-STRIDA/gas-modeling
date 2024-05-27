import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def run_simulation(X, Y, Z, T, F, rho, dt, nu, U, Re, dimensionality, 
                   obstacle_shape, obstacle_center, obstacle_radius, 
                   obstacle_start, obstacle_end, vector_range_start, 
                   vector_range_end, show_streamlines, scheme, time_step):
    xmin, xmax = 0, 2
    ymin, ymax = 0, 2
    zmin, zmax = 0, 2

    dx = (xmax - xmin) / (X - 1)
    dy = (ymax - ymin) / (Y - 1)
    dz = (zmax - zmin) / (Z - 1)

    # Initial conditions
    if dimensionality == "2D":
        p, b = np.zeros((Y, X)), np.zeros((Y, X))
        u, v = np.ones((Y, X)) * U, np.zeros((Y, X))
        w = None
        x = np.linspace(0, xmax, X)
        y = np.linspace(0, ymax, Y)
        nX, nY = np.meshgrid(x, y)
    else:
        p, b = np.zeros((Z, Y, X)), np.zeros((Z, Y, X))
        u, v, w = np.ones((Z, Y, X)) * U, np.zeros((Z, Y, X)), np.zeros((Z, Y, X))
        x = np.linspace(0, xmax, X)
        y = np.linspace(0, ymax, Y)
        z = np.linspace(0, zmax, Z)
        nX, nY, nZ = np.meshgrid(x, y, z, indexing='ij')

    u_left = U
    u_right = U
    u_top = 0
    u_bottom = 0

    v_left = 0
    v_right = 0
    v_top = 0
    v_bottom = 0

    w_front = 0
    w_back = 0
    p_right = 0

    if dimensionality == "2D":
        fig, ax = plt.subplots()
        quiver_scale = X + 29
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    def pressure_poisson_2D(p, dx, dy, b):
        pn = np.empty_like(p)
        for _ in range(50):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])

            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]
            p[-1, :] = p[-2, :]

        return p

    def explicit_step_2D(u, v, p, dx, dy, dt, nu):
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])) + F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        return u, v

    def implicit_step_2D(u, v, p, dx, dy, dt, nu):
        un = u.copy()
        vn = v.copy()
        b = np.zeros_like(p)

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_2D(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        return u, v, p

    def laplacian_filter_2D(arr):
        arr_filtered = arr.copy()
        arr_filtered[1:-1, 1:-1] = (arr[:-2, 1:-1] + arr[2:, 1:-1] +
                                    arr[1:-1, :-2] + arr[1:-1, 2:] -
                                    4 * arr[1:-1, 1:-1])
        return arr_filtered

    def apply_obstacle_2D(u, v):
        if obstacle_shape == "rectangle":
            u[obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1] = 0
            v[obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1] = 0

        elif obstacle_shape == "circle":
            for i in range(Y):
                for j in range(X):
                    if (i - obstacle_center[0]) ** 2 + (j - obstacle_center[1]) ** 2 <= obstacle_radius ** 2:
                        u[i, j] = 0
                        v[i, j] = 0
        return u, v

    def animate_2D(n):
        nonlocal u, v, p, b

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_2D(p, dx, dy, b)

        if scheme == "explicit":
            u, v = explicit_step_2D(u, v, p, dx, dy, dt, nu)
        elif scheme == "implicit":
            u, v, p = implicit_step_2D(u, v, p, dx, dy, dt, nu)

        u = u + laplacian_filter_2D(u) * dt * nu
        v = v + laplacian_filter_2D(v) * dt * nu

        # Apply obstacle
        u, v = apply_obstacle_2D(u, v)

        # Update boundary conditions
        u[:, 0] = u_left
        if u_right is not None:
            u[:, -1] = u_right
        else:
            u[:, -1] = u[:, -2]
        u[0, :] = u_top
        u[-1, :] = u_bottom

        v[:, 0] = v_left
        v[:, -1] = v_right
        v[0, :] = v_top
        v[-1, :] = v_bottom

        mask = np.ones_like(u, dtype=bool)
        ax.clear()

        if show_streamlines:
            ax.streamplot(nX, nY, u, v, color='black')
        else:
            ax.quiver(nX[mask], nY[mask], u[mask], v[mask], scale=50, width=0.002, headlength=4, headwidth=3, headaxislength=4, color='black')

        if obstacle_shape == "rectangle":
            rect = plt.Rectangle((obstacle_start * dx, obstacle_start * dy),
                                 (obstacle_end - obstacle_start) * dx,
                                 (obstacle_end - obstacle_start) * dy,
                                 linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        elif obstacle_shape == "circle":
            circle = plt.Circle((obstacle_center[1] * dx, obstacle_center[0] * dy),
                                obstacle_radius * dx,
                                linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(circle)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(f'Time step: {n}')
        return ax

    def pressure_poisson_3D(p, dx, dy, dz, b):
        pn = np.empty_like(p)
        for _ in range(50):
            pn = p.copy()
            p[1:-1, 1:-1, 1:-1] = (((pn[1:-1, 1:-1, 2:] + pn[1:-1, 1:-1, :-2]) * dy ** 2 * dz ** 2 +
                                    (pn[1:-1, 2:, 1:-1] + pn[1:-1, :-2, 1:-1]) * dx ** 2 * dz ** 2 +
                                    (pn[2:, 1:-1, 1:-1] + pn[:-2, 1:-1, 1:-1]) * dx ** 2 * dy ** 2) /
                                   (2 * (dx ** 2 * dy ** 2 + dy ** 2 * dz ** 2 + dz ** 2 * dx ** 2)) -
                                   dx ** 2 * dy ** 2 * dz ** 2 / (2 * (dx ** 2 * dy ** 2 + dy ** 2 * dz ** 2 + dz ** 2 * dx ** 2)) *
                                   b[1:-1, 1:-1, 1:-1])

            p[:, :, -1] = p[:, :, -2]
            p[:, -1, :] = p[:, -2, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, 0, :] = p[:, 1, :]
            p[0, :, :] = p[1, :, :]

        return p

    def explicit_step_3D(u, v, w, p, dx, dy, dz, dt, nu):
        un = u.copy()
        vn = v.copy()
        wn = w.copy()

        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (un[1:-1, 1:-1, 1:-1] - un[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) +
                               nu * (dt / dx ** 2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[:-2, 1:-1, 1:-1])) + F * dt)

        v[1:-1, 1:-1, 1:-1] = (vn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (vn[1:-1, 1:-1, 1:-1] - vn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) +
                               nu * (dt / dx ** 2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[:-2, 1:-1, 1:-1])))

        w[1:-1, 1:-1, 1:-1] = (wn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (wn[1:-1, 1:-1, 1:-1] - wn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dz) * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) +
                               nu * (dt / dx ** 2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[:-2, 1:-1, 1:-1])))

        return u, v, w

    def implicit_step_3D(u, v, w, p, dx, dy, dz, dt, nu):
        un = u.copy()
        vn = v.copy()
        wn = w.copy()
        b = np.zeros_like(p)

        b[1:-1, 1:-1, 1:-1] = (rho * (1 / dt *
                                      ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx) +
                                       (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
                                       (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) -
                                      ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx)) ** 2 -
                                      2 * ((u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy) *
                                           (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dx)) -
                                      ((v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)) ** 2 -
                                      ((w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_3D(p, dx, dy, dz, b)

        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (un[1:-1, 1:-1, 1:-1] - un[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) +
                               nu * (dt / dx ** 2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[:-2, 1:-1, 1:-1])) + F * dt)

        v[1:-1, 1:-1, 1:-1] = (vn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (vn[1:-1, 1:-1, 1:-1] - vn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) +
                               nu * (dt / dx ** 2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[:-2, 1:-1, 1:-1])))

        w[1:-1, 1:-1, 1:-1] = (wn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (wn[1:-1, 1:-1, 1:-1] - wn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dz) * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) +
                               nu * (dt / dx ** 2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[:-2, 1:-1, 1:-1])))

        return u, v, w, p

    def laplacian_filter_3D(arr):
        arr_filtered = arr.copy()
        arr_filtered[1:-1, 1:-1, 1:-1] = (arr[:-2, 1:-1, 1:-1] + arr[2:, 1:-1, 1:-1] +
                                          arr[1:-1, :-2, 1:-1] + arr[1:-1, 2:, 1:-1] +
                                          arr[1:-1, 1:-1, :-2] + arr[1:-1, 1:-1, 2:] -
                                          6 * arr[1:-1, 1:-1, 1:-1])
        return arr_filtered

    def apply_obstacle_3D(u, v, w):
        if obstacle_shape == "rectangle":
            u[obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1] = 0
            v[obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1] = 0
            w[obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1, obstacle_start:obstacle_end+1] = 0

        elif obstacle_shape == "circle":
            for i in range(Z):
                for j in range(Y):
                    for k in range(X):
                        if (i - obstacle_center[0]) ** 2 + (j - obstacle_center[1]) ** 2 + (
                                k - obstacle_center[2]) ** 2 <= obstacle_radius ** 2:
                            u[i, j, k] = 0
                            v[i, j, k] = 0
                            w[i, j, k] = 0
        return u, v, w

    def simulate_3D(num_steps):
        nonlocal u, v, w, p, b

        for _ in range(num_steps):
            b[1:-1, 1:-1, 1:-1] = (rho * (1 / dt *
                                          ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx) +
                                           (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
                                           (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) -
                                          ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx)) ** 2 -
                                          2 * ((u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy) *
                                               (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dx)) -
                                          ((v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)) ** 2 -
                                          ((w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) ** 2))

            b = np.nan_to_num(b)
            p = pressure_poisson_3D(p, dx, dy, dz, b)

            if scheme == "explicit":
                u, v, w = explicit_step_3D(u, v, w, p, dx, dy, dz, dt, nu)
            elif scheme == "implicit":
                u, v, w, p = implicit_step_3D(u, v, w, p, dx, dy, dz, dt, nu)

            u = u + laplacian_filter_3D(u) * dt * nu
            v = v + laplacian_filter_3D(v) * dt * nu
            w = w + laplacian_filter_3D(w) * dt * nu

            # Apply obstacle
            u, v, w = apply_obstacle_3D(u, v, w)

            # Update boundary conditions
            u[:, :, 0] = u_left
            if u_right is not None:
                u[:, :, -1] = u_right
            else:
                u[:, :, -1] = u[:, :, -2]
            u[:, 0, :] = u_top
            u[:, -1, :] = u_bottom
            u[0, :, :] = 0
            u[-1, :, :] = 0

            v[:, :, 0] = v_left
            v[:, :, -1] = v_right
            v[:, 0, :] = v_top
            v[:, -1, :] = v_bottom
            v[0, :, :] = 0
            v[-1, :, :] = 0

            w[:, :, 0] = w_front
            w[:, :, -1] = w_back
            w[:, 0, :] = 0
            w[:, -1, :] = 0
            w[0, :, :] = 0
            w[-1, :, :] = 0

    def plot_3D(time_step):
        nonlocal u, v, w, p

        simulate_3D(time_step)

        mask = np.zeros_like(u, dtype=bool)

        # Формируем маску для препятствия
        for i in range(1, Z - 1):
            for j in range(1, Y - 1):
                for k in range(1, X - 1):
                    if obstacle_shape == "rectangle":
                        if (obstacle_start - vector_range_start <= i <= obstacle_end + vector_range_end and
                            obstacle_start - vector_range_start <= j <= obstacle_end + vector_range_end and
                            obstacle_start - vector_range_start <= k <= obstacle_end + vector_range_end):
                            mask[i, j, k] = True
                    elif obstacle_shape == "circle":
                        if obstacle_radius - vector_range_start <= np.sqrt(
                                (i - obstacle_center[0]) ** 2 + (j - obstacle_center[1]) ** 2 + (k - obstacle_center[2]) ** 2
                        ) <= obstacle_radius + vector_range_end:
                            mask[i, j, k] = True

        ax.clear()
        ax.quiver(nX[mask], nY[mask], nZ[mask], u[mask], v[mask], w[mask], length=0.1, normalize=True)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        ax.set_title(f'Snapshot at time step: {time_step}')

        if obstacle_shape == "rectangle":
            x = np.linspace(obstacle_start * dx, obstacle_end * dx, 2)
            y = np.linspace(obstacle_start * dy, obstacle_end * dy, 2)
            z = np.linspace(obstacle_start * dz, obstacle_end * dz, 2)
            xx, yy = np.meshgrid(x, y)

            for z_coord in z:
                ax.plot_surface(xx, yy, z_coord * np.ones_like(xx), color='red', alpha=1.0)

            for y_coord in y:
                xx, zz = np.meshgrid(x, z)
                ax.plot_surface(xx, y_coord * np.ones_like(xx), zz, color='red', alpha=1.0)

            for x_coord in x:
                yy, zz = np.meshgrid(y, z)
                ax.plot_surface(x_coord * np.ones_like(yy), yy, zz, color='red', alpha=1.0)

        elif obstacle_shape == "circle":
            u_sphere, v_sphere = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x_sphere = obstacle_radius * np.cos(u_sphere) * np.sin(v_sphere) * dx + obstacle_center[0] * dx
            y_sphere = obstacle_radius * np.sin(u_sphere) * np.sin(v_sphere) * dy + obstacle_center[1] * dy
            z_sphere = obstacle_radius * np.cos(v_sphere) * dz + obstacle_center[2] * dz
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=1.0)

        plt.show()

    if dimensionality == "2D":
        ani = animation.FuncAnimation(fig, animate_2D, frames=T, interval=10)
        plt.show()
    elif dimensionality == "3D":
        plot_3D(time_step)
