#!/usr/bin/env python3
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# this module computes physical
# simulation of the leaf motion
# and energy conversion
##################################


import numpy as np
import matplotlib.pyplot as plt


def compute_quadratic_root(a, b, c):
    if abs(a) < 1e-12:
        return -c/b, -c/b
    temp = np.sqrt(b**2 - 4 * a * c)
    return (-b - temp) / (2 * a), (-b + temp) / (2 * a)


class Integrator():
    """integrator"""
    def __init__(self):
        pass


    def FE(self, df, f0, start, end, n=100):
        """df is the dirivative function"""
        dx = (end - start) / n
        xs = np.linspace(start, end, n + 1)
        ys = np.zeros(n + 1)
        ys[0] = f0
        for i, x in enumerate(xs[:-1], 1):
            ys[i] = df(x) * dx + ys[i - 1]
        return xs, ys


class Light():
    """light"""
    def __init__(self, Il0, omega_l, phi_l, kl=1):
        self.Il0 = Il0
        self.omega_l = omega_l
        self.phi_l = phi_l
        self.kl = kl


    def get_Is(self, ts):
        """compute intensities"""
        return 0.5 * self.Il0 * (1 + np.cos(self.omega_l * ts + self.phi_l))


class Heat():
    """Heat"""
    def __init__(self, Ih0, omega_h, phi_h, kh=1):
        self.Ih0 = Ih0
        self.omega_h = omega_h
        self.phi_h = phi_h
        self.kh = kh


    def get_Is(self, ts):
        return 0.5 * self.Ih0 * (1 + np.cos(self.omega_h * ts + self.phi_h))


class Flow:
    """flow class"""
    def __init__(self, vf0, omega, rho, Cd):
        self.vf0 = vf0
        self.omega = omega
        self.rho = rho
        self.Cd = Cd


    def get_period(self):
        """compute period"""
        return 2 * np.pi / self.omega


    def get_velocity_hat(self, t):
        """compute v_hat = v/vf0 at time t"""
        return np.sin(self.omega * t)


    def get_velocities(self, n):
        """divide period into n times and return the velocities """
        period = self.get_period()
        ts = np.linspace(0, period, n)
        return ts, self.vf0 * np.sin(self.omega * ts)


    def get_omega(self):
        """return omega"""
        return self.omega


class Leaf:
    """leaf class"""
    def __init__(self, phi0, R, E1, E2, h, l0, w, Cpz):
        self.phi0 = phi0
        self.phi = phi0
        self.R = R
        self.E1 = E1
        self.E2 = E2
        self.h = h
        self.l0 = l0
        self.w = w
        self.Cpz = Cpz
        self.init_x = self.R * 3
        self.center_x = self.init_x
        self.E_effect = E1 * E2 / (E1 + E2)


    def set_phi(self, phi):
        """change leaf state phi"""
        self.phi = phi
        self.center_x = self.init_x + (self.phi0 - phi) * self.R


    def get_bottom_exposed_surface(self, phi):
        """compute the exposed surface of the bottom layer"""
        if phi < np.pi / 2:
            return 0.0
        elif phi < np.pi * 1.5:
            theta = phi - np.pi / 2
            return self.R * (1 - np.cos(theta)) * self.w
        else:
            return 2 * self.R * self.w


    def get_top_exposed_surface(self, phi):
        """compute the exposed surface area of the top layer"""
        center_x = self.init_x + (self.phi0 - phi) * self.R
        if phi < np.pi * 1.5:
            theta = phi - np.pi / 2
            end = center_x + np.cos(theta) * self.R
        else:
            end = center_x - self.R
        return max(0, end - self.init_x) * self.w


    def compute_light_conversion(self, light, flow, ts, phis):
        """compute wl_hat"""
        wr_star = self.E_effect * self.w * self.h / 2 * flow.vf0
        wl_hats = np.empty_like(phis)
        Is = light.get_Is(ts) * light.kl
        for i in range(len(wl_hats)):
            phi = phis[i]
            area = self.get_bottom_exposed_surface(phi)
            wl_hats[i] = Is[i] * area
        wl_hats /= wr_star
        return wl_hats


    def compute_heat_conversion(self, heat, flow, ts, phis):
        """compute wh_hat"""
        wr_star = self.E_effect * self.w * self.h / 2 * flow.vf0
        wh_hats = np.empty_like(phis)
        Is = heat.get_Is(ts) * heat.kh
        for i in range(len(wh_hats)):
            phi = phis[i]
            area = self.get_top_exposed_surface(phi)
            wh_hats[i] = Is[i] * area
        wh_hats /= wr_star
        return wh_hats


    def flow_load(self, flow, Apz = 0.01, Br2 = 20, n = 100):
        """compute leaf response under flow in one period"""
        # Apz = self.Cpz / (flow.Cd * flow.rho * flow.vf0)
        # Br2 = 0.5 * self.E_effect / (self.Cd * flow.rho * (flow.vf0 ** 2))
        # flow_period = np.pi * 2 / flow.omega
        flow_period = flow.get_period()
        ts = np.linspace(0, flow_period, n)
        dt = ts[1] - ts[0]
        vcs = np.zeros(n)
        phis = np.zeros(n)
        ws = np.zeros(n)
        h_over_R = self.h / self.R
        wr_star = self.E_effect * self.w * self.h / 2 * flow.vf0
        for i, t in enumerate(ts):
            theta = np.pi if self.phi > np.pi else self.phi
            vf_hat = flow.get_velocity_hat(t)
            '''
            if vf_hat**2 <= Br2 * (h_over_R ** 3):
                theta_f = theta
            else:
                cos_theta_f = 1 - 2 * Br2 * (h_over_R**3) / (vf_hat ** 2)
                theta_f = np.arccos(cos_theta_f)
            if theta < theta_f:
                theta = theta_f
                vc = 0
                vc_hat = 0
            else:
                alpha = np.sqrt(0.5 * (1 - np.cos(theta)))
                vf_hat_sign = 1 if vf_hat >= 0 else -1
                # a = alpha ** 2 if vf_hat >= 0 else -alpha ** 2
                a = alpha ** 2
                b = Apz * h_over_R
                # Br2_temp = Br2 if vf_hat >= 0 else -Br2
                Br2_temp = Br2
                c = -Br2_temp * (h_over_R ** 3) - Apz * h_over_R * abs(vf_hat)
                if (b**2 - 4 * a * c < 0):
                    print(theta, a, b, c)
                r1, r2 = compute_quadratic_root(a, b, c)
                # assert((vf_hat - r1) * vf_hat >= 0 or (vf_hat - r2) * vf_hat >= 0)
                if (vf_hat - r1) * vf_hat >= 0:
                    vc_hat = vf_hat - r1
                else:
                    vc_hat = vf_hat - r2
            '''
            alpha = np.sqrt(0.5 * (1 - np.cos(theta)))
            # compute vc with solving quadratic equations
            a = alpha ** 2
            b = Apz * h_over_R
            c = -Br2 * (h_over_R ** 3) - Apz * (h_over_R) * vf_hat
            if vf_hat < 0:
                a = alpha ** 2
                b = -Apz * h_over_R
                c = Br2 * (h_over_R ** 3) + Apz * (h_over_R) * vf_hat
                if b * b - 4 * a * c < 0:
                    a = -a
            r1, r2 = compute_quadratic_root(a, b, c)
            if vf_hat >= 0:
                Br2_temp = Br2
            else:
                Br2_temp = -Br2
            vc_hat = vf_hat - r2 # not assuming Apz ~ 1
            # vc_hat = vf_hat - Br2_temp/alpha * (h_over_R)**(1.5)
            if (vf_hat >= 0 and vc_hat < 0 and i < 0.25 * len(ts)):
                vc_hat = 0
            if vf_hat < 0 and vc_hat > 0:
                vc_hat = vf_hat - r1
            # print("r1 = {:f}, r2 = {:f}, vc_hat = {:f}; vc1 = {:f}, vc2 = {:f}, vf_hat = {:f}".format(r1, r2, vc_hat, vf_hat - r1, vf_hat - r2, vf_hat))
            vc = vc_hat * flow.vf0
            # vc = abs(vc_hat) * vf_hat_sign * flow.vf0
            vcs[i] = vc
            # update self.phi
            new_phi = self.phi - vc * dt / self.R
            new_phi = min(new_phi, np.pi * 2)
            # new_phi = max(new_phi, theta_f)
            if vc < 0 and new_phi >= np.pi * 2:
                vcs[i] = 0
                vc_hat = 0
            if vf_hat < 0 and i > 0.75 * len(ts) and vc_hat > 0:
                vcs[i] = 0
                vc_hat = 0
            self.set_phi(new_phi)
            phis[i] = new_phi
            w = Apz / Br2 * (vc_hat ** 2) # dimension = 1
            # w *= wr_star
            ws[i] = w
        return ts, vcs, phis, ws


    def get_axes_limits(self):
        """compute x and y axis limits """
        xmax = np.ceil(self.init_x + 2 * np.pi * self.R)
        ymax = 2.5 * self.R
        return xmax, ymax


    def get_shape_coords(self):
        """get leaf coords for plotting """
        cx, cy = self.center_x, self.R
        ts = np.linspace(-np.pi / 2, -np.pi / 2 + self.phi)
        xs = cx + np.cos(ts) * self.R
        ys = cy + np.sin(ts) * self.R
        xs = np.hstack((np.array([0]), xs))
        ys = np.hstack((np.array([0]), ys))
        return xs, ys


    def plot(self):
        """plot leaf state with current phi"""
        xmax, ymax = self.get_axes_limits()
        plt.xlim([0, xmax])
        plt.ylim([0, ymax])
        plt.xticks([])
        plt.yticks([])
        # plt.plot([0, self.center_x], [0, 0], 'g-', linewidth=4.0)
        xs, ys = self.get_shape_coords()
        plt.plot(xs, ys, 'g-', linewidth=4.0)


def animation(leaf, angles_lst, dt=0.01):
    for angles in angles_lst:
        for i, phi in enumerate(angles):
            plt.clf()
            leaf.set_phi(phi)
            leaf.plot()
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.draw()
            plt.pause(dt)
    plt.show()


def animation_test():
    phi = np.pi * 1
    R = 10
    E1, E2 = 5, 5
    h = 1
    l0 = 20
    w = 1
    Cpz = 1
    leaf = Leaf(phi, R, E1, E2, h, l0, w, Cpz)
    n = 50
    ncircles = 1
    fig = plt.figure(figsize=(12, 6))
    start_angle = np.pi / 2
    end_angle = np.pi * 1
    angles_lst = []
    for _ in range(ncircles):
        angles_lst.append(np.linspace(start_angle, end_angle, n))
        angles_lst.append(np.linspace(end_angle, start_angle, n))
    animation(leaf, angles_lst, dt=0.01)


def df(x):
    return x


def test_int():
    y0 = 0.125
    start = 0.5
    end = 100
    n = 1000
    intergor = Integrator()
    xs, ys = intergor.FE(df, y0, start, end, n)
    # print(ys)
    plt.plot(xs, ys)
    plt.show()


def test_flow():
    phi = np.pi * 2
    R = 10
    E1, E2 = 5, 5
    h = 1
    l0 = 20
    w = 1
    Cpz = 1
    leaf = Leaf(phi, R, E1, E2, h, l0, w, Cpz)
    flow = Flow(1, np.pi/100, 1000, 1)
    ts, vs, phis, ws = leaf.flow_load(flow)
    Il0 = 1
    omega_l = np.pi / 100
    phi_l = 0
    kl = 0.8
    light = Light(Il0, omega_l, phi_l, kl)
    Ih0 = 2
    omega_h = np.pi / 100
    kh = 0.5
    phi_h = 0
    heat = Heat(Ih0, omega_h, phi_h, kh)
    # def compute_light_conversion(self, light, flow, ts, phis):
    light_ws = leaf.compute_light_conversion(light, flow, ts, phis)
    heat_ws = leaf.compute_heat_conversion(heat, flow, ts, phis)
    # plt.plot(ts, light_ws)
    light_surfaces = np.empty_like(ts)
    heat_surfaces = np.empty_like(ts)
    for i in range(len(light_surfaces)):
        light_surfaces[i] = leaf.get_bottom_exposed_surface(phis[i])
    plt.plot(ts, light_surfaces)
    plt.show()

    # print(ws)
    # plt.plot(ts, vs)
    # plt.plot(ts, phis)
    # plt.show()
    # print(phis)
    # animation(leaf, [phis])


def test_light_and_heat():
    n = 100
    Il0 = 1
    omega_l = np.pi / 10
    phi_l = 0
    light = Light(Il0, omega_l, phi_l)
    heat = Heat(Il0, omega_l, phi_l)
    T = 2 * np.pi / omega_l
    ts = np.linspace(0, T, num=100)
    Is = light.get_Is(ts) * light.kl
    # Is = heat.get_Is(ts) * heat.kh
    plt.plot(ts, Is)
    plt.show()


if __name__ == "__main__":
    # animation_test()
    # test_int()
    # test_flow()
    test_light_and_heat()
