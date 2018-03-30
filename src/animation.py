#!/usr/bin/env python3
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# this module creates animation
##################################


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from energy_harvest import Flow, Leaf, Light, Heat

# plt.rcParams['font.size'] = 12
# plt.rcParams['font.weight'] = 'bold'


def compute_axis_limits(vmin, vmax, margin=0.1):
    """ compute axes limits considering margin """
    vmargin = (vmax - vmin) * margin
    return vmin - vmargin, vmax + vmargin


class SubplotAnimation(animation.TimedAnimation):
    """ create animation of leaf motion under flow 
        and energy convertion
    """
    def __init__(self, leaf, flow, light, heat, Apz = 0.01, Br2 = 5, n_steps=400, ani_interval=20):
        self.xlabel = 'time'
        self.supsize = 20        # font size of super title
        self.fontweight = 'bold' # font weight of both super and sub title and label
        self.subsize = 16        # font size of sub title and label
        self.ticksize = 12
        self.leaf = leaf
        self.flow = flow
        self.light = light
        self.heat = heat
        self.Apz = Apz
        self.Br2 = Br2
        self.n_steps = n_steps
        fig = plt.figure(figsize=(24, 18))
        fig.tight_layout(h_pad=1.5)
        plt.suptitle(r'Energy Harvest (Apz = {}, $Br^2$ = {}, $\omega$ = {}$\pi$)'.format(Apz, Br2, flow.get_omega()/np.pi), \
                fontsize=self.supsize, weight=self.fontweight)
        ax1 = fig.add_subplot(3, 3, 1) # flow velocity
        ax2 = fig.add_subplot(3, 3, 2) # leaf velocity
        ax3 = fig.add_subplot(3, 1, 3) # leaf shape
        ax5 = fig.add_subplot(3, 3, 4) # heat and light Intensity
        ax6 = fig.add_subplot(3, 3, 5) # heat and light absorption power
        ax7 = fig.add_subplot(3, 3, 6) # light and heat energy harvest
        ax4 = fig.add_subplot(3, 3, 3) # pz energy harvested

        self.ax1 = ax1

        self.flow_period = flow.get_period()

        self.ts = np.linspace(0, self.flow_period, n_steps) # ts
        self.xticks = np.linspace(0, self.flow_period, 5) # x ticks
        _, self.vfs = flow.get_velocities(n_steps)     # vfs
        ts, vcs, phis, pz_ws = leaf.flow_load(flow, Apz, Br2, n_steps)
        self.vcs = vcs
        self.phis = phis
        self.pz_ws = pz_ws
        self.light_ws = leaf.compute_light_conversion(light, flow, ts, phis)
        self.heat_ws = leaf.compute_heat_conversion(heat, flow, ts, phis)
        self.pz_es = np.zeros(n_steps)
        self.light_es = np.zeros(n_steps)
        self.heat_es = np.zeros(n_steps)
        self.dt = self.ts[1] - self.ts[0]
        for i in range(1, n_steps):
            self.pz_es[i] = self.pz_es[i - 1] + self.pz_ws[i - 1] * self.dt
            self.light_es[i] = self.light_es[i - 1] + self.light_ws[i - 1] * self.dt
            self.heat_es[i] = self.heat_es[i - 1] + self.heat_ws[i - 1] * self.dt
        self.Ihs = heat.get_Is(self.ts)
        self.Ils = light.get_Is(self.ts)
        self.print_info()


        ## 1. axis 1 flow velocity
        ax1.set_title('Flow Velocity', fontsize=self.subsize, weight=self.fontweight)
        ax1.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax1.set_ylabel('flow velocity', fontsize=self.subsize, weight=self.fontweight)
        self.line1 = Line2D([], [], color='grey') # background
        self.line1a = Line2D([], [], color='brown', linewidth=2)
        self.line1e = Line2D(
                [], [], color='brown', marker='o', markeredgecolor='brown')
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)

        ax1.set_xlim(0, self.flow_period)
        ymin, ymax = compute_axis_limits(-flow.vf0, flow.vf0)
        ax1.set_ylim(ymin, ymax)
        # ax1.set_aspect('equal', 'datalim')
        ax1.xaxis.set_tick_params(labelsize=self.ticksize)
        ax1.yaxis.set_tick_params(labelsize=self.ticksize)
        ax1.set_xticks(self.xticks)


        ## 2. axis 2 leaf velocity
        ax2.set_title('Leaf Velocity', fontsize=self.subsize, weight=self.fontweight)
        ax2.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax2.set_ylabel('leaf velocity', fontsize=self.subsize, weight=self.fontweight)
        self.line2 = Line2D([], [], color='grey')
        self.line2a = Line2D([], [], color='brown', linewidth=2)
        self.line2e = Line2D(
                [], [], color='brown', marker='o', markeredgecolor='brown')
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.set_xlim(0, self.flow_period)
        # ax2.set_ylim(self.vcs.min(), self.vcs.max())
        ax2.set_ylim(ymin, ymax)
        ax2.xaxis.set_tick_params(labelsize=self.ticksize)
        ax2.yaxis.set_tick_params(labelsize=self.ticksize)
        ax2.set_xticks(self.xticks)


        ## 3. axis 3 leaf motion
        ax3.set_title('Leaf Motion', fontsize=self.subsize, weight=self.fontweight)
        ax3.set_xlabel('x', fontsize=self.subsize, weight=self.fontweight)
        ax3.set_ylabel('z', fontsize=self.subsize, weight=self.fontweight)
        self.line3 = Line2D([], [], color='green', linewidth=4)
        self.quiver3f = ax3.quiver(self.leaf.R, self.leaf.R, [], [], color='brown',\
                units = 'xy', scale=1, headlength=5, headwidth=3, width=0.4)
        ax3.add_line(self.line3)
        self.quiver3h = ax3.quiver(self.leaf.R * 0.5, self.leaf.R * 2.3, [], [],\
                color='red', units='xy', scale=1, headlength=5, headwidth=4, width=0.4)
        self.quiver3l = ax3.quiver(self.leaf.R * 1.5, self.leaf.R * 2.3, [], [],\
                color='orange', units='xy', scale=1, headlength=5, headwidth=4, width=0.4)

        leaf_xmax, leaf_ymax = leaf.get_axes_limits()

        ax3.set_xlim(0, leaf_xmax)
        ax3.set_ylim(0, leaf_ymax)
        ax3.xaxis.set_tick_params(labelsize=self.ticksize)
        ax3.yaxis.set_tick_params(labelsize=self.ticksize)


        ## 4. axis 4 pz energy
        ax4.set_title('Piezoelectricity Energy', fontsize=self.subsize, weight=self.fontweight)
        ax4.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax4.set_ylabel('piezoelectricity energy', fontsize=self.subsize, weight=self.fontweight)
        self.line4 = Line2D([], [], color='grey', linewidth=2)
        self.line4a = Line2D([], [], color='blue', linewidth=2)
        self.line4e = Line2D(
                [], [], color = 'blue', marker='o', markeredgecolor='blue')
        ax4.add_line(self.line4)
        ax4.add_line(self.line4a)
        ax4.add_line(self.line4e)
        ax4_ymin, ax4_ymax = compute_axis_limits(self.pz_es.min(), self.pz_es.max(), margin=0.1)
        ax4.set_xlim(0, self.flow_period)
        ax4.set_ylim(ax4_ymin, ax4_ymax)
        ax4.xaxis.set_tick_params(labelsize=self.ticksize)
        ax4.yaxis.set_tick_params(labelsize=self.ticksize)
        # ax4.set_aspect('equal', 'datalim')
        ax4.set_xticks(self.xticks)


        ## 5. axis 5 heat and light source flux
        ax5.set_title('Heat and Light Source Flux', fontsize=self.subsize, weight=self.fontweight)
        ax5.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax5.set_ylabel('Heat and light source flux', fontsize=self.subsize, weight=self.fontweight)
        self.line5h = Line2D([], [], color='grey', linewidth=2)
        self.line5ha = Line2D([], [], color='red', linewidth=2, label='heat')
        self.line5he = Line2D(
                [], [], color = 'red', marker='o', markeredgecolor='red')
        ax5.add_line(self.line5h)
        ax5.add_line(self.line5ha)
        ax5.add_line(self.line5he)

        self.line5l = Line2D([], [], color='grey', linewidth=2)
        self.line5la = Line2D([], [], color='orange', linewidth=2, label='light')
        self.line5le = Line2D(
                [], [], color = 'orange', marker='o', markeredgecolor='orange')
        ax5.add_line(self.line5l)
        ax5.add_line(self.line5la)
        ax5.add_line(self.line5le)
        # _, ax5_ymax = compute_axis_limits(self.light_es.min(), self.light_es.max(), margin=0.1)
        # ax5_ymin, ax5_ymax = compute_axis_limits(self.light_ws.min(), self.light_ws.max(), margin=0.1)
        ax5_ymin, ax5_ymax = compute_axis_limits(0, max(self.Ihs.max(), self.Ils.max()), margin=0.2)
        ax5.set_xlim(0, self.flow_period)
        ax5.legend(loc='upper left')
        ax5.set_ylim(ax5_ymin, ax5_ymax)
        ax5.xaxis.set_tick_params(labelsize=self.ticksize)
        ax5.yaxis.set_tick_params(labelsize=self.ticksize)
        ax5.set_xticks(self.xticks)


        ## 6. axis 6 heat and light absorption power
        ax6.set_title('Heat and Light Absorption Power', fontsize=self.subsize, weight=self.fontweight)
        ax6.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax6.set_ylabel('Heat and light absorption power', fontsize=self.subsize, weight=self.fontweight)
        self.line6h = Line2D([], [], color='grey', linewidth=2)
        self.line6ha = Line2D([], [], color='red', linewidth=2, label='heat')
        self.line6he = Line2D(
                [], [], color = 'red', marker='o', markeredgecolor='red')
        ax6.add_line(self.line6h)
        ax6.add_line(self.line6ha)
        ax6.add_line(self.line6he)

        self.line6l = Line2D([], [], color='grey', linewidth=2)
        self.line6la = Line2D([], [], color='orange', linewidth=2, label='light')
        self.line6le = Line2D(
                [], [], color = 'orange', marker='o', markeredgecolor='orange')
        ax6.add_line(self.line6l)
        ax6.add_line(self.line6la)
        ax6.add_line(self.line6le)

        ax6.legend(loc='upper left')
        ax6_ymin, ax6_ymax = compute_axis_limits(0, max(self.light_ws.max(), self.heat_ws.max()), margin=0.2)
        ax6.set_xlim(0, self.flow_period)
        ax6.set_ylim(ax6_ymin, ax6_ymax)
        ax6.xaxis.set_tick_params(labelsize=self.ticksize)
        ax6.yaxis.set_tick_params(labelsize=self.ticksize)
        ax6.set_xticks(self.xticks)


        ## 7. axis 7 aborpt heat and light energy
        ax7.set_title('Absorpt Heat and Light Energy', fontsize=self.subsize, weight=self.fontweight)
        ax7.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax7.set_ylabel('Absorpt Heat and light energy', fontsize=self.subsize, weight=self.fontweight)
        self.line7l = Line2D([], [], color='grey', linewidth=2)
        self.line7la = Line2D([], [], color='orange', linewidth=2, label="light")
        self.line7le = Line2D(
                [], [], color = 'orange', marker='o', markeredgecolor='orange')
        ax7.add_line(self.line7l)
        ax7.add_line(self.line7la)
        ax7.add_line(self.line7le)
        self.line7h = Line2D([], [], color='grey', linewidth=2)
        self.line7ha = Line2D([], [], color='red', linewidth=2, label="heat")
        self.line7he = Line2D(
                [], [], color = 'red', marker='o', markeredgecolor='red')
        ax7.add_line(self.line7h)
        ax7.add_line(self.line7ha)
        ax7.add_line(self.line7he)
        ax7.legend(loc='upper left')
        ax7_ymin, ax7_ymax = compute_axis_limits(0, max(self.heat_es.max(), self.light_es.max()), margin=0.1)
        ax7.set_xlim(0, self.flow_period)
        ax7.set_ylim(ax7_ymin, ax7_ymax)
        ax7.xaxis.set_tick_params(labelsize=self.ticksize)
        ax7.yaxis.set_tick_params(labelsize=self.ticksize)
        ax7.set_xticks(self.xticks)


        animation.TimedAnimation.__init__(self, fig, interval=ani_interval, blit=True)


    def _draw_frame(self, framedata):
        # print(framedata) # [0, n_steps - 1]
        i = framedata
        head = i - 1
        # head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.ts, self.vfs)
        self.line1a.set_data(self.ts[:i], self.vfs[:i])
        self.line1e.set_data(self.ts[head], self.vfs[head])

        self.line2.set_data(self.ts, self.vcs)
        self.line2a.set_data(self.ts[:i], self.vcs[:i])
        self.line2e.set_data(self.ts[head], self.vcs[head])


        self.leaf.set_phi(self.phis[head])
        leaf_xs, leaf_ys = self.leaf.get_shape_coords()
        self.line3.set_data(leaf_xs, leaf_ys)
        dfx = 0.8 * self.leaf.R * self.vfs[head] / self.flow.vf0
        self.quiver3f.set_UVC(dfx, 0)

        dhy = -0.6 * self.leaf.R * self.Ihs[head] / self.heat.Ih0
        self.quiver3h.set_UVC(0, dhy)
        dly = -0.6 * self.leaf.R * self.Ils[head] / self.light.Il0
        self.quiver3l.set_UVC(0, dly)
        # self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        # self.line3e.set_data(self.x[head], self.z[head])

        self.line4.set_data(self.ts, self.pz_es)
        self.line4a.set_data(self.ts[:i], self.pz_es[:i])
        self.line4e.set_data(self.ts[head], self.pz_es[head])

        self.line5h.set_data(self.ts, self.Ihs)
        self.line5ha.set_data(self.ts[:i], self.Ihs[:i])
        self.line5he.set_data(self.ts[head], self.Ihs[head])

        self.line5l.set_data(self.ts, self.Ils)
        self.line5la.set_data(self.ts[:i], self.Ils[:i])
        self.line5le.set_data(self.ts[head], self.Ils[head])

        self.line6h.set_data(self.ts, self.heat_ws)
        self.line6ha.set_data(self.ts[:i], self.heat_ws[:i])
        self.line6he.set_data(self.ts[head], self.heat_ws[head])

        self.line6l.set_data(self.ts, self.light_ws)
        self.line6la.set_data(self.ts[:i], self.light_ws[:i])
        self.line6le.set_data(self.ts[head], self.light_ws[head])

        self.line7l.set_data(self.ts, self.light_es)
        self.line7la.set_data(self.ts[:i], self.light_es[:i])
        self.line7le.set_data(self.ts[head], self.light_es[head])

        self.line7h.set_data(self.ts, self.heat_es)
        self.line7ha.set_data(self.ts[:i], self.heat_es[:i])
        self.line7he.set_data(self.ts[head], self.heat_es[head])


        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2, self.line2a, self.line2e,
                               self.line3, self.quiver3f, self.quiver3h, self.quiver3l,
                               self.line4, self.line4a, self.line4e,
                               self.line5h, self.line5ha, self.line5he,
                               self.line5l, self.line5la, self.line5le,
                               self.line6h, self.line6ha, self.line6he,
                               self.line6l, self.line6la, self.line6le,
                               self.line7l, self.line7la, self.line7le,
                               self.line7h, self.line7ha, self.line7he]


    def new_frame_seq(self):
        return iter(range(self.ts.size))


    def _init_draw(self):
        lines = [self.line1, self.line1a, self.line1e,
                 self.line2, self.line2a, self.line2e,
                 self.line3,
                 self.line4, self.line4a, self.line4e,
                 self.line5h, self.line5ha, self.line5he,
                 self.line5l, self.line5la, self.line5le,
                 self.line6h, self.line6ha, self.line6he,
                 self.line6l, self.line6la, self.line6le,
                 self.line7l, self.line7la, self.line7le,
                 self.line7h, self.line7ha, self.line7he]

        for l in lines:
            l.set_data([], [])
        self.quiver3f.set_UVC(0, 0)
        self.quiver3h.set_UVC(0, 0)
        self.quiver3l.set_UVC(0, 0)


    def print_info(self):
        ## print some information
        print("Apz = {}, Br2 = {}, omega = {}".format(self.Apz, self.Br2, self.flow.get_omega()))
        print("max phi = {}, min phi = {}".format(np.rad2deg(self.phis.max()), np.rad2deg(self.phis.min())))
        print("total energy = {}".format(self.pz_es.max()))


    def save_figs(self):
        """save different axes to figres"""
        fig_format = 'svg'
        dpi = 1200
        prefix="{:.2f}-{:.2f}-".format(self.Apz, self.Br2)
        # 1. velocity of flow
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.ts, self.vfs, color='brown', linewidth=2)
        ax.set_title('Flow Velocity', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel('t', fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('velocity of flow', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlim(0, self.flow_period)
        ymin, ymax = compute_axis_limits(-self.flow.vf0, self.flow.vf0)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'flow-velocity.' + fig_format, format=fig_format, dpi=dpi)

        # 2. velocity of leaf
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Leaf Velocity', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('leaf velocity', fontsize=self.subsize, weight=self.fontweight)
        ax.plot(self.ts, self.vcs, color='brown', linewidth=2)
        ax.set_xlim(0, self.flow_period)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'leaf-velocity.' + fig_format, format=fig_format, dpi=dpi)

        # 3. pz energy
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Piezoelectricity Energy', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('Piezoelectricity energy', fontsize=self.subsize, weight=self.fontweight)
        ax.plot(self.ts, self.pz_es, color="blue", linewidth=2)
        ax_ymin, ax_ymax = compute_axis_limits(self.pz_es.min(), self.pz_es.max(), margin=0.1)
        ax.set_xlim(0, self.flow_period)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'pz-energy.' + fig_format, format=fig_format, dpi=dpi)

        # 4. light and heat flux
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title('Heat and Light Source Flux', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('Heat and light source flux', fontsize=self.subsize, weight=self.fontweight)
        ax.plot(self.ts, self.Ihs, color='red', linewidth=2, label='heat')
        ax.plot(self.ts, self.Ils, color='orange', linewidth=2, label='light')
        ax_ymin, ax_ymax = compute_axis_limits(0, max(self.Ihs.max(), self.Ils.max()), margin=0.2)
        ax.set_xlim(0, self.flow_period)
        ax.legend(loc='upper left')
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'light-heat-flux.' + fig_format, format=fig_format, dpi=dpi)

        # 5. light and heat power
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Heat and Light Absorption Power', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('Heat and light absorption power', fontsize=self.subsize, weight=self.fontweight)
        ax.plot(self.ts, self.heat_ws, color='red', linewidth=2, label='heat')
        ax.plot(self.ts, self.light_ws, color='orange', linewidth=2, label='light')
        ax.legend(loc='upper left')
        ax_ymin, ax_ymax = compute_axis_limits(0, max(self.light_ws.max(), self.heat_ws.max()), margin=0.2)
        ax.set_xlim(0, self.flow_period)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'light-heat-power.' + fig_format, format=fig_format, dpi=dpi)

        # 6. light and heat energy
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Absorpt Heat and Light Energy', fontsize=self.subsize, weight=self.fontweight)
        ax.set_xlabel(self.xlabel, fontsize=self.subsize, weight=self.fontweight)
        ax.set_ylabel('Absorpt Heat and light energy', fontsize=self.subsize, weight=self.fontweight)
        ax.plot(self.ts, self.heat_es, color='red', linewidth=2, label='heat')
        ax.plot(self.ts, self.light_es, color='orange', linewidth=2, label='light')
        ax.legend(loc='upper left')
        ax_ymin, ax_ymax = compute_axis_limits(0, max(self.heat_es.max(), self.light_es.max()), margin=0.1)
        ax.set_xlim(0, self.flow_period)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.xaxis.set_tick_params(labelsize=self.ticksize)
        ax.yaxis.set_tick_params(labelsize=self.ticksize)
        ax.set_xticks(self.xticks)
        plt.savefig(prefix + 'light-heat-energy.' + fig_format, format=fig_format, dpi=dpi)


def main(Apz, Br2, omega, save_video=False, save_figs=False):
    # 1. define leaf
    phi = np.pi * 2
    R = 10
    E1, E2 = 1000, 1000
    h = 0.5
    l0 = 20
    w = 1
    Cpz = 1
    leaf = Leaf(phi, R, E1, E2, h, l0, w, Cpz)
    # 2. define flow
    vf0 = 1
    rho = 1000
    Cd = 1
    flow = Flow(vf0, omega, rho, Cd)
    # 3. define light
    Il0 = 0.001
    omega_l = np.pi / 50
    phi_l = 0
    kl = 0.8
    light = Light(Il0, omega_l, phi_l, kl)
    # 4. define heat
    Ih0 = 0.0015
    omega_h = np.pi / 50
    phi_h = np.pi
    kh = 0.5
    heat = Heat(Ih0, omega_h, phi_h, kh)
    FFMpeg_writer = animation.writers['ffmpeg']
    mywriter = FFMpeg_writer(fps=30, bitrate=10000)

    ani = SubplotAnimation(leaf, flow, light, heat, Apz=Apz, Br2=Br2, n_steps = 400, ani_interval=20)
    filename = '{:0.4f}-{:04d}-{:0.4f}.mp4'.format(Apz, Br2, omega/np.pi)
    if save_video:
        # ani.save(filename, fps=60, extra_args=['-vcodec', 'libx264'])
        ani.save(filename, writer=mywriter)
        print("save animation to {}".format(filename))
    if save_figs:
        ani.save_figs()
    plt.show()


def test_Br2():
    Br2s = [5, 10, 20, 40, 80, 100, 200, 500, 800, 1000, 2000]
    Apz = 0.01
    omega = np.pi / 100
    for Br2 in Br2s:
        main(Apz, Br2, omega)


def test_Apz():
    Br2 = 100
    omega = np.pi / 100
    Apzs = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
    for Apz in Apzs:
        main(Apz, Br2, omega)


def test_omega():
    Br2 = 100
    Apz = 0.01
    omegas = np.pi * np.array([1/5, 1/10, 1/20, 1/50, 1/80, 1/100, 1/120])
    for omega in omegas:
        main(Apz, Br2, omega)


def test_ani(Apz=0.01, Br2=100):
    # Br2 = 100  # 100
    # Apz = 0.01 # 0.01
    omega = np.pi / 100
    save_video = False
    save_figs = False
    main(Apz, Br2, omega, save_video=save_video, save_figs=save_figs)

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    # test_Br2()
    # test_Apz()
    # test_omega()
    if len(sys.argv) != 3:
        print("Usage: >> python {} <Apz> <Br2>".format(sys.argv[0]))
        print("e.g. python {} 0.1 100".format(sys.argv[0]))
        sys.exit(1)
    Apz = float(sys.argv[1])
    Br2 = int(sys.argv[2])
    test_ani(Apz, Br2)
