from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import seaborn as sns
# import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib.gridspec import GridSpec
# %matplotlib inline

toggle = False # if false, solve ODE; if true, find equilibria
labels = [0, 1]

def generate_phase_portrait(f, pars, Nframes=200, yrange=[-3.5, 3.5]):
    xrange = [-4, 4]
    Neval = 10
    T = 0.1*Nframes

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    plt.subplots_adjust(bottom=0.2)

    button = plt.axes([0.15, 0.0, 0.26, 0.05])   # [left, bottom, width, height]
    labels[0] = button.text(0.05, 0.3, "Click here to find equilibria", fontsize=10)
    labels[1] = button.text(0.10, 0.3, "Click here to solve ODE", fontsize=10)
    labels[0].set_visible(True)
    labels[1].set_visible(False)
    button.set(xticks=[], yticks=[])
    button.set_facecolor('tab:green')

    ax[0].set_title("Phase portrait")
    ax[0].set_xlabel('angle')
    ax[0].set_ylabel('angular velocity')
    ax[0].set_xlim(xrange)
    ax[0].set_ylim(yrange)
    ax[0].grid(True)
    line1, = ax[0].plot([], [], linewidth=2, color="tab:purple", zorder=2)
    line2, = ax[0].plot([], [], 'o', markersize=7.5, color="tab:purple", zorder=2)
    plotdf(f, pars, xrange, yrange, [11, 11], ax[0])

    ax[1].set_title("Pendulum")
    ax[1].axis([-1.2, 1.2, -1.2, 1.2])
    ax[1].set_aspect('equal')
    line3, = ax[1].plot([], [], linewidth=4, color="tab:blue")
    line4, = ax[1].plot([], [], 'o', markersize=10, color="tab:red")



    def on_click(event):
        global toggle, labels
        if event.button is MouseButton.LEFT:
            if event.inaxes == button:
                labels[toggle].set_visible(False)
                toggle = not toggle
                labels[toggle].set_visible(True)
                fig.canvas.draw_idle()
                return

            if event.inaxes == ax[0]:
                x = event.xdata
                y = event.ydata
                if x is None or y is None:
                    return
                if not (xrange[0] < x < xrange[1] and yrange[0] < y < yrange[1]):
                    return

                if toggle:
                    equilibrium = fsolve(lambda z: f(0, z, pars), [x, y])
                    ax[0].scatter(equilibrium[0], equilibrium[1], s=70, color="tab:red", zorder=2)
                    point = [np.sin(equilibrium[0]), -np.cos(equilibrium[0])]
                    line3.set_data([0, point[0]], [0, point[1]])
                    line4.set_data([point[0]], [point[1]])
                    fig.canvas.draw_idle()
                    return

                # --- trajectory mode (timer-based animation) ---
                sol = solve_ivp(
                    f, [0, T], [x, y],
                    t_eval=np.linspace(0, T, Neval * Nframes),
                    args=[pars], atol=1e-8, rtol=1e-6
                )

                # Keep timer refs alive on the figure
                if not hasattr(fig, "_phaseportrait_timers"):
                    fig._phaseportrait_timers = []

                state = {"k": 1}
                timer = fig.canvas.new_timer(interval=50)  # ms


                def step():
                    k = state["k"]
                    update_graph(k, sol.y, Nframes, Neval, ax[0], line1, line2, line3, line4)
                    fig.canvas.draw_idle()

                    state["k"] = k + 1

                    if state["k"] >= int(Nframes/10):
                        # finalize: draw the complete trajectory once, leave it
                        soln = convert2circle(sol.y.copy())   # copy if convert2circle mutates
                        ax[0].plot(soln[0, :], soln[1, :], linewidth=2, zorder=1)

                        # optional: clear the animated "cursor" artists
                        line1.set_data([], [])
                        line2.set_data([], [])

                        fig.canvas.draw_idle()
                        timer.stop()

                timer.add_callback(step)
                fig._phaseportrait_timers.append(timer)
                timer.start()
                fig.canvas.draw_idle()

    # def on_click(event):
    #     global toggle, labels
    #     if event.button is MouseButton.LEFT:
    #         if event.inaxes == button:
    #             labels[toggle].set_visible(False)
    #             toggle = not toggle
    #             labels[toggle].set_visible(True)
    #         if event.inaxes == ax[0]:
    #             x = event.xdata
    #             y = event.ydata
    #             if x>xrange[0] and x<xrange[1] and y>yrange[0] and y<yrange[1]:
    #                 if toggle:
    #                     equilibrium = fsolve(lambda x: f(0, x, pars), [x, y])
    #                     ax[0].scatter(equilibrium[0], equilibrium[1], s=70, color='tab:red', zorder=2)
    #                     point = [np.sin(equilibrium[0]), -np.cos(equilibrium[0])]
    #                     line3.set_data([0, point[0]], [0, point[1]])
    #                     line4.set_data(point[0], point[1])
    #                     fig.canvas.draw()
    #                 else:
    #                     sol = solve_ivp(f, [0, T], [x, y], t_eval=np.linspace(0, T, Neval*Nframes),
    #                                     args=[pars], atol=1.e-8, rtol=1.e-6)
    #                     ani = animation.FuncAnimation(fig, update_graph, frames=range(1, Nframes),
    #                                     fargs=(sol.y, Nframes, Neval, ax[0], line1, line2, line3, line4),
    #                                     interval=50, repeat=False)
    #                     fig._phaseportrait_ani = ani
    #                     fig.canvas.draw_idle()
    #                     # fig.canvas.draw()

    # plt.connect('button_press_event', on_click)
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    fig._phaseportrait_cid = cid
    

def update_graph(k, u, Nframes, N, ax, line1, line2, line3, line4):
    soln = convert2circle(u[:, :k*N])
    line1.set_data(soln[0, :], soln[1, :])
    line2.set_data([soln[0, -1]], [soln[1, -1]])
    point = [np.sin(soln[0, -1]), -np.cos(soln[0, -1])]
    line3.set_data([0, point[0]], [0, point[1]])
    line4.set_data([point[0]], [point[1]])
    if k==Nframes-1:
        ax.plot(soln[0,:], soln[1,:], linewidth=2, color="tab:green", zorder=1)
    ax.figure.canvas.draw_idle()

def convert2circle(u):
    u = u.copy()
    u[0,:] = np.mod(u[0,:]+np.pi, 2*np.pi) - np.pi
    jumps = abs(np.diff(u[0,:]))
    ind = 1+np.where(jumps > np.pi)[0]
    return np.insert(u, ind, np.nan, axis=1)

def plotdf(rhs, pars, xrange, yrange, grid, ax):
    x = np.linspace(xrange[0], xrange[1], grid[0])
    y = np.linspace(yrange[0], yrange[1], grid[1])
    X, Y = np.meshgrid(x, y)
    DX, DY = rhs(0, [X, Y], pars)
    M = (np.hypot(DX, DY))
    M[M==0] = 1.0
    DX = DX/M
    DY = DY/M
    ax.quiver(X, Y, DX, DY, color='tab:gray', angles='xy', alpha=0.5)

def calc_portrait(pars = (0.2, 0.2, 5.7), T = 300, N = 1, u0 = [1, 1, 1], system=None,
             jitter=(-0.01, 0.01, 3)):
    t = np.linspace(0, T, int(100*T))
    soln = solve_ivp(system, [0, 5000], u0, args=pars)
    u0 = [soln.y[0][-1], soln.y[1][-1], soln.y[2][-1]]
    x_val, y_val, z_val, t_val =[],[],[], []#np.empty([1]),np.empty([1]),np.empty([1])
    for n in range(N):
        u1 = u0 + np.random.uniform(*jitter)
        soln = solve_ivp(system, [0, T], u1, t_eval=t, args=pars)
        x_val.append(soln.y[0])
        y_val.append(soln.y[1])
        z_val.append(soln.y[2])
        t_val.append(soln.t)

    x_val = np.concatenate(x_val,axis=None)
    y_val = np.concatenate(y_val,axis=None)
    z_val = np.concatenate(z_val,axis=None)
    t_val = np.concatenate(t_val,axis=None)
    return x_val, y_val, z_val, t_val


def plot_portrait(data, plots=['xyz'], cols=None, orientation='h', vstretch= 2, hstretch=1, freeze_motion=False):
    pal = sns.color_palette('tab20')
    T = data.pop('T', 'No Key found')
    max_time_int = data.pop('max_time_int', T * 4)

    lim_d = {}
    for axis in ['x_val', 'y_val', 'z_val', 't_val']:
        lim_d[axis] = [np.floor(min([
                            min([min(traj[axis]) for traj in data[c]['traj']]) for c in data.keys()])),
                       np.ceil(max([
                           max([max(traj[axis]) for traj in data[c]['traj']]) for c in data.keys()]))
        ]

    if orientation == 'h':
        if cols == None:
            cols = len(plots)+1
        fig = plt.figure(figsize=(13,7))
        gs_cols = hstretch*cols
        start_subplot_col = gs_cols-hstretch*(cols-(len(plots)-1))

        gs_rows = vstretch*(len(plots)-1)
        start_subplot_row = 0

        start_main_col = 0
        end_main_col = start_subplot_col
        start_main_row = 0
        end_main_row = start_subplot_row+vstretch*(len(plots)-1)

    if orientation == 'v':
        if cols == None:
            cols = 5
        if len(plots) > 1:
            fig = plt.figure(figsize=(7,7+2*(len(plots)-1)))
        elif len(plots):
            fig = plt.figure(figsize=(7,7))
        gs_cols = hstretch*cols
        start_subplot_col = 0
        gs_rows = gs_cols+ vstretch*(len(plots)-1)
        start_subplot_row = gs_cols

        start_main_col = 0
        end_main_col = gs_cols
        start_main_row = 0
        end_main_row = start_subplot_row

    gs = fig.add_gridspec(gs_rows,gs_cols, left=0.1, right=0.9, top = .95, bottom=.1,
                          wspace=.65, hspace=1.25)
    if 'xyz' in plots:
        main_gs = gs[start_main_row:end_main_row,start_main_col:end_main_col]
        ax = fig.add_subplot(main_gs,projection='3d')

    if 'xt' in plots:
        xt_gs =gs[start_subplot_row:start_subplot_row+1*vstretch, start_subplot_col:]
        ax_xt = fig.add_subplot(xt_gs)
    if 'yt' in plots:
        yt_gs =gs[start_subplot_row+1*vstretch:start_subplot_row+2*vstretch, start_subplot_col:]
        ax_yt = fig.add_subplot(yt_gs)
    if 'xz' in plots:
        xz_gs =gs[start_subplot_row+2*vstretch:start_subplot_row+3*vstretch, start_subplot_col:]
        ax_xz = fig.add_subplot(xz_gs)

    plot_data={}
    for ip, c in enumerate(data.keys()):
        trajs =[]
        for im, trj in enumerate(data[c]['traj']):
            color = pal[im]
            x_val = trj['x_val']
            y_val = trj['y_val']
            z_val = trj['z_val']
            t_val = trj['z_val']
            plot_data_tmp = {'color':color}
            if 'xyz' in plots:
                point, = ax.plot(x_val[0], y_val[0], z_val[0], marker='o', color=color, label= str(c))
                line, = ax.plot(x_val[0], y_val[0], z_val[0], linewidth=0.75, alpha=0.685, color=color)
                plot_data_tmp.update({'line':line, 'point':point})
    #         if len(plots) >1:
            if 'xt' in plots:
                line_xt, = ax_xt.plot(t_val[0], x_val[0],lw=1, color=color)
                point_xt, = ax_xt.plot(t_val[0], x_val[0], marker='o', color=color, label=str(c))
                plot_data_tmp.update({'line_xt':line_xt, 'point_xt':point_xt})
            if 'yt' in plots:
                line_yt, = ax_yt.plot(t_val[0], y_val[0], lw=1, color=color)
                point_yt, = ax_yt.plot(t_val[0], y_val[0], marker='o', color=color, label=str(c))
                plot_data_tmp.update({'line_yt':line_yt, 'point_yt':point_yt})
            if 'xz' in plots:
                line_xz, = ax_xz.plot(x_val[0], z_val[0], lw=1, color=color)
                point_xz, = ax_xz.plot(x_val[0], z_val[0], marker='o', color=color, label=str(c))
                plot_data_tmp.update({'line_xz':line_xz, 'point_xz':point_xz})

            trj.update(plot_data_tmp)
            trajs.append(trj)

#         plot_data_tmp = {'color':color, 'line':line, 'point':point,
#                          'line_xt':line_xt, 'point_xt':point_xt,
#                         'line_yt':line_yt, 'point_yt':point_yt,
#                         'line_xz':line_xz, 'point_xz':point_xz}
        d3 = data[c].copy()
        d3.update({'traj': trajs})#plot_data_tmp)
        plot_data[c]= d3#{'traj': trajs}#d3#dict(plot_data_tmp, data[c])

        if 'xyz' in plots:
            ax.set_ylim(lim_d['y_val']) # this shouldn't be necessary but the limits are usually enlarged per defailt
            ax.set_zlim(lim_d['z_val']) # this shouldn't be necessary but the limits are usually enlarged per defailt
            ax.set_xlim(lim_d['x_val'])

            ax.set_title(plot_data[c]['ttl'])
            ax.set_xlabel(r'$x$',fontweight='bold')
            ax.set_ylabel(r'$y$',fontweight='bold')
            ax.set_zlabel(r'$z$',fontweight='bold')

        if 'xt' in plots:
            # ax_xt.set_xlim([lim_d['t_val'][0],lim_d['t_val'][1]/(2*T)])
            ax_xt.set_ylim(lim_d['x_val'])
            ax_xt.set_ylabel(r'$x(t)$')
            ax_xt.set_xlabel(r'$t$')
        #     ax_xt.set_title('X with respect to t')
        if 'yt' in plots:
            ax_yt.set_xlim([lim_d['t_val'][0],lim_d['t_val'][1]/(2*T)])
            ax_yt.set_ylim(lim_d['y_val'])
            ax_yt.set_ylabel(r'$y(t)$')
            ax_yt.set_xlabel(r'$t$')
        #     ax_yt.set_title('Y with respect to t')
        if 'xz' in plots:
            ax_xz.set_xlim(lim_d['x_val'])
            ax_xz.set_ylim(lim_d['z_val'])
            ax_xz.set_ylabel(r'$z(t)$')
            ax_xz.set_xlabel(r'$x(t)$')
    #     ax_xz.set_title('Z with respect to X')
        if orientation == 'v':
            ax.legend(title='c value', bbox_to_anchor=(.15,1))
        else:
            ax.legend(title='c value', bbox_to_anchor=(.15,1))

    def update(num):
        for c in plot_data.keys():
            for traj in plot_data[c]['traj']:
                if 'xyz' in plots:
                    traj['line'].set_data(traj['x_val'][:num], traj['y_val'][:num])
                    traj['line'].set_3d_properties(traj['z_val'][:num])
                    traj['point'].set_data(traj['x_val'][num-1:num], traj['y_val'][num-1:num])
                    traj['point'].set_3d_properties(traj['z_val'][num-1:num])
                if 'xt' in plots:
                    traj['line_xt'].set_data(traj['t_val'][:num], traj['x_val'][:num])
                    traj['point_xt'].set_data(traj['t_val'][num], traj['x_val'][num])
                    min_t = max([lim_d['t_val'][0],traj['t_val'][num]-2*T])
                    ax_xt.set_xlim([min_t, min_t+.1*T])

                if 'yt' in plots:
                    traj['line_yt'].set_data(traj['t_val'][:num], traj['y_val'][:num])
                    traj['point_yt'].set_data(traj['t_val'][num], traj['y_val'][num])
                if 'xz' in plots:
                    traj['line_xz'].set_data(traj['x_val'][:num], traj['z_val'][:num])
                    traj['point_xz'].set_data(traj['x_val'][num], traj['z_val'][num])
                if not freeze_motion:
                    ax.view_init(30, 0.3 * num)
                fig.canvas.draw_idle()


    time_wdgt = widgets.IntSlider(
        description='Test widget:',
        value=0,
        min=0, max=max_time_int, step=T/100,
        layout=widgets.Layout(width='100%'))


    interact(update, num=time_wdgt);