import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider as MplSlider
from matplotlib.backend_bases import MouseButton
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import seaborn as sns
from IPython.display import display
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
            if event.inaxes == ax[0]:
                x = event.xdata
                y = event.ydata
                if x>xrange[0] and x<xrange[1] and y>yrange[0] and y<yrange[1]:
                    if toggle:
                        equilibrium = fsolve(lambda x: f(0, x, pars), [x, y])
                        ax[0].scatter(equilibrium[0], equilibrium[1], s=70, color='tab:red', zorder=2)
                        point = [np.sin(equilibrium[0]), -np.cos(equilibrium[0])]
                        line3.set_data([0, point[0]], [0, point[1]])
                        line4.set_data(point[0], point[1])
                        fig.canvas.draw()
                    else:
                        sol = solve_ivp(f, [0, T], [x, y], t_eval=np.linspace(0, T, Neval*Nframes),
                                        args=[pars], atol=1.e-8, rtol=1.e-6)
                        ani = animation.FuncAnimation(fig, update_graph, frames=range(1, Nframes),
                                        fargs=(sol.y, Nframes, Neval, ax[0], line1, line2, line3, line4),
                                        interval=50, repeat=False)
                        fig.canvas.draw()

    plt.connect('button_press_event', on_click)

def update_graph(k, u, Nframes, N, ax, line1, line2, line3, line4):
    soln = convert2circle(u[:, :k*N])
    line1.set_data(soln[0, :], soln[1, :])
    line2.set_data([soln[0, -1]], [soln[1, -1]])
    point = [np.sin(soln[0, -1]), -np.cos(soln[0, -1])]
    line3.set_data([0, point[0]], [0, point[1]])
    line4.set_data([point[0]], [point[1]])
    if k==Nframes-1:
        ax.plot(soln[0,:], soln[1,:], linewidth=2, color="tab:green", zorder=1)

def convert2circle(u):
    u[0,:] = np.mod(u[0,:]+np.pi, 2*np.pi) - np.pi
    jumps = abs(np.diff(u[0,:]))
    ind = 1+np.where(jumps > np.pi)[0]
    return np.insert(u, ind, np.NaN, axis=1)

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


def plot_portrait(
    data,
    plots=['xyz'],
    figsize=None,
    cols=None,
    grid=True,
    orientation='h',
    vstretch=2,
    hstretch=1,
    freeze_motion=False,
    show_window_widget=True,
    show_end_widget=True,
    default_window_T=None,
    initial_frame=0,
    frame_tracker=None
    
):
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
        if figsize is None:
            figsize= (13,7)
        fig = plt.figure(figsize=figsize)
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
            if figsize is None:
                figsize = (7,7+2*(len(plots)-1))
            fig = plt.figure(figsize=figsize)
        elif len(plots):
            if figsize is None:
                figsize = (7,7)
            fig = plt.figure(figsize=figsize)
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
            color = trj.get('color', pal[im % len(pal)])
            label = trj.get('label', str(c))
            x_val = trj['x_val']
            y_val = trj['y_val']
            z_val = trj['z_val']
            t_val = trj['t_val']
            plot_data_tmp = {'color':color}
            if 'xyz' in plots:
                point, = ax.plot(x_val[0], y_val[0], z_val[0], marker='o', color=color, label=label)
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
            # ax_yt.set_xlim([lim_d['t_val'][0],lim_d['t_val'][1]/(2*T)])
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

    view_state = {'azim': 45.0}
    frame_state = {'num': 0}

    if default_window_T is None:
        default_window_T = T / 4
    default_window_T = float(max(0.1, min(T, default_window_T)))
    control_state = {'window_T': default_window_T, 'end_T': float(T)}

    def update(num, window_T, end_T):
        frame_state['num'] = int(num)
        end_T = float(max(0.1, min(T, end_T)))
        window_T = float(max(0.1, min(end_T, window_T)))
        control_state['window_T'] = window_T
        control_state['end_T'] = end_T
        for c in plot_data.keys():
            for traj in plot_data[c]['traj']:
                npts = len(traj['t_val'])
                samples_per_time = max(1.0, npts / max(T, 1e-6))
                end_idx = min(npts - 1, int(end_T * samples_per_time))
                idx = min(max(int(num), 0), end_idx)
                if frame_tracker is not None:
                    frame_tracker['idx'] = idx
                    frame_tracker['end_idx'] = end_idx
                    frame_tracker['samples_per_time'] = samples_per_time
                half_window = max(1, int((window_T * samples_per_time) / 2.0))
                left_idx = max(0, idx - half_window)
                right_idx = min(end_idx, idx + half_window)
                min_t = max([lim_d['t_val'][0], traj['t_val'][left_idx]])
                max_t = min([lim_d['t_val'][1], traj['t_val'][right_idx]])
                if max_t <= min_t:
                    max_t = min_t + 1e-6
                if 'xyz' in plots:
                    traj['line'].set_data(traj['x_val'][:idx+1], traj['y_val'][:idx+1])
                    traj['line'].set_3d_properties(traj['z_val'][:idx+1])
                    traj['point'].set_data(traj['x_val'][idx:idx+1], traj['y_val'][idx:idx+1])
                    traj['point'].set_3d_properties(traj['z_val'][idx:idx+1])
                if 'xt' in plots:
                    traj['line_xt'].set_data(traj['t_val'][:idx+1], traj['x_val'][:idx+1])
                    traj['point_xt'].set_data(traj['t_val'][idx:idx+1], traj['x_val'][idx:idx+1])
                    ax_xt.set_xlim([min_t,max_t])
                if 'yt' in plots:
                    traj['line_yt'].set_data(traj['t_val'][:idx+1], traj['y_val'][:idx+1])
                    traj['point_yt'].set_data(traj['t_val'][idx:idx+1], traj['y_val'][idx:idx+1])
                    ax_yt.set_xlim([min_t, max_t])
                if 'xz' in plots:
                    traj['line_xz'].set_data(traj['x_val'][:idx+1], traj['z_val'][:idx+1])
                    traj['point_xz'].set_data(traj['x_val'][idx:idx+1], traj['z_val'][idx:idx+1])
                if 'xyz' in plots:
                    ax.view_init(30, view_state['azim'])
                fig.canvas.draw_idle()


    if 'xyz' in plots:
        rot_ax = fig.add_axes([0.12, 0.03, 0.33, 0.03])
        rot_slider = MplSlider(rot_ax, 'Rotate', -180.0, 180.0, valinit=view_state['azim'], valstep=1.0)

        def on_rotate_change(val):
            view_state['azim'] = float(val)
            update(frame_state['num'], control_state['window_T'], control_state['end_T'])

        rot_slider.on_changed(on_rotate_change)

    slider_step = max(1, int(T/100))
    max_idx = 0
    for c in plot_data.keys():
        for traj in plot_data[c]['traj']:
            max_idx = max(max_idx, len(traj['t_val']) - 1)

    time_wdgt = widgets.IntSlider(
        description='Frame:',
        value=0,
        min=0, max=min(max_idx, max_time_int), step=slider_step,
        layout=widgets.Layout(width='100%'))

    if show_end_widget:
        end_wdgt = widgets.BoundedFloatText(
            description='End T:',
            value=float(T),
            min=0.1,
            max=float(T),
            step=max(0.1, T/100),
            layout=widgets.Layout(width='40%')
        )
    else:
        end_wdgt = widgets.fixed(float(T))

    if show_window_widget:
        window_wdgt = widgets.FloatSlider(
            description='Window T:',
            value=default_window_T,
            min=max(0.1, T/200),
            max=float(T),
            step=max(0.1, T/200),
            readout_format='.1f',
            layout=widgets.Layout(width='100%')
        )
    else:
        window_wdgt = widgets.fixed(default_window_T)

    init_frame = int(max(0, min(max_idx, initial_frame)))
    time_wdgt.value = init_frame
    interact(update, num=time_wdgt, window_T=window_wdgt, end_T=end_wdgt)


def launch_portrait_explorer(
    system,
    pars=(0.2, 0.2, 5.7),
    u0=(10, 10, 10),
    n_ic=5,
    T=200,
    plots=('xyz', 'xt', 'yt'),
    orientation='h',
    force_mode=None,
    fixed_members=None, 
    alpha=.5
):
    pars = tuple(pars)
    u0 = np.array(u0, dtype=float)
    plots = list(plots)

    state = {
        'run_id': 0,
        'sim_T': float(T),
        'trajs': [],
        'artists': [],
        'fig': None,
        'ax': None,
        'ax_ts': {},
        'ax_hist': {},
        'frame_idx': 0,
        'azim': 45.0,
    }

    valid_force_modes = {None, 'param_single_ic', 'single_param_multi_ic'}
    if force_mode not in valid_force_modes:
        raise ValueError("force_mode must be one of: None, 'param_single_ic', 'single_param_multi_ic'")
    if fixed_members is not None:
        fixed_members = int(fixed_members)
        if fixed_members < 1:
            raise ValueError("fixed_members must be >= 1")

    mode_wdgt = widgets.ToggleButtons(options=[('IC', 'ic'), ('Param', 'param'), ('Both', 'both')], value='ic', description='Mode:')
    start_time_wdgt = widgets.BoundedFloatText(description='Start T:', value=0.0, min=0.0, max=float(T), step=10.0)
    end_time_wdgt = widgets.BoundedFloatText(description='End T:', value=float(T), min=1.0, max=5000.0, step=10.0)
    window_wdgt = widgets.FloatSlider(description='Window T:', value=max(5.0, T / 4), min=1.0, max=float(T), step=1.0)
    frame_wdgt = widgets.IntSlider(description='Frame:', value=0, min=0, max=int(100 * T), step=max(1, int(T / 100)), layout=widgets.Layout(width='100%'))
    recurrence_tol_wdgt = widgets.FloatSlider(
        description='Rec tol frac:',
        value=0.05, min=0.01, max=0.20, step=0.005, readout_format='.3f',
        layout=widgets.Layout(width='45%')
    )
    status_wdgt = widgets.HTML(value='Ready')
    run_btn = widgets.Button(description='Run / Reset', button_style='primary')
    clear_btn = widgets.Button(description='Clear Plots', button_style='warning')
    out = widgets.Output()

    ic_same_wdgt = widgets.Checkbox(value=True, description='Same IC (jitter=0)')
    jitter_wdgt = widgets.FloatSlider(description='Jitter +/-:', value=0.0, min=0.0, max=10.0, step=0.1, readout_format='.1f')
    ic_members_wdgt = widgets.IntSlider(description='IC members:', value=int(n_ic), min=1, max=20, step=1)
    param_name_wdgt = widgets.Dropdown(options=[('a', 0), ('b', 1), ('c', 2)], value=2, description='Param:')
    param_span_wdgt = widgets.FloatSlider(description='Param +/-:', value=0.1, min=0.0, max=2.0, step=0.01, readout_format='.2f')
    param_use_bounds_wdgt = widgets.Checkbox(value=False, description='Use min/max range')
    param_min_wdgt = widgets.FloatText(description='Param min:', value=float(pars[param_name_wdgt.value]) - float(param_span_wdgt.value))
    param_max_wdgt = widgets.FloatText(description='Param max:', value=float(pars[param_name_wdgt.value]) + float(param_span_wdgt.value))
    param_members_wdgt = widgets.IntSlider(description='Param members:', value=5, min=1, max=20, step=1)

    ic_tab = widgets.VBox([ic_same_wdgt, jitter_wdgt, ic_members_wdgt])
    param_tab = widgets.VBox([param_name_wdgt, param_span_wdgt, param_use_bounds_wdgt, param_min_wdgt, param_max_wdgt, param_members_wdgt])
    tabs = widgets.Tab(children=[ic_tab, param_tab])
    tabs.set_title(0, 'IC')
    tabs.set_title(1, 'Param variation')

    def _refresh_param_range_defaults(*_):
        if param_use_bounds_wdgt.value:
            return
        center = float(pars[param_name_wdgt.value])
        span = float(param_span_wdgt.value)
        param_min_wdgt.value = center - span
        param_max_wdgt.value = center + span

    def _sync_mode(*_):
        if force_mode == 'param_single_ic':
            mode_wdgt.value = 'param'
        elif force_mode == 'single_param_multi_ic':
            mode_wdgt.value = 'ic'

        if mode_wdgt.value == 'ic':
            tabs.selected_index = 0
            ic_same_wdgt.disabled = False
            ic_members_wdgt.disabled = False
            param_name_wdgt.disabled = True
            param_span_wdgt.disabled = True
            param_use_bounds_wdgt.disabled = True
            param_min_wdgt.disabled = True
            param_max_wdgt.disabled = True
            param_members_wdgt.disabled = True
        elif mode_wdgt.value == 'param':
            tabs.selected_index = 1
            ic_same_wdgt.disabled = True
            ic_members_wdgt.disabled = True
            ic_same_wdgt.value = True
            param_name_wdgt.disabled = False
            param_use_bounds_wdgt.disabled = False
            param_span_wdgt.disabled = param_use_bounds_wdgt.value
            param_min_wdgt.disabled = not param_use_bounds_wdgt.value
            param_max_wdgt.disabled = not param_use_bounds_wdgt.value
            param_members_wdgt.disabled = False
        else:
            ic_same_wdgt.disabled = False
            ic_members_wdgt.disabled = False
            param_name_wdgt.disabled = False
            param_use_bounds_wdgt.disabled = False
            param_span_wdgt.disabled = param_use_bounds_wdgt.value
            param_min_wdgt.disabled = not param_use_bounds_wdgt.value
            param_max_wdgt.disabled = not param_use_bounds_wdgt.value
            param_members_wdgt.disabled = False
        jitter_wdgt.disabled = ic_same_wdgt.value

        if force_mode == 'param_single_ic':
            mode_wdgt.disabled = True
            tabs.disabled = True
            ic_same_wdgt.value = True
            ic_same_wdgt.disabled = True
            jitter_wdgt.disabled = True
            ic_members_wdgt.value = 1
            ic_members_wdgt.disabled = True
            param_name_wdgt.disabled = False
            param_use_bounds_wdgt.disabled = False
            param_span_wdgt.disabled = param_use_bounds_wdgt.value
            param_min_wdgt.disabled = not param_use_bounds_wdgt.value
            param_max_wdgt.disabled = not param_use_bounds_wdgt.value
            param_members_wdgt.disabled = False
            tabs.selected_index = 1
        elif force_mode == 'single_param_multi_ic':
            mode_wdgt.disabled = True
            tabs.disabled = True
            param_members_wdgt.value = 1
            param_members_wdgt.disabled = True
            param_name_wdgt.disabled = True
            param_span_wdgt.disabled = True
            param_use_bounds_wdgt.disabled = True
            param_min_wdgt.disabled = True
            param_max_wdgt.disabled = True
            ic_same_wdgt.disabled = False
            jitter_wdgt.disabled = ic_same_wdgt.value
            ic_members_wdgt.disabled = False
            tabs.selected_index = 0
        else:
            mode_wdgt.disabled = False
            tabs.disabled = False

        if fixed_members is not None:
            ic_members_wdgt.value = fixed_members
            param_members_wdgt.value = fixed_members
            ic_members_wdgt.disabled = True
            param_members_wdgt.disabled = True

    def _sync_tabs(change):
        if change['name'] != 'selected_index':
            return
        if tabs.selected_index == 0 and mode_wdgt.value == 'param':
            mode_wdgt.value = 'ic'
        if tabs.selected_index == 1 and mode_wdgt.value == 'ic':
            mode_wdgt.value = 'param'

    def _update_window_bounds(*_):
        start_time_wdgt.max = float(end_time_wdgt.value)
        if start_time_wdgt.value > start_time_wdgt.max:
            start_time_wdgt.value = start_time_wdgt.max
        window_wdgt.max = max(1.0, float(end_time_wdgt.value) - float(start_time_wdgt.value))
        if window_wdgt.value > window_wdgt.max:
            window_wdgt.value = window_wdgt.max

    def _ensure_figure():
        if state['fig'] is not None:
            return
        with out:
            fig = plt.figure(figsize=(11, 5.8))
            gs = fig.add_gridspec(
                3, 4,
                left=0.06, right=0.97, top=0.94, bottom=0.14,
                wspace=0.45, hspace=0.65,
                width_ratios=[1.4, 1.4, 2.6, 0.9]
            )
            state['ax'] = fig.add_subplot(gs[:, :2], projection='3d') if 'xyz' in plots else None
            for i, lbl in enumerate(('x', 'y', 'z')):
                state['ax_ts'][lbl] = fig.add_subplot(gs[i, 2])
                state['ax_hist'][lbl] = fig.add_subplot(gs[i, 3])

            if 'xyz' in plots:
                rot_ax = fig.add_axes([0.10, 0.055, 0.28, 0.025])
                rot_slider = MplSlider(rot_ax, 'Rotate', -180.0, 180.0, valinit=state['azim'], valstep=1.0)

                def _on_rotate(val):
                    state['azim'] = float(val)
                    _update_frame(frame_wdgt.value)
                rot_slider.on_changed(_on_rotate)

            state['fig'] = fig
            display(fig)

    def _compute_trajectories(T_local):
        if force_mode == 'param_single_ic':
            use_param = True
            n_ic_local = 1
            n_param_local = int(param_members_wdgt.value)
        elif force_mode == 'single_param_multi_ic':
            use_param = False
            n_ic_local = int(ic_members_wdgt.value)
            n_param_local = 1
        else:
            use_param = mode_wdgt.value in ('param', 'both')
            n_ic_local = 1 if mode_wdgt.value == 'param' else int(ic_members_wdgt.value)
            n_param_local = int(param_members_wdgt.value) if use_param else 1

        if fixed_members is not None:
            if mode_wdgt.value == 'param':
                n_ic_local = 1
                n_param_local = fixed_members
            elif mode_wdgt.value == 'ic':
                n_ic_local = fixed_members
                n_param_local = 1
            else:
                n_ic_local = fixed_members
                n_param_local = fixed_members
        jitter_amp = 0.0 if ic_same_wdgt.value else float(jitter_wdgt.value)

        ic_cmaps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys']
        param_colors = np.linspace(0.35, 0.95, n_param_local)
        if use_param:
            if param_use_bounds_wdgt.value:
                pmin = float(param_min_wdgt.value)
                pmax = float(param_max_wdgt.value)
                if pmax < pmin:
                    pmin, pmax = pmax, pmin
                param_values = np.linspace(pmin, pmax, n_param_local)
            else:
                p_center = float(pars[param_name_wdgt.value])
                p_offsets = np.linspace(-param_span_wdgt.value, param_span_wdgt.value, n_param_local)
                param_values = p_center + p_offsets
        else:
            param_values = [float(pars[param_name_wdgt.value])]
        trajs = []
        for ic_idx in range(n_ic_local):
            u0_i = u0.copy() if jitter_amp == 0.0 else (u0 + np.random.uniform(-jitter_amp, jitter_amp, 3))
            cmap = plt.get_cmap(ic_cmaps[ic_idx % len(ic_cmaps)])
            for p_idx in range(n_param_local):
                pars_i = list(pars)
                if use_param:
                    pars_i[param_name_wdgt.value] = float(param_values[p_idx])
                pars_i = tuple(pars_i)
                # Use a direct integration here so "same IC" truly means identical u0 across parameter members.
                t = np.linspace(0, T_local, int(100 * T_local))
                soln = solve_ivp(system, [0, T_local], u0_i, t_eval=t, args=pars_i)
                x_val, y_val, z_val, t_val = soln.y[0], soln.y[1], soln.y[2], soln.t
                trajs.append({
                    'x_val': x_val, 'y_val': y_val, 'z_val': z_val, 't_val': t_val, 'color': cmap(param_colors[p_idx]),
                    'label': (f"p={float(param_values[p_idx]):.3f}" if use_param and n_ic_local == 1
                              else (f"IC {ic_idx+1}, p={float(param_values[p_idx]):.3f}" if use_param else f"IC {ic_idx+1}"))
                })
        return trajs, n_ic_local, n_param_local

    def _limits():
        lims = {}
        for k in ['x_val', 'y_val', 'z_val', 't_val']:
            lims[k] = [np.floor(min(np.min(tr[k]) for tr in state['trajs'])), np.ceil(max(np.max(tr[k]) for tr in state['trajs']))]
        return lims

    def _clear_axes():
        all_axes = [state['ax']] + list(state['ax_ts'].values()) + list(state['ax_hist'].values())
        for a in all_axes:
            if a is not None:
                a.clear()
        state['artists'] = []

    def _update_recurrence_axes(start_idx, end_idx):
        if len(state['trajs']) == 0:
            return
        start_idx = int(max(0, start_idx))
        end_idx = int(max(start_idx, end_idx))
        labels = ('x', 'y', 'z')
        rec_bins = 16
        rec_min_separation = max(0.2, float(state['sim_T']) / 400.0)

        for lbl in labels:
            axh = state['ax_hist'][lbl]
            axh.clear()
            intervals_all = []

            for tr in state['trajs']:
                t_win = tr['t_val'][start_idx:end_idx + 1]
                s_win = tr[f'{lbl}_val'][start_idx:end_idx + 1]
                if len(t_win) < 6:
                    continue
                ref = float(np.median(s_win))
                span = float(np.max(s_win) - np.min(s_win))
                tol = max(1e-10, float(recurrence_tol_wdgt.value) * span)
                intervals, _ = _recurrence_intervals_1d(
                    t=t_win,
                    series=s_win,
                    ref_value=ref,
                    tol=tol,
                    min_separation=rec_min_separation,
                )
                if intervals.size:
                    intervals_all.append((intervals, tr['color']))

            if len(intervals_all) == 0:
                axh.text(0.5, 0.5, "n/a", ha='center', va='center', transform=axh.transAxes, fontsize=8)
                axh.set_title(f"{lbl} Δt", fontsize=9)
                axh.set_yticks([])
                continue

            vals = np.concatenate([itv for itv, _ in intervals_all])
            # Plot each member's recurrence intervals in that member's trajectory color.
            for itv, clr in intervals_all:
                axh.hist(itv, bins=rec_bins, density=True, alpha=0.30, color=clr)
            med = float(np.median(vals))
            axh.axvline(med, color='0.2', ls='--', lw=0.9)
            max_dev = float(np.max(np.abs(vals - med))) if vals.size else 0.0
            if max_dev <= 1e-12:
                axh.set_xlim(med - 0.5, med + 0.5)
            else:
                axh.set_xlim(med - 1.1 * max_dev, med + 1.1 * max_dev)
            axh.set_title(f"{lbl} Δt", fontsize=9)
            axh.grid(True, alpha=0.25)
            axh.tick_params(axis='both', labelsize=8)
            if lbl == 'z':
                axh.set_xlabel('interval', fontsize=8)

    def _init_artists():
        _clear_axes()
        if len(state['trajs']) == 0:
            return
        lims = _limits()
        for tr in state['trajs']:
            item = {'tr': tr}
            if state['ax'] is not None:
                p, = state['ax'].plot(tr['x_val'][0:1], tr['y_val'][0:1], tr['z_val'][0:1], marker='o', color=tr['color'], label=tr['label'])
                l, = state['ax'].plot(tr['x_val'][0:1], tr['y_val'][0:1], tr['z_val'][0:1], lw=0.9, alpha=alpha, color=tr['color'])
                item['p'] = p; item['l'] = l
            ltx, = state['ax_ts']['x'].plot(tr['t_val'][0:1], tr['x_val'][0:1], lw=1, color=tr['color'])
            ptx, = state['ax_ts']['x'].plot(tr['t_val'][0:1], tr['x_val'][0:1], marker='o', color=tr['color'])
            lty, = state['ax_ts']['y'].plot(tr['t_val'][0:1], tr['y_val'][0:1], lw=1, color=tr['color'])
            pty, = state['ax_ts']['y'].plot(tr['t_val'][0:1], tr['y_val'][0:1], marker='o', color=tr['color'])
            ltz, = state['ax_ts']['z'].plot(tr['t_val'][0:1], tr['z_val'][0:1], lw=1, color=tr['color'])
            ptz, = state['ax_ts']['z'].plot(tr['t_val'][0:1], tr['z_val'][0:1], marker='o', color=tr['color'])
            item['ltx'] = ltx; item['ptx'] = ptx
            item['lty'] = lty; item['pty'] = pty
            item['ltz'] = ltz; item['ptz'] = ptz
            state['artists'].append(item)
        if state['ax'] is not None:
            state['ax'].set_xlim(lims['x_val']); state['ax'].set_ylim(lims['y_val']); state['ax'].set_zlim(lims['z_val'])
            state['ax'].set_xlabel('x'); state['ax'].set_ylabel('y'); state['ax'].set_zlabel('z')
            state['ax'].legend(loc='upper left', fontsize=8)
        state['ax_ts']['x'].set_ylim(lims['x_val']); state['ax_ts']['x'].set_ylabel('x(t)')
        state['ax_ts']['y'].set_ylim(lims['y_val']); state['ax_ts']['y'].set_ylabel('y(t)')
        state['ax_ts']['z'].set_ylim(lims['z_val']); state['ax_ts']['z'].set_ylabel('z(t)'); state['ax_ts']['z'].set_xlabel('t')
        for lbl in ('x', 'y', 'z'):
            state['ax_ts'][lbl].grid(True, alpha=0.25)
            state['ax_ts'][lbl].tick_params(axis='both', labelsize=8)
            state['ax_hist'][lbl].set_title(f"{lbl} Δt", fontsize=9)
            state['ax_hist'][lbl].grid(True, alpha=0.25)
            state['ax_hist'][lbl].tick_params(axis='both', labelsize=8)
        _update_recurrence_axes(0, min(10, len(state['trajs'][0]['t_val']) - 1))

    def _refresh_frame_max(reset=False):
        if len(state['trajs']) == 0:
            frame_wdgt.max = 0; frame_wdgt.value = 0; return
        npts = len(state['trajs'][0]['t_val'])
        samples_per_time = max(1.0, npts / max(state['sim_T'], 1e-6))
        start_idx = min(npts - 1, int(float(start_time_wdgt.value) * samples_per_time))
        end_idx = min(npts - 1, int(float(end_time_wdgt.value) * samples_per_time))
        frame_wdgt.min = max(0, start_idx)
        frame_wdgt.max = max(0, end_idx)
        if reset:
            frame_wdgt.value = frame_wdgt.min
        else:
            frame_wdgt.value = min(max(frame_wdgt.value, frame_wdgt.min), frame_wdgt.max)

    def _update_frame(idx):
        if len(state['trajs']) == 0:
            return
        end_T = float(end_time_wdgt.value)
        start_T = float(start_time_wdgt.value)
        window_T = float(window_wdgt.value)
        start_idx_global, j_global, li_global, ri_global = 0, 0, 0, 1
        for item in state['artists']:
            tr = item['tr']
            npts = len(tr['t_val'])
            samples_per_time = max(1.0, npts / max(state['sim_T'], 1e-6))
            start_idx = min(npts - 1, int(start_T * samples_per_time))
            end_idx = min(npts - 1, int(end_T * samples_per_time))
            j = min(max(int(idx), start_idx), end_idx)
            hw = max(1, int((window_T * samples_per_time) / 2.0))
            li = max(0, j - hw); ri = min(end_idx, j + hw)
            li = max(li, start_idx)
            start_idx_global, j_global, li_global, ri_global = start_idx, j, li, ri
            if 'l' in item:
                item['l'].set_data(tr['x_val'][start_idx:j+1], tr['y_val'][start_idx:j+1]); item['l'].set_3d_properties(tr['z_val'][start_idx:j+1])
                item['p'].set_data(tr['x_val'][j:j+1], tr['y_val'][j:j+1]); item['p'].set_3d_properties(tr['z_val'][j:j+1])
            item['ltx'].set_data(tr['t_val'][start_idx:j+1], tr['x_val'][start_idx:j+1]); item['ptx'].set_data(tr['t_val'][j:j+1], tr['x_val'][j:j+1])
            item['lty'].set_data(tr['t_val'][start_idx:j+1], tr['y_val'][start_idx:j+1]); item['pty'].set_data(tr['t_val'][j:j+1], tr['y_val'][j:j+1])
            item['ltz'].set_data(tr['t_val'][start_idx:j+1], tr['z_val'][start_idx:j+1]); item['ptz'].set_data(tr['t_val'][j:j+1], tr['z_val'][j:j+1])
        if state['ax'] is not None:
            state['ax'].view_init(30, state['azim'])
            state['ax'].set_title(f"Rössler-like system for [{', '.join(['%.2f']*len(pars))}] | run #{state['run_id']}" % pars)
        if len(state['artists']) > 0:
            t_left = state['artists'][0]['tr']['t_val'][li_global]
            t_right = max(state['artists'][0]['tr']['t_val'][ri_global], t_left + 1e-6)
            for lbl in ('x', 'y', 'z'):
                state['ax_ts'][lbl].set_xlim([t_left, t_right])
            _update_recurrence_axes(start_idx_global, j_global)
        state['frame_idx'] = int(idx)
        state['fig'].canvas.draw_idle()

    def _run(reason='manual'):
        _ensure_figure()
        state['run_id'] += 1
        state['sim_T'] = float(end_time_wdgt.value)
        state['trajs'], n_ic_local, n_param_local = _compute_trajectories(state['sim_T'])
        _init_artists()
        _refresh_frame_max(reset=(reason == 'manual'))
        _update_frame(frame_wdgt.value)
        status_wdgt.value = f"Run #{state['run_id']}: mode={mode_wdgt.value}, Start T={float(start_time_wdgt.value):.1f}, End T={state['sim_T']:.1f}, IC members={n_ic_local}, Param members={n_param_local}"

    def _clear(_=None):
        _ensure_figure()
        state['trajs'] = []
        _clear_axes()
        frame_wdgt.min = 0; frame_wdgt.max = 0; frame_wdgt.value = 0
        state['frame_idx'] = 0
        state['fig'].canvas.draw_idle()
        status_wdgt.value = "Cleared"

    def _on_end_change(change):
        if change['name'] != 'value':
            return
        if float(start_time_wdgt.value) > float(end_time_wdgt.value):
            start_time_wdgt.value = float(end_time_wdgt.value)
        _update_window_bounds()
        if float(change['new']) > state['sim_T']:
            _run(reason='end_change')
        else:
            _refresh_frame_max(reset=False)
            _update_frame(frame_wdgt.value)

    def _on_start_change(change):
        if change['name'] != 'value':
            return
        if float(start_time_wdgt.value) > float(end_time_wdgt.value):
            start_time_wdgt.value = float(end_time_wdgt.value)
        _update_window_bounds()
        _refresh_frame_max(reset=False)
        _update_frame(frame_wdgt.value)

    def _on_frame_or_window(change):
        if change['name'] != 'value':
            return
        _update_frame(frame_wdgt.value)

    def _on_bounds_toggle(change):
        if change['name'] != 'value':
            return
        _refresh_param_range_defaults()
        _sync_mode()

    mode_wdgt.observe(_sync_mode, names='value')
    tabs.observe(_sync_tabs, names='selected_index')
    ic_same_wdgt.observe(lambda *_: _sync_mode(), names='value')
    param_use_bounds_wdgt.observe(_on_bounds_toggle, names='value')
    param_name_wdgt.observe(_refresh_param_range_defaults, names='value')
    param_span_wdgt.observe(_refresh_param_range_defaults, names='value')
    start_time_wdgt.observe(_on_start_change, names='value')
    end_time_wdgt.observe(_on_end_change, names='value')
    window_wdgt.observe(_on_frame_or_window, names='value')
    frame_wdgt.observe(_on_frame_or_window, names='value')
    recurrence_tol_wdgt.observe(_on_frame_or_window, names='value')
    run_btn.on_click(lambda _: _run(reason='manual'))
    clear_btn.on_click(_clear)

    _refresh_param_range_defaults()
    _sync_mode()
    _update_window_bounds()
    display(widgets.VBox([
        mode_wdgt,
        widgets.HBox([start_time_wdgt, end_time_wdgt, window_wdgt]),
        widgets.HBox([frame_wdgt, recurrence_tol_wdgt]),
        tabs,
        widgets.HBox([run_btn, clear_btn]),
        status_wdgt,
        out
    ]))
    _run(reason='manual')


def launch_portrait_comparison_grid(
    system,
    pars=(0.2, 0.2, 5.7),
    u0=(10, 10, 10),
    n_members=4,
    T=200, 
    alpha=.5
):
    pars = tuple(pars)
    u0 = np.array(u0, dtype=float)
    n_members = max(1, min(int(n_members), 4))

    state = {
        'run_id': 0,
        'sim_T': float(T),
        'trajs': [],
        'artists': [],
        'fig': None,
        'ax_ts': [],
        'ax_3d': [],
        'azim': 45.0
    }

    mode_wdgt = widgets.ToggleButtons(options=[('IC', 'ic'), ('Param', 'param')], value='ic', description='Mode:')
    start_time_wdgt = widgets.BoundedFloatText(description='Start T:', value=0.0, min=0.0, max=float(T), step=10.0)
    end_time_wdgt = widgets.BoundedFloatText(description='End T:', value=float(T), min=1.0, max=5000.0, step=10.0)
    window_wdgt = widgets.FloatSlider(description='Window T:', value=max(5.0, T / 4), min=1.0, max=float(T), step=1.0)
    frame_wdgt = widgets.IntSlider(description='Frame:', value=0, min=0, max=int(100 * T), step=max(1, int(T / 100)), layout=widgets.Layout(width='100%'))
    rotate_wdgt = widgets.FloatSlider(description='Rotate:', value=45.0, min=-180.0, max=180.0, step=1.0)
    run_btn = widgets.Button(description='Run / Reset', button_style='primary')
    clear_btn = widgets.Button(description='Clear Plots', button_style='warning')
    status_wdgt = widgets.HTML(value='Ready')
    out = widgets.Output()

    ic_same_wdgt = widgets.Checkbox(value=True, description='Same IC (jitter=0)')
    jitter_wdgt = widgets.FloatSlider(description='Jitter +/-:', value=0.0, min=0.0, max=10.0, step=0.1, readout_format='.1f')
    param_name_wdgt = widgets.Dropdown(options=[('a', 0), ('b', 1), ('c', 2)], value=2, description='Param:')
    param_span_wdgt = widgets.FloatSlider(description='Param +/-:', value=0.1, min=0.0, max=2.0, step=0.01, readout_format='.2f')
    param_use_bounds_wdgt = widgets.Checkbox(value=False, description='Use min/max range')
    param_min_wdgt = widgets.FloatText(description='Param min:', value=float(pars[param_name_wdgt.value]) - float(param_span_wdgt.value))
    param_max_wdgt = widgets.FloatText(description='Param max:', value=float(pars[param_name_wdgt.value]) + float(param_span_wdgt.value))

    def _refresh_param_range_defaults(*_):
        if param_use_bounds_wdgt.value:
            return
        center = float(pars[param_name_wdgt.value])
        span = float(param_span_wdgt.value)
        param_min_wdgt.value = center - span
        param_max_wdgt.value = center + span

    def _sync_mode(*_):
        if mode_wdgt.value == 'param':
            ic_same_wdgt.value = True
            ic_same_wdgt.disabled = True
            jitter_wdgt.disabled = True
            param_name_wdgt.disabled = False
            param_use_bounds_wdgt.disabled = False
            param_span_wdgt.disabled = param_use_bounds_wdgt.value
            param_min_wdgt.disabled = not param_use_bounds_wdgt.value
            param_max_wdgt.disabled = not param_use_bounds_wdgt.value
        else:
            ic_same_wdgt.disabled = False
            jitter_wdgt.disabled = ic_same_wdgt.value
            param_name_wdgt.disabled = True
            param_span_wdgt.disabled = True
            param_use_bounds_wdgt.disabled = True
            param_min_wdgt.disabled = True
            param_max_wdgt.disabled = True

    def _update_bounds(*_):
        start_time_wdgt.max = float(end_time_wdgt.value)
        if start_time_wdgt.value > start_time_wdgt.max:
            start_time_wdgt.value = start_time_wdgt.max
        window_wdgt.max = max(1.0, float(end_time_wdgt.value) - float(start_time_wdgt.value))
        if window_wdgt.value > window_wdgt.max:
            window_wdgt.value = window_wdgt.max

    def _ensure_figure():
        if state['fig'] is not None:
            return
        with out:
            fig = plt.figure(figsize=(3 * n_members, 5))
            gs = fig.add_gridspec(2, n_members, left=0.05, right=0.98, top=0.93, bottom=0.08, wspace=0.35, hspace=0.35, height_ratios=[1,3])
            state['ax_ts'] = [fig.add_subplot(gs[0, i]) for i in range(n_members)]
            state['ax_3d'] = [fig.add_subplot(gs[1, i], projection='3d') for i in range(n_members)]
            state['fig'] = fig
            display(fig)

    def _compute_trajs(T_local):
        trajs = []
        t = np.linspace(0, T_local, int(100 * T_local))
        ic_cmaps = ['Blues', 'Greens', 'Oranges', 'Reds']

        if mode_wdgt.value == 'param':
            if param_use_bounds_wdgt.value:
                pmin = float(param_min_wdgt.value)
                pmax = float(param_max_wdgt.value)
                if pmax < pmin:
                    pmin, pmax = pmax, pmin
                pvals = np.linspace(pmin, pmax, n_members)
            else:
                center = float(pars[param_name_wdgt.value])
                offsets = np.linspace(-param_span_wdgt.value, param_span_wdgt.value, n_members)
                pvals = center + offsets
            cmap = plt.get_cmap('Blues')
            shade = np.linspace(0.35, 0.95, n_members)
            for i, pval in enumerate(pvals):
                pars_i = list(pars)
                pars_i[param_name_wdgt.value] = float(pval)
                pars_i = tuple(pars_i)
                soln = solve_ivp(system, [0, T_local], u0.copy(), t_eval=t, args=pars_i)
                trajs.append({
                    'x_val': soln.y[0], 'y_val': soln.y[1], 'z_val': soln.y[2], 't_val': soln.t,
                    'color': cmap(shade[i]),
                    'label': f"p={float(pval):.3f}"
                })
        else:
            jitter_amp = 0.0 if ic_same_wdgt.value else float(jitter_wdgt.value)
            for i in range(n_members):
                u0_i = u0.copy() if jitter_amp == 0.0 else (u0 + np.random.uniform(-jitter_amp, jitter_amp, 3))
                soln = solve_ivp(system, [0, T_local], u0_i, t_eval=t, args=pars)
                cmap = plt.get_cmap(ic_cmaps[i % len(ic_cmaps)])
                trajs.append({
                    'x_val': soln.y[0], 'y_val': soln.y[1], 'z_val': soln.y[2], 't_val': soln.t,
                    'color': cmap(0.7),
                    'label': f"IC {i+1}"
                })
        return trajs

    def _clear_axes():
        for a in state['ax_ts'] + state['ax_3d']:
            a.clear()
        state['artists'] = []

    def _init_artists():
        _clear_axes()
        if len(state['trajs']) == 0:
            return

        xlim = [np.floor(min(np.min(tr['x_val']) for tr in state['trajs'])), np.ceil(max(np.max(tr['x_val']) for tr in state['trajs']))]
        ylim = [np.floor(min(np.min(tr['y_val']) for tr in state['trajs'])), np.ceil(max(np.max(tr['y_val']) for tr in state['trajs']))]
        zlim = [np.floor(min(np.min(tr['z_val']) for tr in state['trajs'])), np.ceil(max(np.max(tr['z_val']) for tr in state['trajs']))]

        for i, tr in enumerate(state['trajs']):
            ax_t = state['ax_ts'][i]
            ax_3 = state['ax_3d'][i]
            lx, = ax_t.plot(tr['t_val'][0:1], tr['x_val'][0:1], color=tr['color'], lw=1.2, label='x(t)')
            ly, = ax_t.plot(tr['t_val'][0:1], tr['y_val'][0:1], color=tr['color'], lw=1.0, ls='--', alpha=alpha, label='y(t)')
            px, = ax_t.plot(tr['t_val'][0:1], tr['x_val'][0:1], marker='o', color=tr['color'])
            l3, = ax_3.plot(tr['x_val'][0:1], tr['y_val'][0:1], tr['z_val'][0:1], color=tr['color'], lw=1.0)
            p3, = ax_3.plot(tr['x_val'][0:1], tr['y_val'][0:1], tr['z_val'][0:1], marker='o', color=tr['color'])
            ax_t.set_title(tr['label'])
            ax_t.set_xlabel('t')
            ax_t.set_ylabel('x,y')
            ax_t.set_ylim((min(xlim+ylim), max(xlim+ylim)))
            ax_t.legend(loc='upper right', fontsize=7, frameon=False)

            ax_3.set_xlim(xlim); ax_3.set_ylim(ylim); ax_3.set_zlim(zlim)
            ax_3.set_xlabel('x'); ax_3.set_ylabel('y'); ax_3.set_zlabel('z')
            state['artists'].append({'tr': tr, 'lx': lx, 'ly': ly, 'px': px, 'l3': l3, 'p3': p3, 'ax_t': ax_t, 'ax_3': ax_3})

    def _refresh_frame_bounds(reset=False):
        if len(state['trajs']) == 0:
            frame_wdgt.min = 0; frame_wdgt.max = 0; frame_wdgt.value = 0
            return
        npts = len(state['trajs'][0]['t_val'])
        sp = max(1.0, npts / max(state['sim_T'], 1e-6))
        sidx = min(npts - 1, int(float(start_time_wdgt.value) * sp))
        eidx = min(npts - 1, int(float(end_time_wdgt.value) * sp))
        frame_wdgt.min = max(0, sidx)
        frame_wdgt.max = max(frame_wdgt.min, eidx)
        frame_wdgt.value = frame_wdgt.min if reset else min(max(frame_wdgt.value, frame_wdgt.min), frame_wdgt.max)

    def _update_frame(idx):
        if len(state['trajs']) == 0 or state['fig'] is None:
            return
        start_T = float(start_time_wdgt.value)
        end_T = float(end_time_wdgt.value)
        window_T = float(window_wdgt.value)
        state['azim'] = float(rotate_wdgt.value)

        for item in state['artists']:
            tr = item['tr']
            npts = len(tr['t_val'])
            sp = max(1.0, npts / max(state['sim_T'], 1e-6))
            sidx = min(npts - 1, int(start_T * sp))
            eidx = min(npts - 1, int(end_T * sp))
            j = min(max(int(idx), sidx), eidx)
            hw = max(1, int((window_T * sp) / 2.0))
            li = max(sidx, j - hw)
            ri = min(eidx, j + hw)

            item['lx'].set_data(tr['t_val'][sidx:j+1], tr['x_val'][sidx:j+1])
            item['ly'].set_data(tr['t_val'][sidx:j+1], tr['y_val'][sidx:j+1])
            item['px'].set_data(tr['t_val'][j:j+1], tr['x_val'][j:j+1])
            item['ax_t'].set_xlim([tr['t_val'][li], max(tr['t_val'][ri], tr['t_val'][li] + 1e-6)])

            item['l3'].set_data(tr['x_val'][sidx:j+1], tr['y_val'][sidx:j+1])
            item['l3'].set_3d_properties(tr['z_val'][sidx:j+1])
            item['p3'].set_data(tr['x_val'][j:j+1], tr['y_val'][j:j+1])
            item['p3'].set_3d_properties(tr['z_val'][j:j+1])
            item['ax_3'].view_init(30, state['azim'])

        state['fig'].canvas.draw_idle()

    def _run(reason='manual'):
        _ensure_figure()
        state['run_id'] += 1
        state['sim_T'] = float(end_time_wdgt.value)
        state['trajs'] = _compute_trajs(state['sim_T'])
        _init_artists()
        _refresh_frame_bounds(reset=(reason == 'manual'))
        _update_frame(frame_wdgt.value)
        status_wdgt.value = f"Run #{state['run_id']}: mode={mode_wdgt.value}, members={n_members}, Start T={float(start_time_wdgt.value):.1f}, End T={state['sim_T']:.1f}"

    def _clear(_=None):
        _ensure_figure()
        state['trajs'] = []
        _clear_axes()
        frame_wdgt.min = 0; frame_wdgt.max = 0; frame_wdgt.value = 0
        state['fig'].canvas.draw_idle()
        status_wdgt.value = "Cleared"

    def _on_time_change(change):
        if change['name'] != 'value':
            return
        _update_bounds()
        if float(end_time_wdgt.value) > state['sim_T']:
            _run(reason='end_change')
        else:
            _refresh_frame_bounds(reset=False)
            _update_frame(frame_wdgt.value)

    def _on_frame_change(change):
        if change['name'] != 'value':
            return
        _update_frame(frame_wdgt.value)

    def _on_bounds_toggle(change):
        if change['name'] != 'value':
            return
        _refresh_param_range_defaults()
        _sync_mode()

    mode_wdgt.observe(_sync_mode, names='value')
    ic_same_wdgt.observe(lambda *_: _sync_mode(), names='value')
    param_use_bounds_wdgt.observe(_on_bounds_toggle, names='value')
    param_name_wdgt.observe(_refresh_param_range_defaults, names='value')
    param_span_wdgt.observe(_refresh_param_range_defaults, names='value')
    start_time_wdgt.observe(_on_time_change, names='value')
    end_time_wdgt.observe(_on_time_change, names='value')
    window_wdgt.observe(_on_frame_change, names='value')
    frame_wdgt.observe(_on_frame_change, names='value')
    rotate_wdgt.observe(_on_frame_change, names='value')
    run_btn.on_click(lambda _: _run(reason='manual'))
    clear_btn.on_click(_clear)

    _refresh_param_range_defaults()
    _sync_mode()
    _update_bounds()
    display(widgets.VBox([
        mode_wdgt,
        widgets.HBox([start_time_wdgt, end_time_wdgt, window_wdgt]),
        widgets.HBox([frame_wdgt, rotate_wdgt]),
        widgets.HBox([ic_same_wdgt, jitter_wdgt]),
        widgets.HBox([param_name_wdgt, param_span_wdgt, param_use_bounds_wdgt]),
        widgets.HBox([param_min_wdgt, param_max_wdgt]),
        widgets.HBox([run_btn, clear_btn]),
        status_wdgt,
        out
    ]))
    _run(reason='manual')


# def plot_portrait(data, plots=['xyz'], cols=None, orientation='h', vstretch= 2, hstretch=1, freeze_motion=False):
#     pal = sns.color_palette('tab20')
#     T = data.pop('T', 'No Key found')
#     max_time_int = data.pop('max_time_int', T * 4)
#
#     lim_d = {}
#     for axis in ['x_val', 'y_val', 'z_val', 't_val']:
#         lim_d[axis] = [np.floor(min([
#                             min([min(traj[axis]) for traj in data[c]['traj']]) for c in data.keys()])),
#                        np.ceil(max([
#                            max([max(traj[axis]) for traj in data[c]['traj']]) for c in data.keys()]))
#         ]
#
#     if orientation == 'h':
#         if cols == None:
#             cols = len(plots)+1
#         fig = plt.figure(figsize=(13,7*len(data)))
#         gs_cols = hstretch*cols
#         start_subplot_col = gs_cols-hstretch*(cols-(len(plots)-1))
#
#         gs_rows = vstretch*(len(plots)-1)
#         start_subplot_row = 0
#
#         start_main_col = 0
#         end_main_col = start_subplot_col
#         start_main_row = 0
#         end_main_row = start_subplot_row+vstretch*(len(plots)-1)
#
#     if orientation == 'v':
#         if cols == None:
#             cols = 5
#         if len(plots) > 1:
#             fig = plt.figure(figsize=(7,7+2*(len(plots)-1)))
#         elif len(plots):
#             fig = plt.figure(figsize=(7,7))
#         gs_cols = hstretch*cols
#         start_subplot_col = 0
#         gs_rows = gs_cols+ vstretch*(len(plots)-1)
#         start_subplot_row = gs_cols
#
#         start_main_col = 0
#         end_main_col = gs_cols
#         start_main_row = 0
#         end_main_row = start_subplot_row
#
#     gs_full = gridspec.GridSpec(len(data), 1, figure=fig, hspace=1.25)
#     for ic, c in enumerate(data.keys()):
#         gs = gs_full[ic].subgridspec(gs_rows,gs_cols, left=0.1, right=0.9, top = .95, bottom=.1,wspace=.65, hspace=1.25)
#         # gs = fig.add_gridspec(gs_rows,gs_cols, left=0.1, right=0.9, top = .95, bottom=.1,
#         #                       wspace=.65, hspace=1.25)
#         if 'xyz' in plots:
#             main_gs = gs[start_main_row:end_main_row,start_main_col:end_main_col]
#             ax = fig.add_subplot(main_gs,projection='3d')
#
#         if 'xt' in plots:
#             xt_gs =gs[start_subplot_row:start_subplot_row+1*vstretch, start_subplot_col:]
#             ax_xt = fig.add_subplot(xt_gs)
#         if 'yt' in plots:
#             yt_gs =gs[start_subplot_row+1*vstretch:start_subplot_row+2*vstretch, start_subplot_col:]
#             ax_yt = fig.add_subplot(yt_gs)
#         if 'xz' in plots:
#             xz_gs =gs[start_subplot_row+2*vstretch:start_subplot_row+3*vstretch, start_subplot_col:]
#             ax_xz = fig.add_subplot(xz_gs)
#
#         plot_data={}
#         for ip, c in enumerate(data.keys()):
#             trajs =[]
#             for im, trj in enumerate(data[c]['traj']):
#                 color = pal[im]
#                 x_val = trj['x_val']
#                 y_val = trj['y_val']
#                 z_val = trj['z_val']
#                 t_val = trj['z_val']
#                 plot_data_tmp = {'color':color}
#                 if 'xyz' in plots:
#                     point, = ax.plot(x_val[0], y_val[0], z_val[0], marker='o', color=color, label= str(c))
#                     line, = ax.plot(x_val[0], y_val[0], z_val[0], linewidth=0.75, alpha=0.685, color=color)
#                     plot_data_tmp.update({'line':line, 'point':point})
#         #         if len(plots) >1:
#                 if 'xt' in plots:
#                     line_xt, = ax_xt.plot(t_val[0], x_val[0],lw=1, color=color)
#                     point_xt, = ax_xt.plot(t_val[0], x_val[0], marker='o', color=color, label=str(c))
#                     plot_data_tmp.update({'line_xt':line_xt, 'point_xt':point_xt})
#                 if 'yt' in plots:
#                     line_yt, = ax_yt.plot(t_val[0], y_val[0], lw=1, color=color)
#                     point_yt, = ax_yt.plot(t_val[0], y_val[0], marker='o', color=color, label=str(c))
#                     plot_data_tmp.update({'line_yt':line_yt, 'point_yt':point_yt})
#                 if 'xz' in plots:
#                     line_xz, = ax_xz.plot(x_val[0], z_val[0], lw=1, color=color)
#                     point_xz, = ax_xz.plot(x_val[0], z_val[0], marker='o', color=color, label=str(c))
#                     plot_data_tmp.update({'line_xz':line_xz, 'point_xz':point_xz})
#
#                 trj.update(plot_data_tmp)
#                 trajs.append(trj)
#
#     #         plot_data_tmp = {'color':color, 'line':line, 'point':point,
#     #                          'line_xt':line_xt, 'point_xt':point_xt,
#     #                         'line_yt':line_yt, 'point_yt':point_yt,
#     #                         'line_xz':line_xz, 'point_xz':point_xz}
#             d3 = data[c].copy()
#             d3.update({'traj': trajs})#plot_data_tmp)
#             plot_data[c]= d3#{'traj': trajs}#d3#dict(plot_data_tmp, data[c])
#
#             if 'xyz' in plots:
#                 ax.set_ylim(lim_d['y_val']) # this shouldn't be necessary but the limits are usually enlarged per defailt
#                 ax.set_zlim(lim_d['z_val']) # this shouldn't be necessary but the limits are usually enlarged per defailt
#                 ax.set_xlim(lim_d['x_val'])
#
#                 ax.set_title(plot_data[c]['ttl'])
#                 ax.set_xlabel(r'$x$',fontweight='bold')
#                 ax.set_ylabel(r'$y$',fontweight='bold')
#                 ax.set_zlabel(r'$z$',fontweight='bold')
#
#             if 'xt' in plots:
#                 # ax_xt.set_xlim([lim_d['t_val'][0],lim_d['t_val'][1]/(2*T)])
#                 ax_xt.set_ylim(lim_d['x_val'])
#                 ax_xt.set_ylabel(r'$x(t)$')
#                 ax_xt.set_xlabel(r'$t$')
#             #     ax_xt.set_title('X with respect to t')
#             if 'yt' in plots:
#                 # ax_yt.set_xlim([lim_d['t_val'][0],lim_d['t_val'][1]/(2*T)])
#                 ax_yt.set_ylim(lim_d['y_val'])
#                 ax_yt.set_ylabel(r'$y(t)$')
#                 ax_yt.set_xlabel(r'$t$')
#             #     ax_yt.set_title('Y with respect to t')
#             if 'xz' in plots:
#                 ax_xz.set_xlim(lim_d['x_val'])
#                 ax_xz.set_ylim(lim_d['z_val'])
#                 ax_xz.set_ylabel(r'$z(t)$')
#                 ax_xz.set_xlabel(r'$x(t)$')
#         #     ax_xz.set_title('Z with respect to X')
#             if orientation == 'v':
#                 ax.legend(title='c value', bbox_to_anchor=(.15,1))
#             else:
#                 ax.legend(title='c value', bbox_to_anchor=(.15,1))
#
#
#     def update(num):
#         for c in plot_data.keys():
#             for traj in plot_data[c]['traj']:
#                 min_t = max([lim_d['t_val'][0], traj['t_val'][num] - 2 * T])
#                 if 'xyz' in plots:
#                     traj['line'].set_data(traj['x_val'][:num], traj['y_val'][:num])
#                     traj['line'].set_3d_properties(traj['z_val'][:num])
#                     traj['point'].set_data(traj['x_val'][num-1:num], traj['y_val'][num-1:num])
#                     traj['point'].set_3d_properties(traj['z_val'][num-1:num])
#                 if 'xt' in plots:
#                     traj['line_xt'].set_data(traj['t_val'][:num], traj['x_val'][:num])
#                     traj['point_xt'].set_data(traj['t_val'][num], traj['x_val'][num])
#                     min_t = max([lim_d['t_val'][0],traj['t_val'][num]-2*T])
#                     ax_xt.set_xlim([min_t, min_t+.1*T])
#                 if 'yt' in plots:
#                     traj['line_yt'].set_data(traj['t_val'][:num], traj['y_val'][:num])
#                     traj['point_yt'].set_data(traj['t_val'][num], traj['y_val'][num])
#                     ax_yt.set_xlim([min_t, min_t + .1 * T])
#                 if 'xz' in plots:
#                     traj['line_xz'].set_data(traj['x_val'][:num], traj['z_val'][:num])
#                     traj['point_xz'].set_data(traj['x_val'][num], traj['z_val'][num])
#                 if not freeze_motion:
#                     ax.view_init(30, 0.3 * num)
#                 fig.canvas.draw_idle()
#
#
#     time_wdgt = widgets.IntSlider(
#         description='Test widget:',
#         value=0,
#         min=0, max=max_time_int, step=T/100,
#         layout=widgets.Layout(width='100%'))
#
#
#     interact(update, num=time_wdgt);


def _simulate_ensemble(system, pars, u0, T=200, dt=0.02, n_ic=6, jitter=(-0.5, 0.5, 3)):
    """Integrate a small ensemble for one parameter set and return trajectories."""
    u0 = np.array(u0, dtype=float)
    t_eval = np.arange(0.0, T + dt, dt)
    trajs = []
    for _ in range(int(n_ic)):
        u_init = u0 + np.random.uniform(*jitter)
        sol = solve_ivp(system, [0.0, T], u_init, t_eval=t_eval, args=tuple(pars))
        trajs.append(sol.y)
    return t_eval, np.array(trajs)


def _recurrence_intervals_1d(t, series, ref_value, tol, min_separation):
    """Return recurrence intervals for returns into a tolerance band around ref_value."""
    inside = np.abs(series - ref_value) <= tol
    # Event is entry into the band (outside -> inside transition).
    entry_idx = np.where(inside & ~np.r_[False, inside[:-1]])[0]
    if entry_idx.size == 0:
        return np.array([]), np.array([], dtype=int)

    kept = [int(entry_idx[0])]
    for idx in entry_idx[1:]:
        if (t[idx] - t[kept[-1]]) >= min_separation:
            kept.append(int(idx))
    kept = np.array(kept, dtype=int)
    if kept.size < 2:
        return np.array([]), kept
    return np.diff(t[kept]), kept


def plot_recurrence_timeseries_pairs(
    system,
    pars,
    u0=(10, 10, 10),
    T=200,
    dt=0.02,
    transient_frac=0.2,
    tol_frac=0.05,
    min_separation=0.5,
    bins=24,
    use_seaborn=True,
    figsize=(7.2, 4.8),
):
    """
    Plot paired views for each state variable:
      - left: time series with recurrence-band hits
      - right: recurrence-interval distribution

    Works for either Rössler or Lorenz (or other 3D systems with solve_ivp signature).
    """
    t_eval = np.arange(0.0, T + dt, dt)
    sol = solve_ivp(system, [0.0, T], np.array(u0, dtype=float), t_eval=t_eval, args=tuple(pars))
    t = sol.t
    states = sol.y
    labels = ("x", "y", "z")
    cut = min(len(t) - 1, max(0, int(transient_frac * len(t))))

    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex='col')

    results = {"t": t, "states": states, "intervals": {}, "event_times": {}, "ref": {}, "tol": {}}

    for i, lbl in enumerate(labels):
        ax_ts = axs[i, 0]
        ax_hist = axs[i, 1]

        s = states[i]
        s_use = s[cut:]
        t_use = t[cut:]
        if len(s_use) == 0:
            s_use = s
            t_use = t

        ref = float(np.median(s_use))
        span = float(np.max(s_use) - np.min(s_use))
        tol = max(1e-10, tol_frac * span)

        intervals, event_idx = _recurrence_intervals_1d(
            t=t_use,
            series=s_use,
            ref_value=ref,
            tol=tol,
            min_separation=min_separation
        )
        event_times = t_use[event_idx] if event_idx.size else np.array([])

        ax_ts.plot(t_use, s_use, lw=1.0, color=f"C{i}")
        ax_ts.axhline(ref, color="0.25", ls="--", lw=0.9)
        ax_ts.fill_between(t_use, ref - tol, ref + tol, color="0.6", alpha=0.18)
        if event_times.size:
            event_vals = np.interp(event_times, t_use, s_use)
            ax_ts.scatter(event_times, event_vals, s=10, color="black", alpha=0.6, label="recurrence")
        ax_ts.set_ylabel(lbl)
        ax_ts.set_title(f"{lbl}(t) and recurrence band", fontsize=10)
        ax_ts.grid(True, alpha=0.25)

        if intervals.size:
            if use_seaborn:
                sns.histplot(intervals, bins=bins, kde=True, stat="density", ax=ax_hist, color=f"C{i}")
            else:
                ax_hist.hist(intervals, bins=bins, density=True, alpha=0.75, color=f"C{i}")
            ax_hist.axvline(float(np.median(intervals)), color="0.2", ls="--", lw=0.9)
            ax_hist.set_title(f"{lbl} recurrence intervals", fontsize=10)
        else:
            ax_hist.text(0.5, 0.5, "Not enough recurrences", ha="center", va="center", transform=ax_hist.transAxes)
            ax_hist.set_title(f"{lbl} recurrence intervals", fontsize=10)
        ax_hist.grid(True, alpha=0.25)

        results["intervals"][lbl] = intervals
        results["event_times"][lbl] = event_times
        results["ref"][lbl] = ref
        results["tol"][lbl] = tol

    axs[2, 0].set_xlabel("time")
    axs[2, 1].set_xlabel("recurrence interval")
    fig.suptitle(f"Recurrence timing by variable | pars={tuple(np.round(pars, 4))}", y=0.995)
    fig.tight_layout()
    return fig, axs, results


def summarize_recurrence_over_parameters(
    system,
    parameter_list,
    u0=(10, 10, 10),
    T=200,
    dt=0.02,
    transient_frac=0.2,
    tol_frac=0.05,
    min_separation=0.5,
    bins=24,
    use_seaborn=True,
    make_plots=False,
    figsize=(12, 9),
):
    """
    Run recurrence diagnostics over a parameter list.

    Returns a compact list of dictionaries with median recurrence intervals
    (x, y, z) and number of detected intervals per variable.
    """
    summary = []
    outputs = []

    for idx, pars in enumerate(parameter_list):
        fig, axs, rec = plot_recurrence_timeseries_pairs(
            system=system,
            pars=pars,
            u0=u0,
            T=T,
            dt=dt,
            transient_frac=transient_frac,
            tol_frac=tol_frac,
            min_separation=min_separation,
            bins=bins,
            use_seaborn=use_seaborn,
            figsize=figsize,
        )

        if not make_plots:
            plt.close(fig)

        row = {"index": idx, "pars": tuple(pars)}
        for lbl in ("x", "y", "z"):
            vals = rec["intervals"].get(lbl, np.array([]))
            row[f"{lbl}_median"] = float(np.median(vals)) if vals.size else np.nan
            row[f"{lbl}_n"] = int(vals.size)
        summary.append(row)
        outputs.append({"pars": tuple(pars), "results": rec, "fig": fig if make_plots else None, "axs": axs if make_plots else None})

    return summary, outputs


def plot_cross_section_pairs_grid(
    system,
    parameter_list,
    u0=(10, 10, 10),
    T=200,
    dt=0.02,
    n_ic=6,
    jitter=(-0.5, 0.5, 3),
    pairs=((0, 1), (0, 2)),
    labels=("x", "y", "z"),
    transient_frac=0.25,
    figsize=None,
    alpha=0.45,
):
    """Plot static 2D cross-sections for each parameter set and IC ensemble member."""
    parameter_list = list(parameter_list)
    n_rows = len(parameter_list)
    n_cols = len(pairs)
    if figsize is None:
        figsize = (4.8 * n_cols, 2.8 * max(n_rows, 1))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, pars in enumerate(parameter_list):
        _, trajs = _simulate_ensemble(
            system=system,
            pars=pars,
            u0=u0,
            T=T,
            dt=dt,
            n_ic=n_ic,
            jitter=jitter,
        )
        cut = int(transient_frac * trajs.shape[-1])
        for j, (a, b) in enumerate(pairs):
            ax = axs[i, j]
            for k in range(trajs.shape[0]):
                ax.plot(trajs[k, a, cut:], trajs[k, b, cut:], lw=0.8, alpha=alpha)
            ax.set_xlabel(labels[a])
            ax.set_ylabel(labels[b])
            if j == 0:
                ax.set_title(f"pars={tuple(np.round(pars, 4))}")
            else:
                ax.set_title(f"{labels[a]}-{labels[b]}")
            ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axs


def estimate_largest_lyapunov(
    system,
    pars,
    u0=(10, 10, 10),
    T=120,
    dt=0.02,
    delta0=1e-7,
    fit_frac=(0.1, 0.7),
):
    """Estimate largest Lyapunov exponent from two nearby trajectories."""
    u0 = np.array(u0, dtype=float)
    dvec = np.array([delta0, 0.0, 0.0], dtype=float)
    t_eval = np.arange(0.0, T + dt, dt)

    sol_ref = solve_ivp(system, [0.0, T], u0, t_eval=t_eval, args=tuple(pars))
    sol_pert = solve_ivp(system, [0.0, T], u0 + dvec, t_eval=t_eval, args=tuple(pars))

    dist = np.linalg.norm(sol_pert.y - sol_ref.y, axis=0)
    dist = np.maximum(dist, 1e-16)
    log_growth = np.log(dist / max(delta0, 1e-16))

    n = len(t_eval)
    i0 = max(1, int(fit_frac[0] * n))
    i1 = max(i0 + 3, int(fit_frac[1] * n))
    slope, _ = np.polyfit(t_eval[i0:i1], log_growth[i0:i1], deg=1)
    return float(slope)


def compute_max_lyapunov_grid(
    system,
    parameter_list,
    u0=(10, 10, 10),
    T=120,
    dt=0.02,
    delta0=1e-7,
):
    """Compute largest Lyapunov exponent estimates for each parameter set."""
    vals = []
    for pars in parameter_list:
        vals.append(
            estimate_largest_lyapunov(
                system=system,
                pars=pars,
                u0=u0,
                T=T,
                dt=dt,
                delta0=delta0,
            )
        )
    return np.array(vals)


def compute_max_lyapunov_param_sweep(
    system,
    base_pars,
    sweep_index,
    sweep_values=None,
    sweep_min=None,
    sweep_max=None,
    n_sweep=20,
    u0=(10, 10, 10),
    T=120,
    dt=0.02,
    delta0=1e-7,
):
    """
    Student-friendly wrapper: vary one parameter while others stay fixed.

    Returns:
      sweep_values: array of scanned values for selected parameter index
      lambda_max: largest Lyapunov estimate at each sweep value
      parameter_list: full parameter tuples used internally
    """
    base_pars = tuple(base_pars)
    if len(base_pars) != 3:
        raise ValueError("base_pars must be length-3.")
    sweep_index = int(sweep_index)
    if sweep_index not in (0, 1, 2):
        raise ValueError("sweep_index must be 0, 1, or 2.")

    if sweep_values is None:
        if sweep_min is None or sweep_max is None:
            raise ValueError("Provide either sweep_values or both sweep_min and sweep_max.")
        sweep_values = np.linspace(float(sweep_min), float(sweep_max), int(n_sweep))
    else:
        sweep_values = np.array(sweep_values, dtype=float)

    parameter_list = []
    for val in sweep_values:
        p = list(base_pars)
        p[sweep_index] = float(val)
        parameter_list.append(tuple(p))

    lambda_max = compute_max_lyapunov_grid(
        system=system,
        parameter_list=parameter_list,
        u0=u0,
        T=T,
        dt=dt,
        delta0=delta0,
    )
    return sweep_values, lambda_max, parameter_list


def plot_max_lyapunov_param_sweep(
    system,
    base_pars,
    sweep_index,
    sweep_values=None,
    sweep_min=None,
    sweep_max=None,
    n_sweep=20,
    u0=(10, 10, 10),
    T=120,
    dt=0.02,
    delta0=1e-7,
    xlabel=None,
    title=None,
):
    """
    Convenience plot for one-parameter largest-Lyapunov sweeps.
    """
    sweep_values, lambda_max, parameter_list = compute_max_lyapunov_param_sweep(
        system=system,
        base_pars=base_pars,
        sweep_index=sweep_index,
        sweep_values=sweep_values,
        sweep_min=sweep_min,
        sweep_max=sweep_max,
        n_sweep=n_sweep,
        u0=u0,
        T=T,
        dt=dt,
        delta0=delta0,
    )

    if xlabel is None:
        xlabel = f"parameter index {int(sweep_index)}"
    if title is None:
        title = "Largest Lyapunov estimate vs parameter sweep"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sweep_values, lambda_max, marker="o")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Estimated largest Lyapunov exponent")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig, ax, sweep_values, lambda_max, parameter_list
