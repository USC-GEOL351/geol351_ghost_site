# %matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import numpy as np
import matplotlib.cm as cm
import matplotlib
import numpy as np

label_d = {'NetFluxlist': 'Net Flux', 'SabsList': 'Incident solar', 'Tlist': 'Surface temperature',
           "L": 'Solar luminosity', 'OLRlist': 'Outgoing longwave'}
units_d = {'NetFluxlist': '($W/m^2$)', 'SabsList': '($W/m^2$)', 'OLRlist': '($W/m^2$)',
           'Tlist': '(K)', "L": '($W/m^2$)'}


def make_stability_bifurcation(ODE_func, deriv='NetFlux', var='T', param_name='L', param_range=None, colormap=None,
                               bifurcation=True, figsize=None, stability=True, bifurcation_title=None):
    var_name = '{}list'.format(var)
    deriv_name = '{}list'.format(deriv)
    if param_name == 'L':
        if param_range is None:
            param_range = [900, 3000]

    if figsize is None:
        figsize = (10, 8)

    plt.close('all')
    # Initialize global variables to store the state
    Ts = []  # To store temperatures of change
    Ls = []  # To store L values for each temperature of change
    colors = []  # To store colors for each point
    if colormap is None:
        colormap = cm.viridis  # Define a colormap

    minL = min(param_range)
    maxL = max(param_range)
    delL = maxL - minL

    # Create figure and axes
    fig = plt.figure(figsize=figsize)

    cols = bifurcation + stability
    # Adjust figure size or spacing if necessary
    plt.subplots_adjust(bottom=0.025, hspace=0.5, left=.2)  # Adjust bottom margin and spacing

    gs = fig.add_gridspec(3, cols, wspace=.5, height_ratios=[4, .15, .5])  # Adjust grid specs as needed

    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    ax1b.set_ylim(ax1.get_ylim())
    ax1b.set_yticks([])

    if stability is True:
        ax1.set_xlabel(' '.join([label_d[var_name], units_d[var_name]]))
        ax1.set_ylabel(' '.join([label_d[deriv_name], units_d[deriv_name]]))
        ax1.set_title('Stability Diagram')

    if bifurcation is True:
        if cols > 1:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2b = ax2.twinx()
            ax2b.set_ylim(ax2.get_ylim())
            ax2b.set_yticks([])
            ax2.set_ylabel(label_d[var_name])
            ax2.set_xlim([minL, maxL])
            ax2.set_xlabel(
                '{}/n{} {}'.format(' '.join([label_d[param_name], units_d[param_name]]), 'Click to select', param_name))

            # ax_slider = fig.add_subplot(gs[1, 1])
            # slider = Slider(ax=ax_slider, label='Select L', valmin=minL, valmax=maxL, valinit=minL)
            slider_vline = ax2b.axvline(x=minL, color='r', linestyle='--', zorder=100)
            if bifurcation_title is None:
                ax2.set_title('Bifurcation Diagram')
            else:
                ax2.set_title(bifurcation_title)
            ax2.set_xlabel("Click to select {}".format(param_name))

            def onclick_bif(event):
                if event.inaxes == ax2b:
                    val = event.xdata
                    process_and_plot(val)
                    slider_vline.set_xdata([val, val])

            fig.canvas.mpl_connect('button_press_event', onclick_bif)

        else:
            ax1.set_ylabel(' '.join([label_d[var_name], units_d[var_name]]))
            ax1.set_xlim([minL, maxL])
            ax1.set_xlabel(' '.join([label_d[param_name], units_d[param_name]]))


    else:
        ax_click = fig.add_subplot(gs[1, 0])  # Subplot for click detection
        ax_click.set_yticks([])

        # Initial plot on ax_click for click detection
        ax_click.plot([minL, maxL], [0, 0], 'k-', lw=1, zorder=-1)  # Draw a base line for click detection
        ax_click.set_xlabel("Click to select {}".format(param_name))

        def onclick(event):
            if event.inaxes == ax_click:
                color = colormap(event.xdata / delL)

                # Plot point on click detection plot
                ax_click.scatter([event.xdata], [0], color=color,
                                 label='{}={:.2f}'.format(param_name, event.xdata))  # Mark the click position
                ax_click.legend(bbox_to_anchor=(.95, -3.050), ncols=int(len(Ls) / 6) + 1)
                process_and_plot(event.xdata)

        fig.canvas.mpl_connect('button_press_event', onclick)

    # Function to process click and plot
    def process_and_plot(L):
        d = ODE_func(**{param_name: L})
        # Determine color
        color = colormap(L / delL)

        # Identify points of sign change
        sign_change_indexes = np.where(np.diff(np.sign(np.array(d[deriv_name]))))[0]
        temps_of_change = [np.mean([d[var_name][ik], d[var_name][ik + 1]]) for ik in sign_change_indexes]

        # Store data
        Ts.append(temps_of_change)
        Ls.append([L for _ in sign_change_indexes])
        colors.extend([color for _ in sign_change_indexes])

        ax1b.clear()

        if stability is True:
            # Plot on the left subplot
            ax1.plot(d[var_name], d[deriv_name], c=color)
            ax1.set_xlabel(' '.join([label_d[var_name], units_d[var_name]]))
            ax1.set_ylabel(' '.join([label_d[deriv_name], units_d[deriv_name]]))
            ax1b.scatter(temps_of_change, [0 for _ in temps_of_change], color=color, edgecolor='r')
            ax1b.axhline(y=0, lw=1, color='k', ls='--', label='Equilibrium')

        if bifurcation is True:
            if cols > 1:
                # Plot on the right subplot
                ax2.scatter([L for _ in sign_change_indexes], temps_of_change, color=color)
                # ax2.set_ylabel(label_d[var_name])
                ax2.set_xlim([minL, maxL])
                # ax2.set_xlabel(label_d[param_name])
                ax2.set_ylabel(' '.join([label_d[var_name], units_d[var_name]]))
                ax2.set_xlabel(' '.join([label_d[param_name], units_d[param_name]]))

                ax2b.clear()
                slider_vline = ax2b.axvline(x=L, color='r', linestyle='--', linewidth=1, zorder=-1)
                ax2b.set_ylim(ax2.get_ylim())
                ax2b.set_xlim(ax2.get_xlim())
                ax2b.scatter([L for _ in sign_change_indexes], temps_of_change, color=color, edgecolor='r')
                ax2b.set_yticks([])

            else:
                ax1.scatter([L for _ in sign_change_indexes], temps_of_change, color=color)
                ax1.set_xlim([minL, maxL])
                ax1.set_ylabel(' '.join([label_d[var_name], units_d[var_name]]))
                ax1.set_xlabel(' '.join([label_d[param_name], units_d[param_name]]))
                ax1b.scatter([L for _ in sign_change_indexes], temps_of_change, color=color, edgecolor='r')

        ax1b.set_xlim(ax1.get_xlim())
        ax1b.set_ylim(ax1.get_ylim())
        ax1b.set_yticks([])

        fig.canvas.draw()

    # Function to clear all plots and reset data
    def clear_plots(event):
        ax1.clear()
        ax1b.clear()
        if cols > 1:
            ax2.clear()
            ax2b.clear()
        ax_click.clear()
        ax_click.plot([minL, maxL], [0, 0], 'k-', lw=1, zorder=-1)  # Redraw base line for click detection
        ax_click.set_xlabel("Click to select {}".format(param_name))
        Ts.clear()
        Ls.clear()
        colors.clear()
        fig.canvas.draw()

    # Setup the clear button
    button_position = [.7, 0.95, 0.2, 0.025]  # Adjust these values as needed [left, bottom, width, height]
    clear_button_ax = fig.add_axes(button_position)
    button = Button(clear_button_ax, 'Clear Figures')
    button.on_clicked(clear_plots)


def plot_energy_balance(ode_func, param_name='L', y_var='SabsList', x_var='Tlist', x_label=None, y_label=None,
                        param_list=None, figsize=(5, 5)):
    plt.close()

    if param_list is not None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        for L in param_list:
            d = ode_func(**{param_name: L})
            ax.plot(d[x_var], d[y_var], label='{}, {}={}'.format(label_d[y_var], param_name, str(L)))

        ax.plot(d['Tlist'], d['OLRlist'], linestyle='--', alpha=.7, color='k', label='OLR')
        T_0 = d['Tlist'][np.argwhere(np.diff(np.array(d['aList'])) != 0)[-1][0] + 1]
        T_i = d['Tlist'][np.argwhere(np.diff(np.array(d['aList'])) != 0)[0][0]]

        ax.axvline(x=T_i, label=r'$T_i$', color='k', alpha=.3, ls=':')
        ax.axvline(x=T_0, label=r'$T_0$', color='k', alpha=.6, ls=':')
        ax.legend(bbox_to_anchor=(1, 1))

        if y_label is None:
            y_label = label_d[y_var]
        if x_label is None:
            x_label = label_d[x_var]

        ax.set_ylabel(' '.join([y_label, units_d[y_var]]))
        ax.set_xlabel(' '.join([x_label, units_d[x_var]]))
        ax.set_title('Energy Balance Diagram')

        return fig, ax
