import matplotlib.pyplot as plt
import numpy as np


def corner_fun(fun, x, plot_ranges):
    N_1d = 100
    N_2d = 100
    x = x.squeeze()
    D = np.size(x)
    for i in range(D):
        assert x[i] <= plot_ranges[i][1] and x[i] >= plot_ranges[i][0]

    K = D
    xs_all_1d = [
        np.linspace(plot_ranges[i][0], plot_ranges[i][1], N_1d)
        for i in range(D)
    ]
    xs_all_2d = [
        np.linspace(plot_ranges[i][0], plot_ranges[i][1], N_2d)
        for i in range(D)
    ]

    ys_all_1d = []
    for dim in range(D):
        xs = xs_all_1d[dim]
        points = np.tile(x, (N_1d, 1))
        points[:, dim] = xs
        ys = fun(points)
        ys_all_1d.append(ys)
    reverse = False
    fig = None
    # Some magic numbers for pretty axis layout.
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    new_fig = True
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    for i in range(K):
        for j in range(K):

            if reverse:
                ax = axes[K - i - 1, K - j - 1]
            else:
                ax = axes[i, j]
            if j > 0 and j != i:
                ax.set_yticklabels([])
            if i < K - 1:
                ax.set_xticklabels([])
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                ax.yaxis.tick_right()
                ax.plot(xs_all_1d[i], ys_all_1d[i])
                continue
            else:
                dim_1 = j
                dim_2 = i
                xs_1, xs_2 = np.meshgrid(xs_all_2d[dim_1], xs_all_2d[dim_2])
                points = np.tile(x, (np.size(xs_1), 1))
                points[:, dim_1] = xs_1.ravel()
                points[:, dim_2] = xs_2.ravel()
                ys = fun(points)
                ys = ys.reshape(N_2d, N_2d)
                im = ax.pcolormesh(xs_1, xs_2, ys)
    cbar_ax = fig.add_axes([1 - 1 / K, 0.5, 0.03, 1 / K])
    fig.colorbar(im, cax=cbar_ax)
    return fig
