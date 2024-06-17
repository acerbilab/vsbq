import matplotlib.pyplot as plt
from corner import corner


def corner_plot(gt_samples, algo_samples, title=None, txt=None, save_as=None):
    fig = corner(
        gt_samples,
        color="tab:orange",
        hist_kwargs={"density": True},
    )
    corner(
        algo_samples,
        fig=fig,
        color="tab:blue",
        contour_kwargs=dict(linestyles="dashed"),
        hist_kwargs={"density": True},
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)
    lgd = fig.legend(
        labels=["Ground truth", "VSBQ"],
        loc="upper center",
        bbox_to_anchor=(0.8, 0.8),
    )
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if txt is not None:
        text_art = fig.text(
            0.0, -0.10, txt, wrap=True, horizontalalignment="left", fontsize=12
        )
        extra_artists = (text_art, lgd)
    else:
        extra_artists = (lgd,)
    if save_as is not None:
        fig.savefig(
            save_as,
            dpi=300,
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )
    return fig
