import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import generator.utils as utils
from matplotlib.lines import Line2D  # noqa: F401

utils.set_plot()
# colors_list = ["#C93F3F", "#F97316", "#D4B106", "#16A34A", "#2563EB", "#A855F7"]
colors_list = ["#16A34A", "#D4B106", "#F97316", "#C93F3F", "#A855F7", "#2563EB"]
colors_zorder = [91, 92, 93, 94, 95, 96]
colors_zorder = colors_zorder[::-1]  # reverse order for plotting

# single plots
# TXT_FONT_SIZE = 24
# XLABEL_FONT_SIZE = 24

# combined plots
TXT_FONT_SIZE = 14
XLABEL_FONT_SIZE = 16

INPUT_MAIN_LINE_WIDTH = 3.0
INPUT_BG_LINE_WIDTH = 3.0
OUTPUT_MAIN_LINE_WIDTH = 3.0
OUTPUT_BG_LINE_WIDTH = 5.0
FIXATION_COLOR = "#A02B93"


OUTPUT_LINE_STYLE = "--"

plt.rcParams["font.sans-serif"] = "Arial"


def style_axis(ax, remove_spines=["top", "right"]):
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)

    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    # cut axis
    xt = ax.get_xticks()
    if xt.size >= 2:
        ax.spines["bottom"].set_bounds(xt[1], xt[-2])

    yt = ax.get_yticks()
    if yt.size >= 2:
        ax.spines["left"].set_bounds(yt[1], yt[-2])


def add_phase_lines(ax, tps, y_max, y_min=None):
    if y_min is None:
        y_min = ax.get_ylim()[0]

    # Initial Fixation:     0                   ->  tps["stim1_start"]
    # Encoding:             tps["stim1_start"]  ->  tps["stim3_off"]
    # Delay:                tps["stim3_off"]    ->  tps["delay_end"]
    # Retrieval:            tps["delay_end"]    ->  tps["target3_end"]

    phase_boundaries = [tps["stim1_start"], tps["stim3_off"] - 1, tps["delay_end"] - 1]
    phase_labels = ["Encoding", "Maintenance", "Retrieval"]
    label_positions = [
        (tps["stim1_start"] + tps["stim3_off"]) / 2 - 1,
        (tps["delay_end"] + tps["stim3_off"]) / 2 - 1,
        (tps["delay_end"] + tps["target3_end"]) / 2 - 1,
    ]

    for x in phase_boundaries:
        ax.axvline(x=x, color="gray", linestyle="-", linewidth=1.2, alpha=0.5)

    return phase_boundaries, label_positions, phase_labels


def plot_input_comparison(
    clean_inputs,
    noisy_inputs,
    tps,
    presented_items,
    title="Inputs",
    sample_index=0,
    enable_legend=False,
):
    clean_inputs = np.asarray(clean_inputs[sample_index])  # (tot_len, n_in)
    noisy_inputs = np.asarray(noisy_inputs[sample_index])  # (tot_len, n_in)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Plot fixation with unique color
    fixation_idx = -1

    ax.plot(
        noisy_inputs[: tps["delay_end"], fixation_idx],
        color=FIXATION_COLOR,
        alpha=0.25,
        linewidth=INPUT_BG_LINE_WIDTH,
        label="Fixation (Noisy)",
    )
    ax.plot(
        [tps["delay_end"] - 1, tps["delay_end"] - 1],
        noisy_inputs[tps["delay_end"] - 1 : tps["delay_end"] + 1, fixation_idx],
        color=FIXATION_COLOR,
        alpha=0.25,
        linewidth=INPUT_BG_LINE_WIDTH,
        label="_nolegend_",
    )
    ax.plot(
        range(tps["delay_end"], len(noisy_inputs)),
        noisy_inputs[tps["delay_end"] :, fixation_idx],
        color=FIXATION_COLOR,
        alpha=0.25,
        linewidth=INPUT_BG_LINE_WIDTH,
        label="_nolegend_",
    )

    ax.plot(
        clean_inputs[:, fixation_idx],
        color=FIXATION_COLOR,
        alpha=1.0,
        linewidth=INPUT_MAIN_LINE_WIDTH,
        label="Fixation (Clean)",
        drawstyle="steps-pre",
    )

    # Plot stimulus channels
    stim_channels = 24

    # Define time segments for each item
    segments = [
        (0, tps["stim1_start"], -1),  # pre-stimulus
        (tps["stim1_start"], tps["stim1_off"], presented_items[0]),
        (tps["stim1_off"], tps["stim2_start"], -1),  # inter-stimulus
        (tps["stim2_start"], tps["stim2_off"], presented_items[1]),
        (tps["stim2_off"], tps["stim3_start"], -1),  # inter-stimulus
        (tps["stim3_start"], tps["stim3_off"], presented_items[2]),
        (tps["stim3_off"], END, -1),  # rest
    ]

    for i in range(stim_channels):
        # Calculate overall amplitude for this channel to determine alpha
        amplitude = np.max(np.abs(clean_inputs[:, i]))
        alpha = 0.3 + 0.7 * min(amplitude, 1.0)

        # Draw each segment with corresponding item color
        for seg_idx, (start_t, end_t, item_value) in enumerate(segments):
            if (
                start_t < clean_inputs.shape[0]
                and end_t <= clean_inputs.shape[0]
                and item_value != -1
            ):
                color = colors_list[item_value]

                ax.plot(
                    range(start_t, end_t),
                    clean_inputs[start_t:end_t, i],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax.plot(
                    range(start_t, end_t),
                    noisy_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.15,
                    linewidth=INPUT_MAIN_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax.plot(
                    [start_t, start_t],
                    [0, clean_inputs[start_t:end_t, i][0]],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax.plot(
                    [end_t - 1, end_t - 1],
                    [0, clean_inputs[start_t:end_t, i][-1]],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )

            if item_value == -1:
                color = "gray"

                ax.plot(
                    range(start_t, end_t),
                    clean_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.2,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                )
                ax.plot(
                    range(start_t, end_t),
                    noisy_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.2,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                )

                if start_t > 0:
                    # clean
                    ax.plot(
                        [start_t - 1, start_t],
                        [0, 0],
                        color=color,
                        alpha=0.2,
                        linewidth=INPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                    )
                    # noisy, gaussian sigma=0.1
                    ax.plot(
                        [start_t - 1, start_t],
                        np.random.normal(0, 0.1, 2),
                        color=color,
                        alpha=0.05,
                        linewidth=INPUT_MAIN_LINE_WIDTH,
                        label="_nolegend_",
                    )
                if end_t < END:
                    # clean
                    ax.plot(
                        [end_t - 1, end_t],
                        [0, 0],
                        color=color,
                        alpha=0.2,
                        linewidth=INPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                    )
                    # noisy, gaussian sigma=0.1
                    ax.plot(
                        [end_t - 1, end_t],
                        np.random.normal(0, 0.1, 2),
                        color=color,
                        alpha=0.05,
                        linewidth=INPUT_MAIN_LINE_WIDTH,
                        label="_nolegend_",
                    )

                continue  # skip non-stimulus segments

    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_xlabel("Input", fontsize=XLABEL_FONT_SIZE)

    style_axis(ax)
    _, pos, labs = add_phase_lines(ax, tps, 1)

    # stage labels
    ylim = ax.get_ylim()
    y_text = ylim[1] + (ylim[1] - ylim[0]) * 0.05
    for x, txt in zip(pos, labs):
        ax.text(x, y_text, txt, ha="center", fontsize=TXT_FONT_SIZE)

    # legend - show colors for presented items, positioned above phase_labels
    if enable_legend:
        handles = [Line2D([0], [0], color=FIXATION_COLOR, lw=2, alpha=0.8)]
        labels = ["fixation"]
        for item in range(min(3, len(presented_items))):
            handles.append(
                Line2D(
                    [0], [0], color=colors_list[presented_items[item]], lw=4, alpha=0.8
                )
            )
            labels.append(f"item{presented_items[item]+1}")
        legend_y = y_text + (ylim[1] - ylim[0]) * 0.24
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 + (legend_y - ylim[1]) / (ylim[1] - ylim[0])),
            ncol=4,
            frameon=True,
            fontsize=10,
        ).get_frame().set_alpha(0.25)

    plt.tight_layout()
    return fig


def plot_output_comparison(
    outputs,
    targets,
    tps,
    presented_items,
    title="Outputs",
    sample_index=0,
    enable_legend=False,
):
    outputs = np.asarray(outputs[sample_index])
    targets = np.asarray(targets[sample_index])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Plot fixation with unique color
    fix_idx = -1

    ax.plot(
        targets[:, fix_idx],
        color=FIXATION_COLOR,
        alpha=0.3,
        linewidth=OUTPUT_BG_LINE_WIDTH,
        label="fixation",
        drawstyle="steps-pre",
    )
    ax.plot(
        outputs[:, fix_idx],
        color=FIXATION_COLOR,
        alpha=1.0,
        linewidth=OUTPUT_MAIN_LINE_WIDTH,
        label="_nolegend_",
    )

    # Plot response channels
    item_channels = 6

    # Define retrieval segments for each item
    retrieval_segments = [
        (0, tps["delay_end"], -1),  # pre-retrieval
        (tps["delay_end"], tps["target1_end"] + 1, presented_items[0]),
        (tps["target1_end"], tps["target2_end"] + 1, presented_items[1]),
        (tps["target2_end"], tps["target3_end"] + 1, presented_items[2]),
        (tps["target3_end"], END, -1),  # post-retrieval
    ]

    for i in range(min(item_channels, outputs.shape[1] - 1)):
        # Calculate overall amplitude for this channel to determine alpha
        amplitude = np.max(np.abs(outputs[:, i]))
        alpha = 0.4 + 0.6 * min(amplitude, 1.0)

        # Draw each retrieval segment with corresponding item color
        for start_t, end_t, item_value in retrieval_segments:
            if item_value == -1:
                color = "gray"
                ax.plot(
                    range(start_t, end_t),
                    outputs[start_t:end_t, i],
                    color=color,
                    alpha=0.2,
                    linewidth=OUTPUT_MAIN_LINE_WIDTH,
                    label="_nolegend_",
                )
                continue  # skip non-stimulus segments

            if start_t < outputs.shape[0] and end_t <= outputs.shape[0]:
                color = colors_list[item_value]

                ax.plot(
                    range(start_t, end_t),
                    outputs[start_t:end_t, i],
                    color=color,
                    alpha=alpha,
                    linewidth=OUTPUT_MAIN_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )

                # Check if this segment has target activity
                segment_has_target = np.max(np.abs(targets[start_t:end_t, i])) > 0.01
                if segment_has_target:
                    x_targets = list(range(start_t, end_t))
                    y_targets = list(targets[start_t:end_t, i])
                    ax.plot(
                        x_targets,
                        y_targets,
                        color=color,
                        alpha=0.3,
                        linewidth=OUTPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )
                    ax.plot(
                        [start_t, start_t],
                        [0, targets[start_t:end_t, i][0]],
                        color=color,
                        alpha=0.3,
                        linewidth=OUTPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )
                    ax.plot(
                        [end_t - 1, end_t - 1],
                        [0, targets[start_t:end_t, i][-1]],
                        color=color,
                        alpha=0.3,
                        linewidth=OUTPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )

    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_xlabel("Output", fontsize=XLABEL_FONT_SIZE)
    ax.set_yticks([0, 1])

    style_axis(ax)
    add_phase_lines(ax, tps, 1)

    _, pos, labs = add_phase_lines(ax, tps, 1)
    # stage labels
    ylim = ax.get_ylim()
    y_text = ylim[1] + (ylim[1] - ylim[0]) * 0.05
    for x, txt in zip(pos, labs):
        ax.text(x, y_text, txt, ha="center", fontsize=TXT_FONT_SIZE)

    # legend - show colors for presented items and fixation, positioned above phase_labels
    if enable_legend:
        handles = [Line2D([0], [0], color=FIXATION_COLOR, lw=2, alpha=0.8)]
        labels = ["fixation"]
        for item in range(min(3, len(presented_items))):
            handles.append(
                Line2D(
                    [0], [0], color=colors_list[presented_items[item]], lw=4, alpha=0.8
                )
            )
            labels.append(f"item{presented_items[item]+1}")
        legend_y = y_text + (ylim[1] - ylim[0]) * 0.24
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 + (legend_y - ylim[1]) / (ylim[1] - ylim[0])),
            ncol=4,
            frameon=True,
            fontsize=10,
        ).get_frame().set_alpha(0.25)

    plt.tight_layout()
    return fig


def plot_input_and_output_comparison(
    clean_inputs,
    noisy_inputs,
    outputs,
    targets,
    tps,
    presented_items,
    title="Inputs and Outputs",
    sample_index=0,
    enable_legend=False,
):
    clean_inputs = np.asarray(clean_inputs[sample_index])  # (tot_len, n_in)
    noisy_inputs = np.asarray(noisy_inputs[sample_index])  # (tot_len, n_in)
    outputs = np.asarray(outputs[sample_index])
    targets = np.asarray(targets[sample_index])

    fig, ax_input = plt.subplots(1, 1, figsize=(7, 2.25))
    ax_output = ax_input.twinx()

    # Add background spans for phases
    # Encoding: tps["stim1_start"] -> tps["stim3_off"]
    ax_input.axvspan(
        tps["stim1_start"], tps["stim3_off"] - 1, color="#E3F2FD", alpha=1.0, zorder=0
    )
    # Maintaining: tps["stim3_off"] -> tps["delay_end"]
    ax_input.axvspan(
        tps["stim3_off"] - 1, tps["delay_end"] - 1, color="#F5F5F5", alpha=1.0, zorder=0
    )
    # Retrieval: tps["delay_end"] -> tps["target3_end"]
    ax_input.axvspan(
        tps["delay_end"] - 1, tps["target3_end"], color="#E0F2F1", alpha=1.0, zorder=0
    )

    # Plot output fixation with unique color
    fix_idx = -1
    ax_output.plot(
        targets[:, fix_idx],
        color=FIXATION_COLOR,
        alpha=0.3,
        linewidth=OUTPUT_BG_LINE_WIDTH,
        label="fixation (target)",
        drawstyle="steps-pre",
        zorder=100,
    )
    ax_output.plot(
        outputs[:, fix_idx],
        color=FIXATION_COLOR,
        alpha=1.0,
        linewidth=OUTPUT_MAIN_LINE_WIDTH,
        linestyle=OUTPUT_LINE_STYLE,
        label="_nolegend_",
        zorder=100,
    )

    # Plot input stimulus channels (excluding fixation)
    stim_channels = 24

    # Define time segments for each item (for inputs)
    input_segments = [
        (0, tps["stim1_start"], -1),  # pre-stimulus
        (tps["stim1_start"], tps["stim1_off"], presented_items[0]),
        (tps["stim1_off"], tps["stim2_start"], -1),  # inter-stimulus
        (tps["stim2_start"], tps["stim2_off"], presented_items[1]),
        (tps["stim2_off"], tps["stim3_start"], -1),  # inter-stimulus
        (tps["stim3_start"], tps["stim3_off"], presented_items[2]),
        (tps["stim3_off"], END, -1),  # rest
    ]

    for i in range(stim_channels):
        # Calculate overall amplitude for this channel to determine alpha
        amplitude = np.max(np.abs(clean_inputs[:, i]))
        alpha = 0.3 + 0.7 * min(amplitude, 1.0)

        # Draw each segment with corresponding item color
        for seg_idx, (start_t, end_t, item_value) in enumerate(input_segments):
            if (
                start_t < clean_inputs.shape[0]
                and end_t <= clean_inputs.shape[0]
                and item_value != -1
            ):
                color = colors_list[item_value]

                ax_input.plot(
                    range(start_t, end_t),
                    clean_inputs[start_t:end_t, i],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax_input.plot(
                    range(start_t, end_t),
                    noisy_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.15,
                    linewidth=INPUT_MAIN_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax_input.plot(
                    [start_t, start_t],
                    [0, clean_inputs[start_t:end_t, i][0]],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax_input.plot(
                    [end_t - 1, end_t - 1],
                    [0, clean_inputs[start_t:end_t, i][-1]],
                    color=color,
                    alpha=alpha,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )

            if item_value == -1:
                color = "gray"

                ax_input.plot(
                    range(start_t, end_t),
                    clean_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.2,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )
                ax_input.plot(
                    range(start_t, end_t),
                    noisy_inputs[start_t:end_t, i],
                    color=color,
                    alpha=0.2,
                    linewidth=INPUT_BG_LINE_WIDTH,
                    label="_nolegend_",
                    zorder=100,
                )

                if start_t > 0:
                    # clean
                    ax_input.plot(
                        [start_t - 1, start_t],
                        [0, 0],
                        color=color,
                        alpha=0.2,
                        linewidth=INPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )
                    # noisy, gaussian sigma=0.1
                    ax_input.plot(
                        [start_t - 1, start_t],
                        np.random.normal(0, 0.1, 2),
                        color=color,
                        alpha=0.05,
                        linewidth=INPUT_MAIN_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )
                if end_t < END:
                    # clean
                    ax_input.plot(
                        [end_t - 1, end_t],
                        [0, 0],
                        color=color,
                        alpha=0.2,
                        linewidth=INPUT_BG_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )
                    # noisy, gaussian sigma=0.1
                    ax_input.plot(
                        [end_t - 1, end_t],
                        np.random.normal(0, 0.1, 2),
                        color=color,
                        alpha=0.05,
                        linewidth=INPUT_MAIN_LINE_WIDTH,
                        label="_nolegend_",
                        zorder=100,
                    )

                continue  # skip non-stimulus segments

    # Plot output response channels (including all channels)
    item_channels = 6

    # Define retrieval segments for each item (for outputs)
    retrieval_segments = [
        # (0, tps["delay_end"], -1),  # pre-retrieval
        (tps["delay_end"] - 1, tps["target1_end"] + 1, presented_items[0]),
        (tps["target1_end"], tps["target2_end"] + 1, presented_items[1]),
        (tps["target2_end"], tps["target3_end"] + 1, presented_items[2]),
        # (tps["target3_end"], END, -1),  # post-retrieval
    ]

    for i in range(item_channels):
        amplitude = np.max(np.abs(outputs[:, i]))
        alpha = 0.4 + 0.6 * min(amplitude, 1.0)

        if i not in presented_items:
            ax_output.plot(
                range(0, END),
                outputs[0:END, i],
                color="gray",
                linestyle=OUTPUT_LINE_STYLE,
                linewidth=OUTPUT_MAIN_LINE_WIDTH,
                label="_nolegend_",
                zorder=10,
            )
            continue

        color = colors_list[i]
        zorder = colors_zorder[i]
        ax_output.plot(
            range(0, END),
            outputs[0:END, i],
            color=color,
            linestyle=OUTPUT_LINE_STYLE,
            linewidth=OUTPUT_MAIN_LINE_WIDTH,
            label="_nolegend_",
            zorder=zorder,
        )

        for start_t, end_t, item_value in retrieval_segments:
            if item_value != i:
                continue
            ax_output.plot(
                [0, start_t],
                [0, 0],
                color=color,
                alpha=0.3,
                linewidth=OUTPUT_BG_LINE_WIDTH,
                label="_nolegend_",
                zorder=zorder,
            )

            # Draw vertical line at start_t from 0 to 1
            ax_output.plot(
                [start_t, start_t],
                [0, 1],
                color=color,
                alpha=0.3,
                linewidth=OUTPUT_BG_LINE_WIDTH,
                label="_nolegend_",
                zorder=zorder,
            )
            # Draw horizontal line from start_t to end_t at y=1
            ax_output.plot(
                [start_t, end_t - 1],
                [1, 1],
                color=color,
                alpha=0.3,
                linewidth=OUTPUT_BG_LINE_WIDTH,
                label="_nolegend_",
                zorder=zorder,
            )
            # Draw vertical line at end_t from 1 to 0
            ax_output.plot(
                [end_t - 1, end_t - 1],
                [1, 0],
                color=color,
                alpha=0.3,
                linewidth=OUTPUT_BG_LINE_WIDTH,
                label="_nolegend_",
                zorder=zorder,
            )

            # end_t -> trial end
            ax_output.plot(
                [end_t - 1, END - 1],
                [0, 0],
                color=color,
                alpha=0.3,
                linewidth=OUTPUT_BG_LINE_WIDTH,
                label="_nolegend_",
                zorder=zorder,
            )

    ax_input.yaxis.set_label_coords(-0.05, 0.5)
    ax_input.set_xlabel("Timestep", fontsize=XLABEL_FONT_SIZE)
    ax_input.set_ylabel("Stimulus", fontsize=XLABEL_FONT_SIZE)
    ax_output.set_ylabel("Output", fontsize=XLABEL_FONT_SIZE)

    ax_input.set_ylim(-1, 5)
    ax_input.set_yticks([0, 4])
    ax_output.set_ylim(-0.25, 1.25)
    ax_output.set_yticks([0, 1])

    # Annotate inter-stimulus intervals with double-headed arrows
    arrow_y = 1.8
    ax_input.annotate(
        "",
        xy=(tps["stim2_start"] + 1, arrow_y),
        xytext=(tps["stim1_off"] - 2, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="gray", linewidth=2.0),
        clip_on=False,
    )
    ax_input.text(
        (tps["stim1_off"] + tps["stim2_start"]) / 2,
        arrow_y + 0.12,
        r"$t_\mathrm{int}$",
        ha="center",
        va="bottom",
        fontsize=TXT_FONT_SIZE,
    )

    # ax_input.annotate(
    #     "",
    #     xy=(tps["stim3_start"]+1, arrow_y),
    #     xytext=(tps["stim2_off"]-2, arrow_y),
    #     arrowprops=dict(arrowstyle="<->", color="gray", linewidth=1.2),
    #     clip_on=False,
    # )
    # ax_input.text(
    #     (tps["stim2_off"] + tps["stim3_start"]) / 2,
    #     arrow_y + 0.12,
    #     r"$t_\mathrm{int}^\mathrm{(2-3)}$",
    #     ha="center",
    #     va="bottom",
    #     fontsize=TXT_FONT_SIZE,
    # )

    for spine in ["top"]:
        ax_input.spines[spine].set_visible(False)
        ax_output.spines[spine].set_visible(False)

    ax_input.spines["right"].set_visible(False)
    ax_output.spines["left"].set_visible(False)
    ax_output.spines["bottom"].set_visible(False)

    ax_input.tick_params(direction="out", length=4, width=1)
    ax_output.tick_params(direction="out", length=4, width=1)

    # cut axis
    xt = ax_input.get_xticks()
    if xt.size >= 2:
        ax_input.spines["bottom"].set_bounds(xt[1], xt[-2])

    ax_input.spines["left"].set_bounds(0, 4)
    ax_output.spines["right"].set_bounds(0, 1)

    _, pos, labs = add_phase_lines(ax_input, tps, 5, -1)

    # stage labels
    ylim = ax_input.get_ylim()
    y_text = ylim[1] + (ylim[1] - ylim[0]) * 0.2
    for x, txt in zip(pos, labs):
        ax_input.text(x, y_text, txt, ha="center", fontsize=TXT_FONT_SIZE)

    # legend - show colors for presented items and fixation, positioned above phase_labels
    if enable_legend:
        handles = [Line2D([0], [0], color=FIXATION_COLOR, lw=2, alpha=0.8)]
        labels = ["fixation"]
        for item in range(min(3, len(presented_items))):
            handles.append(
                Line2D(
                    [0], [0], color=colors_list[presented_items[item]], lw=4, alpha=0.8
                )
            )
            labels.append(f"item{presented_items[item]+1}")
        legend_y = y_text + (ylim[1] - ylim[0]) * 0.24
        ax_input.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 + (legend_y - ylim[1]) / (ylim[1] - ylim[0])),
            ncol=4,
            frameon=True,
            fontsize=10,
        ).get_frame().set_alpha(0.25)

    ax_input.grid(False)
    ax_output.grid(False)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=0)

    return fig


if __name__ == "__main__":
    TASK = "forward"
    N_SAMPLES = 120
    FILENAME = Path(__file__).name
    clean_params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME, noise_sigma=0.0)
    noisy_params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME, noise_sigma=0.1)

    N_HID = clean_params["N_HID"]
    N_CLASS = clean_params["N_CLASS"]
    PERMS = clean_params["PERMS"]
    FIG_DIR = clean_params["FIG_DIR"]
    tps = clean_params["TIMEPOINTS"]
    END = tps["target3_end"] + 1

    clean_outputs_dict = clean_params["OUTPUTS_DICT"]
    clean_inputs = clean_outputs_dict["inputs"][:, :END, :]
    clean_outputs = clean_outputs_dict["outputs"][:, :END, :]
    clean_targets = clean_outputs_dict["targets"][:, :END, :]

    noisy_outputs_dict = noisy_params["OUTPUTS_DICT"]
    noisy_inputs = noisy_outputs_dict["inputs"]
    noisy_inputs = noisy_inputs[:, :END, :]

    # idx 51: permutation (2, 3, 5)
    idx = 51
    presented_items = PERMS[idx]

    # fig1 = plot_input_comparison(
    #     clean_inputs, noisy_inputs, tps, presented_items, sample_index=idx
    # )
    # fig1.savefig(f"{FIG_DIR}/inputs.png", dpi=600)
    # print(f"Figure 1 generated: {FIG_DIR}/inputs.png")

    # fig2 = plot_output_comparison(
    #     clean_outputs, clean_targets, tps, presented_items, sample_index=idx
    # )
    # fig2.savefig(f"{FIG_DIR}/outputs.png", dpi=600)
    # print(f"Figure 2 generated: {FIG_DIR}/outputs.png")

    fig3 = plot_input_and_output_comparison(
        clean_inputs,
        noisy_inputs,
        clean_outputs,
        clean_targets,
        tps,
        presented_items,
        sample_index=idx,
    )
    fig3.savefig(f"{FIG_DIR}/inputs_and_outputs.pdf")
    print(f"Figure 3 generated: {FIG_DIR}/inputs_and_outputs.pdf")

    # plt.show()
