import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import generator.utils as utils

#### parameters
TASK = "switch"
N_SAMPLES = 240
FILENAME = Path(__file__).name
params = utils.initialize_analysis(TASK, N_SAMPLES, FILENAME)

N_HID = params["N_HID"]
N_CLASS = params["N_CLASS"]
PERMS = params["PERMS"]
FIG_DIR = params["FIG_DIR"]

outputs_dict = params["OUTPUTS_DICT"]
tps = params["TIMEPOINTS"]

hiddens = outputs_dict["hiddens"]
batch = outputs_dict["batch"]

hiddens = np.tanh(hiddens)  # shape (240, T, N_HID)

pre_cue_st = tps["pre_cue_start"]
cue_st = tps["cue_start"]
post_cue_ed = tps["post_cue_end"]

# Select time points from pre_cue_st to post_cue_ed
time_indices = np.arange(pre_cue_st, post_cue_ed)
T_selected = len(time_indices)

# Concatenate forward and backward hiddens along the time dimension to form matrix of size 2T x N_HID
hiddens_forward_selected = hiddens[
    :N_CLASS, time_indices, :
]  # (120, T_selected, N_HID)
hiddens_backward_selected = hiddens[
    N_CLASS:, time_indices, :
]  # (120, T_selected, N_HID)

# Reshape to concatenate time and sample dimensions
# We want to combine all forward and backward samples at all selected time points
hiddens_forward_concat = hiddens_forward_selected.reshape(
    -1, N_HID
)  # (120*T_selected, N_HID)
hiddens_backward_concat = hiddens_backward_selected.reshape(
    -1, N_HID
)  # (120*T_selected, N_HID)

# Stack forward and backward vertically: (2*120*T_selected, N_HID)
hiddens_combined = np.vstack([hiddens_forward_concat, hiddens_backward_concat])

# Perform PCA on the N_HID dimension
pca = PCA(n_components=3)
pca.fit(hiddens_combined)
pc_subspace = pca.components_  # (3, N_HID) - first 3 PCs

print(f"PCA mean shape: {pca.mean_.shape}")
print(
    f"PCA mean statistics - min: {pca.mean_.min():.3f}, max: {pca.mean_.max():.3f}, mean: {pca.mean_.mean():.3f}"
)

# Project all hiddens onto the 3D subspace using pca.transform (which handles centering)
# Reshape for transform: (batch*T_selected, N_HID)
hiddens_forward_for_transform = hiddens_forward_selected.reshape(-1, N_HID)
hiddens_backward_for_transform = hiddens_backward_selected.reshape(-1, N_HID)

# Transform and reshape back
hiddens_forward_3d = pca.transform(hiddens_forward_for_transform).reshape(
    N_CLASS, T_selected, 3
)
hiddens_backward_3d = pca.transform(hiddens_backward_for_transform).reshape(
    N_CLASS, T_selected, 3
)

mean_forward = np.mean(hiddens_forward_3d, axis=0)
mean_backward = np.mean(hiddens_backward_3d, axis=0)

# Plot 3D trajectories
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# forward trajectories
for i in range(N_CLASS):
    ax.plot(
        hiddens_forward_3d[i, :, 0],
        hiddens_forward_3d[i, :, 1],
        hiddens_forward_3d[i, :, 2],
        color="blue",
        alpha=0.3,
        linewidth=1,
        label="Forward" if i == 0 else "",
    )

# backward trajectories
for i in range(N_CLASS):
    ax.plot(
        hiddens_backward_3d[i, :, 0],
        hiddens_backward_3d[i, :, 1],
        hiddens_backward_3d[i, :, 2],
        color="red",
        alpha=0.3,
        linewidth=1,
        label="Backward" if i == 0 else "",
    )

# Plot mean trajectories
ax.plot(
    mean_forward[:, 0],
    mean_forward[:, 1],
    mean_forward[:, 2],
    color="darkblue",
    linewidth=3,
    label="Mean Forward",
)
ax.plot(
    mean_backward[:, 0],
    mean_backward[:, 1],
    mean_backward[:, 2],
    color="darkred",
    linewidth=3,
    label="Mean Backward",
)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)")
ax.set_title(
    f"Switch Trajectories in 3D PC Subspace\nTime range: {pre_cue_st} to {post_cue_ed} (T={T_selected})"
)
ax.legend()

plt.tight_layout()
save_path = Path(FIG_DIR) / "switch_trajectory_3d_pca.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"3D trajectory plot saved to {save_path}")
plt.show()

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(
    f"Total variance explained by first 3 PCs: {np.sum(pca.explained_variance_ratio_)*100:.2f}%"
)
