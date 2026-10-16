"""
04 - Visualise LoRA weights
===========================
Figure 1 (toy, trained here in about 2 seconds): what W = W0 + (alpha/r)·B·A looks like.
Figure 2 (real): the adapter you trained in script 03, opened up layer by layer.

    python 04_visualize_lora.py            # needs outputs/ from script 03 for fig 2
"""
import json, os
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

torch.manual_seed(0)
BLUE, ORANGE, TEAL = "#2a6fdb", "#d9730d", "#12876f"
os.makedirs("figures", exist_ok=True)

def heat(ax, M, title, vmax=None):
    """Signed weights -> diverging map centred on 0 (white = 0)."""
    M = M.detach().numpy()
    v = vmax or abs(M).max()
    ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

def sym(ax, s):
    ax.axis("off"); ax.text(0.5, 0.5, s, fontsize=22, ha="center", va="center")

# ------------------------------------------------------------------ toy
D, TRUE_R = 24, 3
W0 = torch.randn(D, D) * 0.5                                  # "pretrained"
dW_true = (torch.randn(D, TRUE_R) @ torch.randn(TRUE_R, D)) * 0.3  # rank-3 task change
X = torch.randn(512, D); Y = X @ (W0 + dW_true).T

def train(r, steps=800):
    A = torch.nn.Parameter(torch.randn(r, D) / r)    # small random
    B = torch.nn.Parameter(torch.zeros(D, r))        # zero -> dW starts at 0
    s = 2.0                                          # alpha / r
    opt = torch.optim.Adam([A, B], lr=1e-2); losses = []
    for _ in range(steps):
        loss = ((X @ (W0 + s * B @ A).T - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    return A.detach(), B.detach(), s, losses

runs = {r: train(r) for r in (1, 3, 8)}
A, B, s, _ = runs[3]
dW = s * B @ A

fig = plt.figure(figsize=(13, 7.2))
gs = GridSpec(2, 7, figure=fig, height_ratios=[1, 1.05],
              width_ratios=[D, 5, TRUE_R * 2.2, 3, D, 5, D], hspace=0.35)
v = abs(W0 + dW).max().item()
heat(fig.add_subplot(gs[0, 0]), W0, f"W0  frozen\n{D}×{D} = {D*D} params", v)
sym(fig.add_subplot(gs[0, 1]), "+  α/r ·")
heat(fig.add_subplot(gs[0, 2]), B, f"B (trained)\n{D}×{TRUE_R}"); sym(fig.add_subplot(gs[0, 3]), "×")
heat(fig.add_subplot(gs[0, 4]), A.repeat_interleave(3, 0), f"A (trained, rows stretched)\n{TRUE_R}×{D}")
sym(fig.add_subplot(gs[0, 5]), "=")
heat(fig.add_subplot(gs[0, 6]), W0 + dW, "W used at inference\n(merge = add ΔW once)", v)
fig.text(0.5, 0.96, f"LoRA: train only B and A  →  {2*D*TRUE_R} params instead of {D*D} "
         f"({100*2*TRUE_R/D:.0f}%)", ha="center", fontsize=12, weight="bold")

g2 = gs[1, :].subgridspec(1, 4, wspace=0.35, width_ratios=[1, 1, 1.5, 1.5])
heat(fig.add_subplot(g2[0]), dW_true, "true ΔW the task needs")
heat(fig.add_subplot(g2[1]), dW, "learned (α/r)·B·A, r=3\n(should match)", abs(dW_true).max().item())

ax = fig.add_subplot(g2[2])
for (r, (_, _, _, L)), c in zip(runs.items(), (ORANGE, BLUE, TEAL)):
    ax.plot(L, color=c, lw=2, label=f"r={r}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (log)")
ax.set_title("Loss: r=1 too small, r≥3 fits", fontsize=9); ax.legend(frameon=False)

ax = fig.add_subplot(g2[3])
A8, B8, s8, _ = runs[8]
sv_true = torch.linalg.svdvals(dW_true)[:8]; sv_8 = torch.linalg.svdvals(s8 * B8 @ A8)[:8]
idx = torch.arange(1, 9)
ax.bar(idx - 0.2, sv_true, 0.4, color="#9aa4b2", label="true ΔW")
ax.bar(idx + 0.2, sv_8, 0.4, color=TEAL, label="learned, r=8")
ax.set_xlabel("singular value #"); ax.set_title("Given r=8, it still uses only 3 directions", fontsize=9)
ax.legend(frameon=False)
for a in fig.axes[-2:]:
    a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
fig.savefig("figures/fig1_lora_toy.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ real adapter
AD = "outputs/smollm-scorecard-lora/adapter"
if not os.path.exists(AD):
    raise SystemExit("fig1 saved. Run script 03 first to get fig 2.")
from safetensors.torch import load_file
cfg = json.load(open(f"{AD}/adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
sd = load_file(f"{AD}/adapter_model.safetensors")
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
n_layers = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in sd)

def delta(layer, mod):
    pre = next(k for k in sd if f"layers.{layer}." in k and f".{mod}.lora_A" in k)
    return scale * sd[pre.replace("lora_A", "lora_B")].float() @ sd[pre].float()

rms = torch.tensor([[delta(l, m).pow(2).mean().sqrt().item() for m in MODS]
                    for l in range(n_layers)])
mid = n_layers // 2

fig, ax = plt.subplots(1, 3, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
im = ax[0].imshow(rms, cmap="Blues", aspect="auto")
ax[0].set_xticks(range(7), [m.replace("_proj", "") for m in MODS], rotation=45)
ax[0].set_ylabel("layer"); ax[0].set_title("How much each layer changed\nRMS of ΔW", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=0.05)

for m, c in zip(["q_proj", "v_proj", "down_proj"], (BLUE, ORANGE, TEAL)):
    sv = torch.linalg.svdvals(delta(mid, m))[:24]
    ax[1].plot(range(1, 25), sv / sv[0], "o-", color=c, ms=4, lw=2, label=m)
ax[1].axvline(cfg["r"] + 0.5, color="#666", ls="--", lw=1)
ax[1].text(cfg["r"] + 0.8, 0.8, f"r={cfg['r']}: rank can't go\nbeyond this", fontsize=8)
ax[1].set_xlabel("singular value #"); ax[1].set_ylabel("σᵢ / σ₁")
ax[1].set_title(f"Layer {mid}: few directions dominate", fontsize=10); ax[1].legend(frameon=False)
ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(alpha=0.25)

heat(ax[2], delta(mid, "q_proj")[:64, :64], f"Layer {mid} q_proj ΔW (64×64 crop of 576×576)\n"
     "row/column stripes = low-rank structure")
fig.tight_layout(); fig.savefig("figures/fig2_real_adapter.png", dpi=150, bbox_inches="tight")
print("saved figures/fig1_lora_toy.png, figures/fig2_real_adapter.png")
