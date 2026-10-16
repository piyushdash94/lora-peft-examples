"""
01 - LoRA from scratch in plain PyTorch (no PEFT library)
==========================================================
Goal: see exactly what LoRA does before using a library that hides it.

Idea in one line:
    Instead of updating a big weight matrix W (d_out x d_in), freeze it and learn
    a small low-rank correction  dW = B @ A, where A is (r x d_in), B is (d_out x r).
    Output:  y = W x + (alpha / r) * B A x

Experiment:
    1. Make a "pretrained" layer W0 (512 x 512 = 262,144 weights).
    2. Create a "new task" whose ideal weights are W0 + a rank-4 change.
    3. Fine-tune ONLY the LoRA matrices with r = 1, 4, 16 and compare.
    4. Merge the adapter back into W0 and check outputs are identical.

Runs on CPU in a few seconds.  `python 01_lora_from_scratch.py`
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
D = 512          # layer width
TRUE_RANK = 4    # rank of the change the new task actually needs


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # freeze pretrained weights
            p.requires_grad = False
        # A: small random init, B: zeros  ->  B@A = 0 at start,
        # so the model behaves exactly like the pretrained one before training.
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold B@A into W so inference has zero extra cost."""
        merged = nn.Linear(self.base.in_features, self.base.out_features)
        merged.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
        merged.bias.copy_(self.base.bias)
        return merged


def count(params):
    return sum(p.numel() for p in params)


# ---- 1. "pretrained" layer and a new task that needs a low-rank change ----
pretrained = nn.Linear(D, D)
with torch.no_grad():
    delta = torch.randn(D, TRUE_RANK) @ torch.randn(TRUE_RANK, D) * 0.05
    target_layer = nn.Linear(D, D)
    target_layer.weight.copy_(pretrained.weight + delta)
    target_layer.bias.copy_(pretrained.bias)

X = torch.randn(4096, D)
with torch.no_grad():
    Y = target_layer(X)                  # labels for the new task

print(f"Full fine-tuning would train {count(pretrained.parameters()):,} params\n")

# ---- 2. fine-tune LoRA at different ranks ----
for r in [1, 4, 16]:
    base = nn.Linear(D, D)
    base.load_state_dict(pretrained.state_dict())   # fresh copy of W0
    model = LoRALinear(base, r=r, alpha=2 * r)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-2)

    with torch.no_grad():
        start_loss = nn.functional.mse_loss(model(X), Y).item()
    for step in range(500):
        loss = nn.functional.mse_loss(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pct = 100 * count(trainable) / count(model.parameters())
    print(f"r={r:<3} trainable={count(trainable):>6,} ({pct:4.1f}% of layer) "
          f"loss {start_loss:.4f} -> {loss.item():.6f}")

    # ---- 3. merge and verify ----
    if r == 4:
        merged = model.merge()
        with torch.no_grad():
            diff = (merged(X) - model(X)).abs().max().item()
        print(f"      merged layer vs LoRA layer, max abs diff = {diff:.2e}")

print("""
What to notice
- r=1 cannot fully fit a rank-4 change -> loss stays higher (underfits).
- r=4 matches the true rank and fits it with ~1.5% of the parameters.
- r=16 also fits, but spends 4x the parameters for no gain.
- After merging, outputs match: LoRA adds no inference latency.
This is the core LoRA bet: the change a new task needs is usually low-rank.
""")
