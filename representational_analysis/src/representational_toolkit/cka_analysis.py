# src/cka_analyzer/analysis.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_cka_analysis(
    model_reference_path: str,
    model_path: str,
    query: list[str],
    output_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
):
    """
    Single‐file CKA analysis:
      - load reference & updated models
      - tokenize & dataloader
      - extract token‐0 activations
      - compute layerwise linear CKA
      - plot & save PDF
    """
    tokenizer = AutoTokenizer.from_pretrained(model_reference_path, use_fast=True)
    def load_model(path):
        m = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return m.to(device).eval()

    model_ref = load_model(model_reference_path)
    model_upd = load_model(model_path)

    # Dataset & DataLoader
    class TextDataset(Dataset):
        def __init__(self, query):
            enc = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            ).to(device)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }

    loader = DataLoader(TextDataset(query), batch_size=batch_size, shuffle=False)

    # Centering Gram matrix
    def center_gram(K):
        n = K.shape[0]
        u = np.ones((n,n)) / n
        return K - u @ K - K @ u + u @ K @ u

    def linear_cka(X, Y):
        Xc = X - X.mean(0, keepdims=True)
        Yc = Y - Y.mean(0, keepdims=True)
        Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
        hsic = np.trace(center_gram(Kx) @ center_gram(Ky))
        denom = np.sqrt(np.trace(center_gram(Kx) @ center_gram(Kx)) *
                        np.trace(center_gram(Ky) @ center_gram(Ky)) + 1e-12)
        return float(hsic / denom)

    # Extract token-0 activations
    def extract_acts(model):
        acts = {}
        num_layers = len(model.model.layers)
        for L in range(num_layers):
            buf = []
            def hook(m,i,o):
                t = o[0] if isinstance(o, tuple) else o
                buf.append(t[:,0,:].float().detach().cpu().numpy())
            h = dict(model.named_modules())[f"model.layers.{L}"].register_forward_hook(hook)
            with torch.no_grad():
                for i,b in enumerate(loader):
                    if i >= num_batches: break
                    model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            h.remove()
            acts[L] = np.concatenate(buf, axis=0)
        return acts

    ref_acts = extract_acts(model_ref)
    upd_acts = extract_acts(model_upd)

    # Compute linear CKA
    cka = {L: linear_cka(ref_acts[L], upd_acts[L]) for L in ref_acts}

    # 1) Configure global plotting style (scientific look + DejaVu Serif font)
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['DejaVu Serif'],
        'font.size':          18,
        'axes.titlesize':     22,
        'axes.labelsize':     18,
        'xtick.labelsize':    16,
        'ytick.labelsize':    16,
        'lines.linewidth':    2,
        'lines.markersize':   8,
        'axes.linewidth':     1.2,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.linestyle':     '--',
        'grid.linewidth':     0.6,
        'grid.alpha':         0.6,
        'legend.frameon':     True,
        'legend.fontsize':    16,
        'legend.title_fontsize': 16,
    })

    # 2) Define color palette and line/marker style
    plot_color = "#1b9e77"   # green for Updated vs Reference
    linestyle = "--"
    marker    = "o"

    # 3) Prepare data for plotting
    layers = list(cka.keys())
    values = [cka[L] for L in layers]

    # 4) Create the figure and axis
    fig, ax = plt.subplots(figsize=(5, 3))

    # 5) Plot the CKA curve with markers every N points
    marker_freq = 5
    marker_indices = [i for i in range(len(layers)) if i % marker_freq == 0]

    ax.plot(
        layers, values,
        linestyle=linestyle,
        color=plot_color,
        label="Updated vs Reference"
    )
    ax.plot(
        [layers[i] for i in marker_indices],
        [values[i] for i in marker_indices],
        marker, 
        color=plot_color,
        markersize=8,
        linestyle='None'
    )

    # 6) Configure ticks on the x-axis
    ax.set_xticks([layers[i] for i in range(len(layers)) if i % marker_freq == 0])
    ax.set_xticklabels([str(l) for i, l in enumerate(layers) if i % marker_freq == 0])

    # 7) Set axis labels and title
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Layerwise CKA", pad=12)

    # 8) Set y-axis limits
    ax.set_ylim(-1, 3)

    # 9) Add legend inside the plot
    ax.legend(loc="best", frameon=False, fancybox=True)

    # 10) Final layout adjustments, save as PDF, then close
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

