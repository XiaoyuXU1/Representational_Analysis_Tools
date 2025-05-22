import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def run_fim_analysis(
    model_reference_path: str,
    model_path: str,
    query: list[str],
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
    layers_to_analyze: list[int] | None = None,
):
    """
    Compare the reference and updated models by computing the diagonal of the
    Fisher Information Matrix (FIM) for specified layers, then plot histograms
    of these values for each layer.

    Args:
        model_reference_path: Path or identifier for the reference (original) model.
        model_path: Path or identifier for the updated model.
        query: List of input text strings for analysis.
        output_dir: Directory where output histograms will be saved.
        batch_size: Number of samples per inference batch.
        num_batches: Number of batches to use for FIM estimation.
        max_length: Maximum token length for padding/truncation.
        layers_to_analyze: Specific layer indices to analyze; if None, all layers.
    """

    # Load tokenizer and define a helper to load models
    tokenizer = AutoTokenizer.from_pretrained(model_reference_path, use_fast=True)
    def load_model(path: str) -> AutoModelForCausalLM:
        """Load and prepare a causal language model."""
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return model.to(device).eval()
    # Prepare dataset and dataloader for input query
    class TextDataset(Dataset):
        def __init__(self, query: list[str]):
            # Tokenize with padding and truncation
            enc = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["input_ids"]  # Labels equal inputs for loss computation

        def __len__(self) -> int:
            return self.input_ids.size(0)

        def __getitem__(self, idx: int) -> dict:
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    loader = DataLoader(
        TextDataset(query),
        batch_size=batch_size,
        shuffle=False
    )

    # Function to compute diagonal of the Fisher Information Matrix for a layer
    def compute_fim_diag(model: AutoModelForCausalLM, layer_key: str) -> np.ndarray:
        # Collect parameters belonging to the specified layer
        params = [p for name, p in model.named_parameters() if p.requires_grad and layer_key in name]
        # Initialize accumulator for squared gradients
        acc = [torch.zeros_like(p) for p in params]

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            # Move batch data to the compute device
            batch = {k: v.to(device) for k, v in batch.items()}
            # Zero gradients before backward pass
            model.zero_grad()
            # Forward pass with labels to compute loss
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            ).loss
            # Backward pass to accumulate gradients
            loss.backward()
            # Accumulate squared gradients for each parameter
            for j, p in enumerate(params):
                acc[j] += p.grad.detach() ** 2

        # Average the accumulated squared gradients
        denom = min(len(loader), num_batches)
        # Flatten and concatenate all parameter arrays
        fim_values = torch.cat([a.view(-1).float() / denom for a in acc]).cpu().numpy()
        return fim_values

    # --- 1) Configure global plotting style (scientific look + DejaVu Serif font) ---
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['DejaVu Serif'],
        'font.size':          18,
        'axes.titlesize':     20,
        'axes.labelsize':     16,
        'xtick.labelsize':    16,
        'ytick.labelsize':    16,
        'legend.fontsize':    16,
        'figure.dpi':         300,
        'axes.grid':          True,
        'grid.linestyle':     '--',
        'grid.linewidth':     0.6,
        'grid.alpha':         0.6
    })

    # --- 2) Define color palette and line styles for Reference vs. Updated ---
    plot_colors = {
        "Reference": "#d62728",  # red
        "Updated":   "#1f77b4"   # blue
    }
    linestyles = {
        "Reference": "-",
        "Updated":   "--"
    }

    # --- 3) Ensure the output directory exists ---
    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(model_reference_path)
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    layers_to_analyze = list(range(num_layers))
    # --- 4) Loop over each layer and plot the FIM histograms ---
    for L in layers_to_analyze:
        # Compute the diagonal FIM values for both models
        # Instantiate reference and updated models
        model_ref = load_model(model_reference_path)
        model_upd = load_model(model_path)
        fim_ref = compute_fim_diag(model_ref, f"model.layers.{L}")
        fim_upd = compute_fim_diag(model_upd, f"model.layers.{L}")

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(5, 3))

        # Plot histograms for each tag
        for tag, vals in [("Reference", fim_ref), ("Updated", fim_upd)]:
            ax.hist(
                vals,
                bins=40,
                histtype='step',
                linewidth=2,
                linestyle=linestyles[tag],
                color=plot_colors[tag],
                label=tag
            )

        # Use a logarithmic x-axis for dynamic range
        ax.set_xscale('log')
        ax.set_xlabel("FIM diagonal values (log scale)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"FIM Histogram @ Layer {L}", pad=12)

        # Dynamically adjust the y-axis to avoid overlap with the legend
        ymax = 0
        for vals in (fim_ref, fim_upd):
            counts, _ = np.histogram(vals, bins=40)
            ymax = max(ymax, counts.max())
        ax.set_ylim(0, ymax *1.2)

        # Dynamically pad the x-axis limits to prevent clipping at the edges
        x_all = np.concatenate([fim_ref, fim_upd])
        x_min, x_max = x_all.min(), x_all.max()
        pad = 0.1 * (np.log10(x_max + 1e-12) - np.log10(x_min + 1e-12))
        ax.set_xlim(
            10 ** (np.log10(x_min + 1e-12) - pad),
            10 ** (np.log10(x_max + 1e-12) + pad)
        )

        # Place the legend inside the plot area
        ax.legend(loc="upper right", frameon=False, fancybox=True)

        # Finalize layout, save as PDF, and close the figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"layer_{L}.pdf"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)
        # === 清理缓存 ===
        del fim_upd,fim_ref,model_upd,model_ref
        torch.cuda.empty_cache()

    print(f"[✓] All histograms saved under '{output_dir}'")



