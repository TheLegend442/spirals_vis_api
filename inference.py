
from transformer import SpiralTransformer, SpiralDataset, spiral_collate

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

def run_inference_and_plot(
        model,
        dataset: "SpiralDataset",
        device,
        thr_tight: float = 0.5,
        plot_title: str = "Spiral Prediction",
        save_path: str = None,
        plot_recovered: bool = False,
):
    """
    Runs model on a single spiral, maps patch-level predictions back to T,
    and calls `plot_tight` with predicted outputs.

    - tight/flat are multi-label per patch -> sigmoid -> threshold
    - severity/normality are logits -> sigmoid to [0,1]
    - k is one value per spiral
    """
    model.eval()
    spiral_preds = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]

            seq_emb = sample["seq_emb"].unsqueeze(0).to(device)  # [1, L, d_emb]
            L = sample["seq_emb"].shape[0]
            padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)

            out = model(seq_emb, padding_mask=padding_mask)

            k_pred = out["k_pred"][0].item()  # scalar
            k_tight_pred = out["k_tight_pred"][0].item()  # scalar
            ang_raw_pred = out["ang_raw"][0].item() # scalar


            theta = sample["theta"].cpu().numpy()  # [T]
            r     = sample["r"].cpu().numpy()      # [T]
            k     = sample["k"].item()             # scalar
            k_tight = sample["k_tight"].item()     # scalar
            theta_tight = sample.get("theta_tight", None) # scalar or None
            
            shift_angle = sample['shift_angle'].item()
            inversion = sample['inversion'].item()
            r_shift = sample['r_shift'].item()
            org_theta = sample['org_theta']
            org_r = sample['org_r']
        
            # build dict for plotting / analysis
            spiral_pred = {
                "theta": theta,
                "r": r,
                "k": k,
                "k_tight": k_tight,
                "theta_tight": theta_tight,

                # predictions
                "k_pred": k_pred,
                "k_tight_pred": k_tight_pred,
                "ang_raw_pred": ang_raw_pred,
                
                # recover params
                "shift_angle": shift_angle,
                "inversion": inversion,
                "r_shift": r_shift,
                "org_theta": org_theta,
                "org_r": org_r,
            }

            spiral_preds.append(spiral_pred)
        
        plot_dict = spiral_preds[0]
        plot_tight_distribution(plot_dict, plot_title=plot_title, threshold=thr_tight, save_path=save_path, plot_recovered=plot_recovered)

def plot_tight_distribution(
    spiral_dict,
    plot_title,
    VIEW_WIDTH=5 * np.pi / 50, 
    ax1=None,
    ax2=None,
    threshold=None,
    save_path=None,
    plot_recovered: bool = False,
):

    ## PARSE THE SPIRAL
    theta = np.asarray(spiral_dict['theta'])
    r = np.asarray(spiral_dict['r'])
    
    # Parse predictions
    k_pred = spiral_dict['k_pred']
    k_tight_pred = spiral_dict['k_tight_pred']
    theta_tight_pred = spiral_dict['ang_raw_pred']
    
    # Check if there is ground truth info for recovery
    GROUND_TRUTH = False
    if spiral_dict['shift_angle'] != -1.0:
        SHIFT_ANGLE = spiral_dict['shift_angle']
        INVERSION = spiral_dict['inversion']
        R_SHIFT = spiral_dict['r_shift']
        org_theta = np.asarray(spiral_dict['org_theta'])
        org_r = np.asarray(spiral_dict['org_r'])
        GROUND_TRUTH = True
        print(INVERSION, SHIFT_ANGLE)
    
    ## RECOVER ORIGINAL SPIRAL
    if GROUND_TRUTH:
        tt_pred_unrec = theta_tight_pred # Unrecovered angle
        theta_tight_pred += SHIFT_ANGLE
        theta_tight_pred = -theta_tight_pred if INVERSION else theta_tight_pred
        
        theta_rec = theta.copy()
        r_rec = r.copy()
        theta_rec += SHIFT_ANGLE
        if INVERSION:
            theta_rec = -theta_rec
        r_rec = r_rec + spiral_dict['r_shift']
    
    IS_TIGHT = True
    if threshold is not None:
        IS_TIGHT = (k_pred / k_tight_pred) > threshold
    
    ## DIVERGENCE CALCULATION
    def get_divergence(_theta, _r, view_angle, view_width, inv=False):
        r_tight = []
        r_normal = []
        theta_tight = []
        theta_normal = []
        
        if inv:
            tight_slots = []
            ang = view_angle % (2*np.pi) - 2*np.pi
            
            while ang > _theta[-1] - view_width / 2:
                tight_slots.append((ang - view_width/2, ang + view_width/2))
                ang -= 2*np.pi
                
            normal_slots = []
            ang = view_angle % (2*np.pi) + np.pi - 2*np.pi
            
            while ang > _theta[-1] - view_width / 2:
                normal_slots.append((ang - view_width/2, ang + view_width/2))
                ang -= 2*np.pi
        else:
            tight_slots = []
            ang = view_angle % (2*np.pi)
            
            while ang < _theta[-1] + view_width / 2:
                tight_slots.append((ang - view_width/2, ang + view_width/2))
                ang += 2*np.pi
                
            normal_slots = []
            ang = view_angle + np.pi
            while ang < _theta[-1] + view_width / 2:
                normal_slots.append((ang - view_width/2, ang + view_width/2))
                ang += 2*np.pi
        
        for ts in tight_slots:
            idxs = np.where((_theta >= ts[0]) & (_theta <= ts[1]))[0]
            r_tight.extend(_r[idxs])
            theta_tight.extend(_theta[idxs])
            
        for ns in normal_slots:
            idxs = np.where((_theta >= ns[0]) & (_theta <= ns[1]))[0]
            r_normal.extend(_r[idxs])
            theta_normal.extend(_theta[idxs])
            
        return np.array(r_tight), np.array(r_normal), np.array(theta_tight), np.array(theta_normal)

    r_tight, r_normal, theta_tight, theta_normal = get_divergence(org_theta, org_r, theta_tight_pred, VIEW_WIDTH, inv=spiral_dict.get('inversion', False))
    
    # Ensure tight side has lower mean radius
    if np.mean(r_tight) > np.mean(r_normal):
        r_tight, r_normal = r_normal, r_tight
        theta_tight, theta_normal = theta_normal, theta_tight
        theta_tight_pred += np.pi
        tt_pred_unrec += np.pi
    
    ## PLOTTING VALUES
    # Gaussian Fit
    tight_side_sigma = np.std(r_tight)
    tight_side_mean = np.mean(r_tight)
    normal_side_sigma = np.std(r_normal)
    normal_side_mean = np.mean(r_normal)
    
    # Angles
    mid_theta = theta_tight_pred
    lb_theta = mid_theta - VIEW_WIDTH / 2
    ub_theta = mid_theta + VIEW_WIDTH / 2
    
    # New r
    r_max = max(r) * 1.1
    r_plot = np.linspace(0, r_max, 200)
    
    # Figure
    if plot_recovered:
        fig, (ax_hist, ax_dummy, ax_r) = plt.subplots(
            1, 3, figsize=(18, 6),
            gridspec_kw={'wspace': 0.3}
        )
        
        # Replace second axes with polar
        ax_polar = fig.add_subplot(1, 3, 2, projection='polar')
    else:
        fig, (ax_hist, ax_dummy) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={'wspace': 0.2}
        )
        
        # Replace second axes with polar
        ax_polar = fig.add_subplot(1, 2, 2, projection='polar')
    ax_dummy.remove()    
    
    
    ## POLAR PLOT
    # Processed spiral
    # ax_polar.plot(theta, r, linewidth=1.2, color='gray', linestyle='--', alpha=0.5, label='Procesirana spirala')
    # ax_polar.plot(np.full_like(r, tt_pred_unrec), r, linestyle="--", color="navy", alpha=0.3)
    # ax_polar.plot(np.full_like(r, tt_pred_unrec + np.pi), r, linestyle="--", color="navy", alpha=0.3)
    
    ax_polar.plot(org_theta, org_r, linewidth=2, color='black', alpha=0.8, label='Narisana spirala')
    if plot_recovered: ax_polar.plot(theta_rec, r_rec, linewidth=1.5, color='red', alpha=0.6, label='Obnovljena spirala')
    
    print(np.max(org_r))
    # ax_polar.legend(fontsize=10, bbox_to_anchor=(1.15, 1.07))
    
    def draw_angle_band(ax, theta_mid, theta_lb, theta_ub, r, region_color):

        # bounds
        ax.plot(np.full_like(r, theta_lb), r, linestyle="--", color=region_color, alpha=0.6)
        ax.plot(np.full_like(r, theta_ub), r, linestyle="--", color=region_color, alpha=0.6)

        # ---- shaded band ----
        theta = np.concatenate([
            np.full_like(r, theta_lb),
            np.full_like(r, theta_ub)[::-1],
        ])
        r_fill = np.concatenate([r, r[::-1]])

        ax.fill(theta, r_fill, color=region_color, alpha=0.25, linewidth=0)
        
        # central angle
        ax.plot(np.full_like(r, theta_mid), r, linestyle="-", color="navy", alpha=0.75)
    
    # Tigth side
    draw_angle_band(ax_polar, mid_theta, lb_theta, ub_theta, r_plot, region_color='orange')
    # Normal side
    draw_angle_band(ax_polar, mid_theta + np.pi, lb_theta + np.pi, ub_theta + np.pi, r_plot, region_color='forestgreen')

    ax_polar.scatter(theta_tight, r_tight, color='darkorange', s=20, zorder=3, alpha=0.9)
    ax_polar.scatter(theta_normal, r_normal, color='darkgreen', s=20, zorder=3, alpha=0.9)
    
    ax_polar.set_rmax(r_max)
    ax_polar.grid(alpha=0.3)
    if not IS_TIGHT: ax_polar.set_title("IZRAZITA STISNJENOST NI ZAZNANA\n", fontsize=14, color='red')
    
    
    plot_theta = theta_tight_pred % (2 * np.pi)
    if plot_theta < 0:
        plot_theta += 2 * np.pi
    ha = 'left' if (np.pi/2 > plot_theta or plot_theta > 3*np.pi/2) else 'right'

    ax_polar.text(plot_theta, r_max*1.05, f"{plot_theta:.2f}π", color='navy', ha=ha, va='center', fontsize=13)
    
    ax_polar.set_yticklabels([])
    
    # ## HISTOGRAM + GAUSSIANS
    bins = np.linspace(0, r_max, 30)

    def gauss_func(x, mu, sigma):
        coeff = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)
    
    
    bw = (bins[1] - bins[0])

    # --- Tight (right) ---
    x_tight = bins
    gauss_tight = gauss_func(x_tight, tight_side_mean, tight_side_sigma)
    y_tight = gauss_tight * len(r_tight) * bw

    ax_hist.plot(x_tight, y_tight, color="orange", linestyle="--", label="Gaussova distribucija stisnjene strani")
    ax_hist.fill_between(x_tight, y_tight, color="orange", alpha=0.15)

    # --- Normal (left) ---
    # mirror x to negative AND keep x increasing for fill_between
    x_normal = -bins[::-1]  # goes from -max .. 0 (increasing)
    mu_normal_left = -normal_side_mean  # mirror the mean to the left
    gauss_normal = gauss_func(x_normal, mu_normal_left, normal_side_sigma)
    y_normal = gauss_normal * len(r_normal) * bw

    ax_hist.plot(x_normal, y_normal, color="forestgreen", linestyle="--", label="Gaussova distribucija običajne strani")
    ax_hist.fill_between(x_normal, y_normal, color="forestgreen", alpha=0.15)

    ax_hist.axvline(0.0, color="black", linewidth=1, alpha=0.5)
    ax_hist.set_xlim(-bins.max(), bins.max())  # avoid any autoscale weirdness
    ax_hist.set_ylim(0, np.max([y_tight.max(), y_normal.max()]) * 1.4)
    
    ax_hist.set_xlabel("Polmer")
    ax_hist.set_ylabel("Število")
    ax_hist.legend(loc="upper left", fontsize=10)
    ax_hist.grid(alpha=0.3)
    
    ## DISTRIBUTION SIMILARITY
    def hellinger_gaussians(mu1, sigma1, mu2, sigma2, eps=1e-12):
        s1 = max(float(sigma1), eps)
        s2 = max(float(sigma2), eps)

        denom = s1*s1 + s2*s2
        coeff = np.sqrt((2.0*s1*s2) / denom)
        expo = np.exp(-((mu1 - mu2)**2) / (4.0*denom))

        h2 = 1.0 - coeff * expo
        # numerical safety
        h2 = float(np.clip(h2, 0.0, 1.0))
        return np.sqrt(h2)  # in [0,1]
    
    hell = hellinger_gaussians(tight_side_mean, tight_side_sigma, normal_side_mean, normal_side_sigma)
    
    # Set title to histogram
    ax_hist.set_title(f"Napovedani koeficient k/k_tight = {k_pred/k_tight_pred:.3f}", pad=4)
    
    # Overlay colored lines in the reserved space
    ax_hist.text(
        0.02, 0.84,
        f"Stisnjena: μ={tight_side_mean:.1f}, σ={tight_side_sigma:.1f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="orange",
        clip_on=False,
        fontsize=11,
    )

    ax_hist.text(
        0.02, 0.795,
        f"Običajna: μ={normal_side_mean:.1f}, σ={normal_side_sigma:.1f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="forestgreen",
        clip_on=False,
        fontsize=11,
    )

    ax_hist.text(
        0.02, 0.75,
        f"Hellingerjeva razdalja: {hell:.3f}",
        transform=ax_hist.transAxes,
        ha="left", va="bottom",
        color="black",
        clip_on=False,
        fontsize=11,
    )
    
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
    
    if plot_recovered:
        theta_tight_pred = theta_tight_pred % (2 * np.pi)
        tt_pred_unrec = tt_pred_unrec % (2 * np.pi)

        ax_r.scatter(theta, r, color="gray", alpha=0.5, label="Procesirana spirala", s=10)
        ax_r.scatter(org_theta, org_r, color="black", alpha=0.8, label="Narisana spirala", s=10)
        ax_r.scatter(theta_rec, r_rec, color="red", alpha=0.6, label="Obnovljena spirala", s=10)
        
        t = tt_pred_unrec
        while t < theta.max():
            ax_r.plot([t, t], [0, r_max], linestyle="--", color="grey", alpha=0.7)
            t += 2*np.pi
            
        t = theta_tight_pred
        if INVERSION:
            while t > theta_rec.min():
                ax_r.plot([t, t], [0, r_max], linestyle="--", color="red", alpha=0.7)
                t -= 2*np.pi
        else:
            while t < theta_rec.max():
                ax_r.plot([t, t], [0, r_max], linestyle="--", color="red", alpha=0.7)
                t += 2*np.pi
                
        ax_r.set_xlabel("Kot θ (rad)")
        ax_r.set_ylabel("Polmer r")
        ax_r.set_title(f"Napovedani kot = {tt_pred_unrec:.2f}π rad\nProcesirani kot = {theta_tight_pred:.2f}π rad", pad=4)
                
    if save_path == "":
        save_path= "transformer5_results/spiral_tight_divg_plot.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=500)

def main():
    
    MODEL_PATH = "models/" + "sp_trans_20260112_081707.pt"
    
    parser = argparse.ArgumentParser(
        description="Evaluate Spiral Transformer and plot results.",
    )
    
    parser.add_argument(
        "--spiral_path", type=str,
        help="Input spiral path for inference.",
        default="",
    )  
    
    tight_spirals = [
        "testna_mn_5/normal/normal_992.npz",
        "testna_mn_5/tight/tight_8.npz",
        "testna_mn_5/tight/tight_42.npz",
        "testna_mn_5/tight/tight_43.npz",
        "testna_mn_5/tight/tight_44.npz",
        "testna_mn_5/tight/tight_45.npz",
    ]
    tight_spirals = [Path(sp) for sp in tight_spirals]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    input_path = Path(args.spiral_path)
    
    if input_path == Path(""):
        spirals = tight_spirals
        dataset = SpiralDataset([spirals])
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=spiral_collate)

        sample = dataset[0]
        d_emb = sample["seq_emb"].shape[1]
        
        model = SpiralTransformer(
            embed_dim=d_emb,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=512,
            dropout=0.1,
            max_seq_len=1024,
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        run_inference_and_plot(
            model=model,
            dataset=dataset,
            device=device,
            thr_tight=0.5,
            plot_title="Spiral Tightening Prediction",
            plot_type="tight"
        )
    else:
        dataset = SpiralDataset([[input_path]])
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=spiral_collate)
        
        sample = dataset[0]
        d_emb = sample["seq_emb"].shape[1]

        model = SpiralTransformer(
            embed_dim=d_emb,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=512,
            dropout=0.1,
            max_seq_len=1024,
        ).to(device)
        
        # Load the pretrained model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        run_inference_and_plot(
            model=model,
            dataset=dataset,
            device=device,
            thr_tight=0.5,
            plot_title="Spiral Tightening Prediction",
            plot_type="tight"
        )

    
if __name__ == "__main__":
    main()