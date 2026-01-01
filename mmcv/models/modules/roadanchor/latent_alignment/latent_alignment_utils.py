import torch
import torch.nn.functional as F

def _js_divergence_from_feats(e_feat_pairs, m_feat_pairs, eps=1e-8):
    """
    e_feat_pairs, m_feat_pairs: (Nv, D)
    Jensen-Shannon divergence between two latents (softmax -> distributions).
    """
    # (Nv, D)
    p = F.softmax(e_feat_pairs, dim=-1)
    q = F.softmax(m_feat_pairs, dim=-1)
    m = 0.5 * (p + q)

    # KL(p || m)
    kl_pm = (p * (torch.log(p.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
    # KL(q || m)
    kl_qm = (q * (torch.log(q.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)      # (Nv,)
    return js.mean()                # scalar


def latent_alignment_loss_segments(
    feat_e2e,       # (B, Ne, D)  - E2E map segment latent
    seg_pts_e2e,    # (B, Ne, Te, 2)
    seg_mask_e2e,   # (B, Ne) True = padding / invalid

    feat_m2m,       # (B, Nm, D)
    seg_pts_m2m,    # (B, Nm, Tm, 2)
    seg_mask_m2m,   # (B, Nm) True = padding / invalid

    radius=1.0,
    alpha_cos=1.0,
    beta_mse=0.1,
    use_js=False,       # enable the Jensen-Shannon divergence term
    gamma_js=0.05,      # weight of the Jensen-Shannon term
):
    """
    Segment-level latent alignment loss (no pts_mask version).

    - matching is based on segment polyline overlap.
    - the loss is applied to the matched latent pairs.
        L = alpha_cos * (1 - cos) + beta_mse * MSE (+ gamma_js * JS)
    """
    B, Ne, D = feat_e2e.shape
    device = feat_e2e.device

    total_cos = feat_e2e.new_tensor(0.0)
    total_mse = feat_e2e.new_tensor(0.0)
    total_js  = feat_e2e.new_tensor(0.0)   # accumulated Jensen-Shannon term
    pair_count = 0

    for b in range(B):
        # 1) pick the valid segment indices
        valid_e_seg = ~seg_mask_e2e[b]   # (Ne,), True = valid
        valid_m_seg = ~seg_mask_m2m[b]   # (Nm,)

        if valid_e_seg.sum() == 0 or valid_m_seg.sum() == 0:
            continue

        f_e = feat_e2e[b, valid_e_seg]      # (Ne_v, D)
        p_e = seg_pts_e2e[b, valid_e_seg]   # (Ne_v, Te, 2)

        f_m = feat_m2m[b, valid_m_seg]      # (Nm_v, D)
        p_m = seg_pts_m2m[b, valid_m_seg]   # (Nm_v, Tm, 2)

        Ne_v = f_e.shape[0]
        Nm_v = f_m.shape[0]

        # 2) overlap score between segments (Ne_v, Nm_v)
        overlaps = torch.zeros(Ne_v, Nm_v, device=device)

        for i in range(Ne_v):
            pts_e_i = p_e[i]  # (Te, 2)
            if pts_e_i.numel() == 0:
                continue

            for j in range(Nm_v):
                pts_m_j = p_m[j]  # (Tm, 2)
                if pts_m_j.numel() == 0:
                    continue

                dist_ij = torch.cdist(
                    pts_e_i.unsqueeze(0),  # (1, Te, 2)
                    pts_m_j.unsqueeze(0),  # (1, Tm, 2)
                    p=2
                ).squeeze(0)  # (Te, Tm)

                min_e2m, _ = dist_ij.min(dim=1)  # (Te,)
                min_m2e, _ = dist_ij.min(dim=0)  # (Tm,)

                overlap_e = (min_e2m < radius).float().mean()
                overlap_m = (min_m2e < radius).float().mean()

                overlaps[i, j] = 0.5 * (overlap_e + overlap_m)

        # 3) for each e2e segment take the m2m segment with the largest overlap
        max_overlap, best_j = overlaps.max(dim=1)   # (Ne_v,)
        has_match = max_overlap > 0.0

        if not has_match.any():
            continue

        e_idx = torch.where(has_match)[0]      # (Nv,)
        m_idx = best_j[has_match]              # (Nv,)

        e_feat_pairs = f_e[e_idx]              # (Nv, D)
        m_feat_pairs = f_m[m_idx]              # (Nv, D)

        if e_feat_pairs.shape[0] == 0:
            continue

        n_pairs = e_feat_pairs.shape[0]

        # 4) Cosine + MSE
        cos_term = 1.0 - F.cosine_similarity(
            e_feat_pairs, m_feat_pairs, dim=-1
        ).mean()

        mse_term = F.mse_loss(
            e_feat_pairs, m_feat_pairs, reduction='mean'
        )

        total_cos += cos_term * n_pairs
        total_mse += mse_term * n_pairs

        # 5) Jensen-Shannon divergence (optional)
        if use_js:
            js_term = _js_divergence_from_feats(e_feat_pairs, m_feat_pairs)
            total_js += js_term * n_pairs

        pair_count += n_pairs

    if pair_count == 0:
        return feat_e2e.new_tensor(0.0)

    total_cos = total_cos / pair_count
    total_mse = total_mse / pair_count
    loss = alpha_cos * total_cos + beta_mse * total_mse

    if use_js:
        total_js = total_js / pair_count
        loss = loss + gamma_js * total_js

    return loss


def latent_alignment_loss_nn_with_radius(
    feat_e2e,   # (B, N, D)
    pos_e2e,    # (B, N, 2)
    mask_e2e,   # (B, N), True = padding / unused
    feat_m2m,   # (B, K, D)
    pos_m2m,    # (B, K, 2)
    mask_m2m,   # (B, K), True = padding / unused
    radius=2.0,
    alpha_cos=1.0,
    beta_mse=0.1,
    use_js=False,      # enable the Jensen-Shannon divergence term
    gamma_js=0.05,     # weight of the Jensen-Shannon term
):
    """
    1:1 nearest-neighbour matching inside a radius, then alignment.
    - mask_*: True = padding / invalid
    - the ego token is masked out by the caller (e.g. mask_e2e[:, 0] = True)
    """
    B, N, D = feat_e2e.shape
    device = feat_e2e.device

    total_cos = feat_e2e.new_tensor(0.0)
    total_mse = feat_e2e.new_tensor(0.0)
    total_js  = feat_e2e.new_tensor(0.0)
    pair_count = 0

    for b in range(B):
        # 1) valid token indices
        valid_e = ~mask_e2e[b]   # (N,)
        valid_m = ~mask_m2m[b]   # (K,)

        if valid_e.sum() == 0 or valid_m.sum() == 0:
            continue

        feat_e = feat_e2e[b, valid_e]   # (Ne, D)
        pos_e  = pos_e2e[b, valid_e]    # (Ne, 2)

        feat_m = feat_m2m[b, valid_m]   # (Nm, D)
        pos_m  = pos_m2m[b, valid_m]    # (Nm, 2)

        Ne = feat_e.shape[0]
        Nm = feat_m.shape[0]

        # 2) distance matrix (Ne, Nm)
        dist = torch.cdist(pos_e, pos_m, p=2)

        # 3) valid inside the radius
        within_radius = dist < radius
        has_valid = within_radius.any(dim=-1)   # (Ne,)

        if not has_valid.any():
            continue

        masked_dist = dist.clone()
        masked_dist[~within_radius] = 1e6

        # nearest-neighbour indices
        nn_idx = masked_dist.argmin(dim=-1)

        # keep only the tokens that actually have a match inside the radius
        valid_e_indices = torch.where(has_valid)[0]     # (Nv,)
        valid_m_indices = nn_idx[has_valid]            # (Nv,)

        e_feat_pairs = feat_e[valid_e_indices]         # (Nv, D)
        m_feat_pairs = feat_m[valid_m_indices]         # (Nv, D)

        if e_feat_pairs.shape[0] == 0:
            continue

        n_pairs = e_feat_pairs.shape[0]

        # 4) Cosine + MSE
        cos_term = 1.0 - F.cosine_similarity(
            e_feat_pairs, m_feat_pairs, dim=-1
        ).mean()

        mse_term = F.mse_loss(
            e_feat_pairs, m_feat_pairs, reduction='mean'
        )

        total_cos += cos_term * n_pairs
        total_mse += mse_term * n_pairs

        # 5) Jensen-Shannon divergence (optional)
        if use_js:
            js_term = _js_divergence_from_feats(e_feat_pairs, m_feat_pairs)
            total_js += js_term * n_pairs

        pair_count += n_pairs

    if pair_count == 0:
        return feat_e2e.new_tensor(0.0)

    total_cos = total_cos / pair_count
    total_mse = total_mse / pair_count
    loss = alpha_cos * total_cos + beta_mse * total_mse

    if use_js:
        total_js = total_js / pair_count
        loss = loss + gamma_js * total_js

    return loss
