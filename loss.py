import torch
import torch.nn.functional as F

# Losses implemented here are for Sigmoid style binary segmentation.
# However, it is not difficult to expand them to multi-class version. See paper for details.

def fidelity_loss(u, u_orig):
    """
    Penalize only where u < u_orig.
    This is because we can always assume u gets larger (or at least stays the same) after convexification.
    This allows increase u but should not decrease it below the original.
    """  
    return torch.mean(torch.clamp(u_orig - u, min=0.0) ** 2)

def C2nd_loss(u,
        margin=1e-3,
        softplus_beta=1.0,
        eps=1e-8,
        grad_threshold=1e-5,):
    """
    Computes the second-order quasi-concavity loss based on:
    Loss = mean(ReLU(Q2 + margin) * ||∇u||),

    where ReLU is relaxed by softplus.

    Args:
        u (Mask Tensor): shape [B, 1, H, W], values usually in [0, 1]
        margin (float): small positive margin added to the curvature term
        softplus_beta (float): controls sharpness of softplus; larger -> closer to ReLU
        eps (float): small constant for numerical stability when computing ||∇u||
        grad_threshold (float): ignores unstable curvature where gradient magnitude is too small

    Returns:
        Tensor: scalar loss
    """
    
    def compute_hessian_derivatives(u):
      # First derivatives (prepend last col/row for same shape)
        ux = torch.diff(u, dim=3, prepend=u[..., :, :, :1])  # ∂u/∂x along width (dim=3)
        uy = torch.diff(u, dim=2, prepend=u[..., :, :1, :])  # ∂u/∂y along height (dim=2)

        # Second derivatives
        uxx = torch.diff(ux, dim=3, prepend=ux[..., :, :, :1])  # ∂²u/∂x²
        uyy = torch.diff(uy, dim=2, prepend=uy[..., :, :1, :])  # ∂²u/∂y²

        # Mixed derivatives
        uxy = torch.diff(ux, dim=2, prepend=ux[..., :, :1, :])  # ∂/∂y(∂u/∂x)
        uyx = torch.diff(uy, dim=3, prepend=uy[..., :, :, :1])  # ∂/∂x(∂u/∂y)

        return ux, uy, uxx, uyy, uxy, uyx
    
    B, C, H, W = u.shape
    assert C == 1, "Only single-channel masks supported."
    
    # Step 1: Compute gradient and Hessian-related finite differences.
    ux, uy, uxx, uyy, uxy, uyx = compute_hessian_derivatives(u)
    uxy_sym = 0.5 * (uxy + uyx)

    # Step 2: Compute Q2
    g2 = ux*ux + uy*uy
    gnorm = torch.sqrt(g2 + eps)
    t1 = -uy/gnorm
    t2 = ux/gnorm
    Q2 = (t1 * t1) * uxx + 2.0 * (t1 * t2) * uxy_sym + (t2 * t2) * uyy
    
    # Step 3: Mask-out small gradient for stability
    grad_mask = (gnorm > grad_threshold).float()
    x = (Q2 + margin) * grad_mask

    # Step 4: Total loss
    violation = F.softplus(x, beta=softplus_beta)
    # ReLU is also fine, violation = F.ReLU(x)
    violation = violation * gnorm

    return violation.mean()

def C1st_loss(u, window_size=7):
    """
    Computes the first-order quasi-concavity loss based on:
    Loss = sum_{j} sum_{i in N(j)} sigmoid(u(i) - u(j)) * ReLU( -∇u(j) · (i - j) )

    Args:
        u (Mask Tensor): shape [B, 1, H, W], values in [0, 1]
        window_size (int): neighborhood size (odd)
    Returns:
        Tensor: scalar loss
    """
    B, C, H, W = u.shape
    assert C == 1, "Only single-channel masks supported."
    pad = window_size // 2
    K = window_size * window_size

    # Step 1: ∇u(j) via finite difference
    ux = torch.diff(u, dim=3, prepend=u[..., :, :, :1])  # ∂u/∂x along width (dim=3)
    uy = torch.diff(u, dim=2, prepend=u[..., :, :1, :])  # ∂u/∂y along height (dim=2)
    grad = torch.cat([uy, ux], dim=1)  # [B, 2, H, W]

    # Step 2: Unfold u to extract N(j)
    patches = F.unfold(u, kernel_size=window_size, padding=pad)  # [B, K, H*W]
    patches = patches.view(B, K, H, W)  # [B, K, H, W]

    # Step 3: u(j) repeated
    u_j = u.expand(B, K, H, W)  # [B, K, H, W]
    diff_ij = patches - u_j  # u(i) - u(j)

    # Step 4: sigmoid weight
    weight = torch.sigmoid(diff_ij)

    # Step 5: (i - j) direction vectors (normalized)
    coords = torch.stack(torch.meshgrid(
        torch.arange(window_size), torch.arange(window_size), indexing='ij'
    ), dim=-1).to(u.device)  # [ws, ws, 2]
    center = torch.tensor([pad, pad], device=u.device)
    direction_vectors = coords - center  # [ws, ws, 2]
    direction_vectors = direction_vectors.view(K, 2).float()
    norm_dirs = direction_vectors / (direction_vectors.norm(dim=1, keepdim=True) + 1e-8)  # [K, 2]

    # Step 6: ∇u(j) dotted with each (i - j)
    grad_j = grad.unsqueeze(1).expand(B, K, 2, H, W)  # [B, K, 2, H, W]
    norm_dirs_exp = norm_dirs.view(K, 2, 1, 1).expand(K, 2, H, W)
    dot = (grad_j * norm_dirs_exp.unsqueeze(0)).sum(dim=2)  # [B, K, H, W]

    # Step 7: ReLU of negative dot
    relu_term = F.relu(-dot)

    # Step 8: total loss
    loss_map = weight * relu_term  # [B, K, H, W]
    loss = loss_map.mean()

    return loss