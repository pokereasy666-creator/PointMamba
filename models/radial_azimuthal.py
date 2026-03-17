import torch


@torch.inference_mode()
def radial_encode(coord: torch.Tensor) -> torch.Tensor:
    """Order points by squared Euclidean distance from origin."""
    return (coord ** 2).sum(dim=-1).to(torch.float64)


@torch.inference_mode()
def azimuthal_encode(coord: torch.Tensor) -> torch.Tensor:
    """Order points by azimuthal angle around the Z-axis."""
    return torch.atan2(coord[:, 1], coord[:, 0]).to(torch.float64)
