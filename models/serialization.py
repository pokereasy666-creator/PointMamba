"""
Point, z-order and h-order are copied form PointCept
https://github.com/Pointcept/Pointcept
"""
import torch
from .hilbert import encode as hilbert_encode_
from .radial_azimuthal import radial_encode, azimuthal_encode
from addict import Dict

class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="hilbert", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        hilbert_orders = [o for o in order if o in ("hilbert", "hilbert-trans")]
        if hilbert_orders:
            if "grid_coord" not in self.keys():
                assert {"grid_size", "coord"}.issubset(self.keys())
                self["grid_coord"] = torch.div(
                    self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
                ).int()

            if depth is None:
                depth = int(self.grid_coord.max()).bit_length()
            self["serialized_depth"] = depth
            assert depth * 3 + len(self.offset).bit_length() <= 63
            assert depth <= 16

        code = [
            encode(self.grid_coord if o in ("hilbert", "hilbert-trans") else None,
                   self.batch, depth, order=o,
                   coord=self.get("coord", None)) for o in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="hilbert", coord=None):
    if order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "radial":
        assert coord is not None, "radial encoding requires continuous coordinates"
        code = radial_encode(coord)
    elif order == "azimuthal":
        assert coord is not None, "azimuthal encoding requires continuous coordinates"
        code = azimuthal_encode(coord)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        if order in ("hilbert", "hilbert-trans"):
            code = batch << depth * 3 | code
        else:
            max_range = code.max() - code.min() + 1.0
            code = code + batch.to(torch.float64) * max_range * 2.0
    return code

def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)

