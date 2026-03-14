"""
GGUF model loader for SkyReels-V3 pipeline.
Adapted from ComfyUI-GGUF (city96, Apache-2.0 license).
Loads GGUF quantized weights and dequantizes on-the-fly during forward pass.
"""

import gc
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gguf
except ImportError:
    raise ImportError(
        "gguf package is required for GGUF model loading. "
        "Install it with: pip install gguf"
    )


# GGML quantization type constants
GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "BF16": 30,
}

# Block sizes and type sizes for quantized formats
QUANT_INFO = {
    # type: (block_size, type_size_bytes)
    GGML_TYPES["Q8_0"]: (32, 34),   # 32 values per block, 2 bytes scale + 32 bytes data
    GGML_TYPES["Q4_0"]: (32, 18),
    GGML_TYPES["Q4_1"]: (32, 20),
    GGML_TYPES["Q5_0"]: (32, 22),
    GGML_TYPES["Q5_1"]: (32, 24),
    GGML_TYPES["Q2_K"]: (256, 84),
    GGML_TYPES["Q3_K"]: (256, 110),
    GGML_TYPES["Q4_K"]: (256, 144),
    GGML_TYPES["Q5_K"]: (256, 176),
    GGML_TYPES["Q6_K"]: (256, 210),
}


def split_block_dims(blocks, *args):
    """Split blocks tensor along last dimension into chunks of specified sizes."""
    dims = list(args)
    # Last chunk gets the remainder
    dims.append(blocks.shape[-1] - sum(dims))
    return torch.split(blocks, dims, dim=-1)


# --- Dequantization functions (adapted from ComfyUI-GGUF dequant.py) ---

def dequantize_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8).to(dtype)
    return (d * x)


def dequantize_Q4_0(blocks, block_size, type_size, dtype=None):
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.view(torch.uint8)
    n = qs.shape[-1]
    x0 = (qs & 0x0F).to(dtype) - 8.0
    x1 = (qs >> 4).to(dtype) - 8.0
    x = torch.cat([x0, x1], dim=-1)
    return (d * x)


def dequantize_Q4_1(blocks, block_size, type_size, dtype=None):
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.view(torch.uint8)
    x0 = (qs & 0x0F).to(dtype)
    x1 = (qs >> 4).to(dtype)
    x = torch.cat([x0, x1], dim=-1)
    return (d * x + m)


def dequantize_Q5_0(blocks, block_size, type_size, dtype=None):
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = qh.view(torch.int32)
    qs = qs.view(torch.uint8)
    n = qs.shape[-1]
    x0 = (qs & 0x0F).to(dtype)
    x1 = (qs >> 4).to(dtype)
    # Extract high bits
    bits = torch.arange(block_size, device=qh.device, dtype=torch.int32)
    xh = ((qh >> bits.view(1, -1)) & 1).to(dtype) * 16.0
    xh0, xh1 = xh[..., :n], xh[..., n:]
    x0 = x0 + xh0 - 16.0
    x1 = x1 + xh1 - 16.0
    x = torch.cat([x0, x1], dim=-1)
    return (d * x)


def dequantize_Q5_1(blocks, block_size, type_size, dtype=None):
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = qh.view(torch.int32)
    qs = qs.view(torch.uint8)
    n = qs.shape[-1]
    x0 = (qs & 0x0F).to(dtype)
    x1 = (qs >> 4).to(dtype)
    bits = torch.arange(block_size, device=qh.device, dtype=torch.int32)
    xh = ((qh >> bits.view(1, -1)) & 1).to(dtype) * 16.0
    xh0, xh1 = xh[..., :n], xh[..., n:]
    x0 = x0 + xh0
    x1 = x1 + xh1
    x = torch.cat([x0, x1], dim=-1)
    return (d * x + m)


def dequantize_Q4_K(blocks, block_size, type_size, dtype=None):
    # Q4_K: 256 values per super-block, 144 bytes
    # Layout: d (2 bytes) | dmin (2 bytes) | scales (12 bytes) | qs (128 bytes)
    d, dmin, scales_raw, qs = split_block_dims(blocks, 2, 2, 12)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    s = scales_raw.view(torch.uint8)

    # Decode 8 scales and 8 mins from 12-byte packed format (get_scale_min_k4)
    sc = torch.zeros(*s.shape[:-1], 8, dtype=dtype, device=s.device)
    mn = torch.zeros(*s.shape[:-1], 8, dtype=dtype, device=s.device)

    # j < 4: sc[j] = s[j] & 63, mn[j] = s[j+4] & 63
    sc[..., 0] = (s[..., 0] & 63).to(dtype)
    sc[..., 1] = (s[..., 1] & 63).to(dtype)
    sc[..., 2] = (s[..., 2] & 63).to(dtype)
    sc[..., 3] = (s[..., 3] & 63).to(dtype)
    mn[..., 0] = (s[..., 4] & 63).to(dtype)
    mn[..., 1] = (s[..., 5] & 63).to(dtype)
    mn[..., 2] = (s[..., 6] & 63).to(dtype)
    mn[..., 3] = (s[..., 7] & 63).to(dtype)

    # j >= 4: sc[j] = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)
    #         mn[j] = (s[j+4] >> 4) | ((s[j] >> 6) << 4)
    sc[..., 4] = ((s[..., 8] & 0x0F) | ((s[..., 0] >> 6) << 4)).to(dtype)
    sc[..., 5] = ((s[..., 9] & 0x0F) | ((s[..., 1] >> 6) << 4)).to(dtype)
    sc[..., 6] = ((s[..., 10] & 0x0F) | ((s[..., 2] >> 6) << 4)).to(dtype)
    sc[..., 7] = ((s[..., 11] & 0x0F) | ((s[..., 3] >> 6) << 4)).to(dtype)
    mn[..., 4] = ((s[..., 8] >> 4) | ((s[..., 4] >> 6) << 4)).to(dtype)
    mn[..., 5] = ((s[..., 9] >> 4) | ((s[..., 5] >> 6) << 4)).to(dtype)
    mn[..., 6] = ((s[..., 10] >> 4) | ((s[..., 6] >> 6) << 4)).to(dtype)
    mn[..., 7] = ((s[..., 11] >> 4) | ((s[..., 7] >> 6) << 4)).to(dtype)

    # Dequantize 4-bit values: 128 bytes -> 256 values in 8 sub-blocks of 32
    qs = qs.view(torch.uint8)
    q0 = (qs[..., :32] & 0xF).to(dtype)    # sub-block 0
    q1 = (qs[..., :32] >> 4).to(dtype)      # sub-block 1
    q2 = (qs[..., 32:64] & 0xF).to(dtype)   # sub-block 2
    q3 = (qs[..., 32:64] >> 4).to(dtype)     # sub-block 3
    q4 = (qs[..., 64:96] & 0xF).to(dtype)   # sub-block 4
    q5 = (qs[..., 64:96] >> 4).to(dtype)     # sub-block 5
    q6 = (qs[..., 96:] & 0xF).to(dtype)     # sub-block 6
    q7 = (qs[..., 96:] >> 4).to(dtype)       # sub-block 7

    # result[sub_i] = d * sc[i] * q[i] - dmin * mn[i]
    r0 = d * sc[..., 0:1] * q0 - dmin * mn[..., 0:1]
    r1 = d * sc[..., 1:2] * q1 - dmin * mn[..., 1:2]
    r2 = d * sc[..., 2:3] * q2 - dmin * mn[..., 2:3]
    r3 = d * sc[..., 3:4] * q3 - dmin * mn[..., 3:4]
    r4 = d * sc[..., 4:5] * q4 - dmin * mn[..., 4:5]
    r5 = d * sc[..., 5:6] * q5 - dmin * mn[..., 5:6]
    r6 = d * sc[..., 6:7] * q6 - dmin * mn[..., 6:7]
    r7 = d * sc[..., 7:8] * q7 - dmin * mn[..., 7:8]

    return torch.cat([r0, r1, r2, r3, r4, r5, r6, r7], dim=-1)


def dequantize_Q6_K(blocks, block_size, type_size, dtype=None):
    # Q6_K: 256 values per block, 210 bytes
    # Layout: ql (128) | qh (64) | scales (16) | d (2)
    ql, qh, scales, d = split_block_dims(blocks, 128, 64, 16)
    d = d.view(torch.float16).to(dtype).unsqueeze(-1)  # (*, 1, 1)
    scales = scales.view(torch.int8).to(dtype)
    ql = ql.view(torch.uint8)
    qh = qh.view(torch.uint8)

    # First 128 values (ql[0:64], qh[0:32])
    q1 = (ql[..., :32] & 0xF) | ((qh[..., :32] & 3) << 4)
    q2 = (ql[..., 32:64] & 0xF) | (((qh[..., :32] >> 2) & 3) << 4)
    q3 = (ql[..., :32] >> 4) | (((qh[..., :32] >> 4) & 3) << 4)
    q4 = (ql[..., 32:64] >> 4) | ((qh[..., :32] >> 6) << 4)

    # Second 128 values (ql[64:128], qh[32:64])
    q5 = (ql[..., 64:96] & 0xF) | ((qh[..., 32:64] & 3) << 4)
    q6 = (ql[..., 96:128] & 0xF) | (((qh[..., 32:64] >> 2) & 3) << 4)
    q7 = (ql[..., 64:96] >> 4) | (((qh[..., 32:64] >> 4) & 3) << 4)
    q8 = (ql[..., 96:128] >> 4) | ((qh[..., 32:64] >> 6) << 4)

    # Combine all 256 values, reshape to (*, 16 sub-blocks, 16 values)
    x = torch.cat([q1, q2, q3, q4, q5, q6, q7, q8], dim=-1)
    x = x.reshape(*ql.shape[:-1], 16, 16).to(dtype)
    scales = scales.reshape(*scales.shape[:-1], 16, 1)
    x = d * scales * (x - 32.0)
    return x.reshape(*x.shape[:-2], -1)


# Dispatch table
DEQUANTIZE_FUNCTIONS = {
    GGML_TYPES["Q8_0"]: dequantize_Q8_0,
    GGML_TYPES["Q4_0"]: dequantize_Q4_0,
    GGML_TYPES["Q4_1"]: dequantize_Q4_1,
    GGML_TYPES["Q5_0"]: dequantize_Q5_0,
    GGML_TYPES["Q5_1"]: dequantize_Q5_1,
    GGML_TYPES["Q4_K"]: dequantize_Q4_K,
    GGML_TYPES["Q6_K"]: dequantize_Q6_K,
}


def dequantize_tensor(data, tensor_type, tensor_shape, dtype=torch.bfloat16):
    """Dequantize a GGML tensor to a regular PyTorch tensor."""
    if tensor_type in (GGML_TYPES["F32"], GGML_TYPES["F16"], GGML_TYPES["BF16"]):
        # Already float, just reshape
        if tensor_type == GGML_TYPES["F32"]:
            return data.view(torch.float32).reshape(tensor_shape).to(dtype)
        elif tensor_type == GGML_TYPES["F16"]:
            return data.view(torch.float16).reshape(tensor_shape).to(dtype)
        elif tensor_type == GGML_TYPES["BF16"]:
            return data.view(torch.bfloat16).reshape(tensor_shape).to(dtype)

    if tensor_type not in QUANT_INFO:
        raise ValueError(f"Unsupported GGML quantization type: {tensor_type}")

    block_size, type_size = QUANT_INFO[tensor_type]
    rows = data.reshape(-1, type_size)

    if tensor_type not in DEQUANTIZE_FUNCTIONS:
        # Fallback to gguf library dequantization (slow, numpy-based)
        logging.warning(f"Using slow numpy dequant for type {tensor_type}")
        import numpy as np
        np_data = data.numpy()
        result = gguf.quants.dequantize(np_data, tensor_type)
        return torch.from_numpy(result).reshape(tensor_shape).to(dtype)

    dequant_fn = DEQUANTIZE_FUNCTIONS[tensor_type]
    result = dequant_fn(rows, block_size, type_size, dtype=dtype)
    return result.reshape(tensor_shape)


class _GGMLWeightProxy:
    """Proxy object that provides .device and .dtype for GGML quantized weights."""
    def __init__(self, ggml_weight):
        self._ggml_weight = ggml_weight

    @property
    def device(self):
        return self._ggml_weight.device

    @property
    def dtype(self):
        return torch.bfloat16


class GGMLLinear(nn.Module):
    """Linear layer that stores GGML quantized weights and dequantizes on-the-fly."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # These will be set by load_gguf_into_model
        self.ggml_weight = None
        self.ggml_type = None
        self.ggml_shape = None
        self.compute_dtype = torch.bfloat16
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Proxy that provides .device and .dtype for compatibility."""
        return _GGMLWeightProxy(self.ggml_weight)

    def _apply(self, fn, recurse=True):
        """Override to move ggml_weight along with other parameters."""
        super()._apply(fn, recurse)
        if self.ggml_weight is not None:
            self.ggml_weight = fn(self.ggml_weight)
        return self

    def forward(self, x):
        weight = dequantize_tensor(
            self.ggml_weight, self.ggml_type, self.ggml_shape, dtype=x.dtype
        )
        return F.linear(x, weight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ggml_type={self.ggml_type}"
        )


class GGMLConv3d(nn.Module):
    """Conv3d that stores GGML quantized weights and dequantizes on-the-fly."""

    def __init__(self, original_conv):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.ggml_weight = None
        self.ggml_type = None
        self.ggml_shape = None
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Proxy that provides .device and .dtype for compatibility."""
        return _GGMLWeightProxy(self.ggml_weight)

    def _apply(self, fn, recurse=True):
        """Override to move ggml_weight along with other parameters."""
        super()._apply(fn, recurse)
        if self.ggml_weight is not None:
            self.ggml_weight = fn(self.ggml_weight)
        return self

    def forward(self, x):
        weight = dequantize_tensor(
            self.ggml_weight, self.ggml_type, self.ggml_shape, dtype=x.dtype
        )
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def load_gguf_state_dict(gguf_path):
    """
    Load a GGUF file and return a dict mapping tensor names to (data, tensor_type, shape).
    Tensor names have common prefixes stripped for compatibility with WanModel state dict.
    """
    logging.info(f"Loading GGUF file: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # Read architecture metadata
    arch = None
    for field_name, field in reader.fields.items():
        if field_name == "general.architecture":
            arch = str(bytes(field.parts[-1]), encoding="utf-8")
            break
    logging.info(f"GGUF architecture: {arch}")

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        tensor_type = int(tensor.tensor_type)
        shape = list(reversed(tensor.shape.tolist()))  # GGUF stores in reverse order

        # Strip common prefixes used by GGUF converters
        for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Read tensor data as raw bytes
        data = torch.from_numpy(tensor.data.copy())

        state_dict[name] = {
            "data": data,
            "type": tensor_type,
            "shape": shape,
        }

    logging.info(f"Loaded {len(state_dict)} tensors from GGUF")
    return state_dict


def _replace_modules_recursive(model, gguf_sd, prefix="", device="cuda"):
    """Recursively replace nn.Linear and nn.Conv3d with GGML versions where weights exist."""
    replaced = 0
    for name, module in list(model.named_children()):
        full_name = f"{prefix}{name}" if prefix else name
        weight_key = f"{full_name}.weight"

        if isinstance(module, nn.Linear) and weight_key in gguf_sd:
            info = gguf_sd[weight_key]
            new_module = GGMLLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None
            )
            new_module.ggml_weight = info["data"].to(device)
            new_module.ggml_type = info["type"]
            new_module.ggml_shape = info["shape"]

            # Load bias if present
            bias_key = f"{full_name}.bias"
            if bias_key in gguf_sd and module.bias is not None:
                bias_data = dequantize_tensor(
                    gguf_sd[bias_key]["data"],
                    gguf_sd[bias_key]["type"],
                    gguf_sd[bias_key]["shape"],
                    dtype=torch.bfloat16,
                )
                new_module.bias = nn.Parameter(bias_data.to(device))
            elif module.bias is not None:
                new_module.bias = nn.Parameter(module.bias.data.to(device))

            setattr(model, name, new_module)
            replaced += 1

        elif isinstance(module, nn.Conv3d) and weight_key in gguf_sd:
            info = gguf_sd[weight_key]
            new_module = GGMLConv3d(module)
            new_module.ggml_weight = info["data"].to(device)
            new_module.ggml_type = info["type"]
            new_module.ggml_shape = info["shape"]

            bias_key = f"{full_name}.bias"
            if bias_key in gguf_sd and module.bias is not None:
                bias_data = dequantize_tensor(
                    gguf_sd[bias_key]["data"],
                    gguf_sd[bias_key]["type"],
                    gguf_sd[bias_key]["shape"],
                    dtype=torch.bfloat16,
                )
                new_module.bias = nn.Parameter(bias_data.to(device))
            elif module.bias is not None:
                new_module.bias = nn.Parameter(module.bias.data.to(device))

            setattr(model, name, new_module)
            replaced += 1
        else:
            # Recurse into children
            child_replaced = _replace_modules_recursive(
                module, gguf_sd, prefix=f"{full_name}.", device=device
            )
            replaced += child_replaced

    return replaced


def _get_module_by_path(model, path):
    """Get a module from model by dot-separated path."""
    parts = path.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def _set_module_by_path(model, path, new_module):
    """Set a module in model by dot-separated path."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def load_gguf_into_model(model, gguf_path, device="cuda"):
    """
    Load GGUF weights into a WanModel, replacing Linear/Conv3d layers with GGML versions.
    Non-quantized parameters (norms, modulations, embeddings) are loaded directly.
    Memory-efficient: processes tensors incrementally and frees GGUF data as we go.
    """
    logging.info(f"Loading GGUF file: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # Read architecture metadata
    arch = None
    for field_name, field in reader.fields.items():
        if field_name == "general.architecture":
            arch = str(bytes(field.parts[-1]), encoding="utf-8")
            break
    logging.info(f"GGUF architecture: {arch}")

    # Build a map of model parameter names to their expected shapes
    model_param_shapes = {}
    for name, param in model.named_parameters():
        model_param_shapes[name] = param.shape

    # Track which Linear/Conv3d modules need GGML replacement
    # Key: module path -> (weight_tensor_data, weight_type, weight_shape, bias_data_or_None)
    module_replacements = {}
    direct_params = {}  # Non-weight/bias params to load directly

    total_tensors = len(reader.tensors)
    logging.info(f"Processing {total_tensors} tensors from GGUF")

    for tensor in reader.tensors:
        name = tensor.name
        tensor_type = int(tensor.tensor_type)
        shape = list(reversed(tensor.shape.tolist()))

        # Strip common prefixes
        for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer."]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Determine if this is a weight of a Linear/Conv3d module
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_path, param_name = parts
            if param_name == "weight" and tensor_type in QUANT_INFO:
                # This is a quantized weight — mark for GGML replacement
                if module_path not in module_replacements:
                    module_replacements[module_path] = {}
                # Keep quantized data on CPU for now, move to CUDA during replacement
                module_replacements[module_path]["weight_data"] = torch.from_numpy(tensor.data.copy())
                module_replacements[module_path]["weight_type"] = tensor_type
                module_replacements[module_path]["weight_shape"] = shape
                continue
            elif param_name == "bias" and module_path in module_replacements:
                # Bias for a module being replaced
                data = torch.from_numpy(tensor.data.copy())
                module_replacements[module_path]["bias_data"] = data
                module_replacements[module_path]["bias_type"] = tensor_type
                module_replacements[module_path]["bias_shape"] = shape
                continue

        # Everything else: dequantize and store for direct loading
        data = torch.from_numpy(tensor.data.copy())
        direct_params[name] = {
            "data": data,
            "type": tensor_type,
            "shape": shape,
        }

    # Step 1: Replace Linear/Conv3d modules with GGML versions
    replaced = 0
    for module_path, info in module_replacements.items():
        try:
            original = _get_module_by_path(model, module_path)
        except (AttributeError, IndexError):
            logging.warning(f"Module not found in model: {module_path}")
            continue

        if isinstance(original, nn.Linear):
            new_module = GGMLLinear(
                original.in_features, original.out_features,
                bias=original.bias is not None
            )
            new_module.ggml_weight = info["weight_data"].to(device)
            new_module.ggml_type = info["weight_type"]
            new_module.ggml_shape = info["weight_shape"]

            if "bias_data" in info and original.bias is not None:
                bias = dequantize_tensor(
                    info["bias_data"], info["bias_type"], info["bias_shape"],
                    dtype=torch.bfloat16,
                )
                new_module.bias = nn.Parameter(bias.to(device))
            elif original.bias is not None:
                new_module.bias = nn.Parameter(torch.zeros(original.out_features, device=device))

            _set_module_by_path(model, module_path, new_module)
            replaced += 1

        elif isinstance(original, nn.Conv3d):
            new_module = GGMLConv3d(original)
            new_module.ggml_weight = info["weight_data"].to(device)
            new_module.ggml_type = info["weight_type"]
            new_module.ggml_shape = info["weight_shape"]

            if "bias_data" in info and original.bias is not None:
                bias = dequantize_tensor(
                    info["bias_data"], info["bias_type"], info["bias_shape"],
                    dtype=torch.bfloat16,
                )
                new_module.bias = nn.Parameter(bias.to(device))

            _set_module_by_path(model, module_path, new_module)
            replaced += 1

    # Free replacement data
    del module_replacements
    gc.collect()
    logging.info(f"Replaced {replaced} modules with GGML quantized versions")

    # Step 2: Load remaining parameters directly
    loaded_direct = 0
    for name, info in direct_params.items():
        # Find the parameter in the model
        try:
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = _get_module_by_path(model, parts[0])
                param_name = parts[1]
            else:
                parent = model
                param_name = name

            if not hasattr(parent, param_name):
                continue

            current = getattr(parent, param_name)
            if not isinstance(current, (nn.Parameter, torch.Tensor)):
                continue

            tensor = dequantize_tensor(
                info["data"], info["type"], info["shape"], dtype=torch.bfloat16
            )
            # Keep modulation parameters in float32 for stability
            if "modulation" in name:
                tensor = tensor.float()
            # Fix shape mismatch: GGUF may store [6, dim] but model expects [1, 6, dim]
            if tensor.shape != current.shape:
                try:
                    tensor = tensor.reshape(current.shape)
                except RuntimeError:
                    logging.warning(
                        f"Shape mismatch for {name}: GGUF={tensor.shape}, "
                        f"model={current.shape}, skipping"
                    )
                    continue

            new_param = nn.Parameter(tensor.to(device), requires_grad=False)
            setattr(parent, param_name, new_param)
            loaded_direct += 1
        except (AttributeError, IndexError) as e:
            logging.debug(f"Could not load {name}: {e}")

    del direct_params
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"Loaded {loaded_direct} parameters directly (norms, embeddings, etc.)")

    return model
