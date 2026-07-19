"""Paper-5 style CQMLP starter model with optional Brevitas layers."""

from __future__ import annotations

import torch.nn as nn

try:
    import brevitas.nn as qnn
except ImportError:  # pragma: no cover - fallback for environments without brevitas
    class _QuantIdentity(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class _QuantLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

    class _QuantReLU(nn.ReLU):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _QnnFallback:
        QuantIdentity = _QuantIdentity
        QuantLinear = _QuantLinear
        QuantReLU = _QuantReLU

    qnn = _QnnFallback()


class CQMLP(nn.Module):
    """Compact quantized MLP skeleton for 40-byte CAN window inputs."""

    def __init__(self, input_dim: int = 40, num_classes: int = 4, bit_width: int = 2):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(
            input_dim, 256, bias=True, weight_bit_width=bit_width
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        self.fc2 = qnn.QuantLinear(256, 128, bias=True, weight_bit_width=bit_width)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        self.fc3 = qnn.QuantLinear(128, 64, bias=True, weight_bit_width=bit_width)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        self.fc4 = qnn.QuantLinear(64, 32, bias=True, weight_bit_width=bit_width)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        self.fc5 = qnn.QuantLinear(32, num_classes, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x
