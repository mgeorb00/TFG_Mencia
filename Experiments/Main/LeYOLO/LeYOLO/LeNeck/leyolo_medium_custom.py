"""
LeYOLO-Medium Personalizado — Modificaciones para máxima precisión (mAP)
=========================================================================
Modificaciones implementadas:
  1. SEBlock         → atención de canal en el backbone (Squeeze-and-Excitation)
  2. BiFPN Neck      → fusión bidireccional de features multiescala con pesos aprendibles
  3. Attention Head  → cabeza DNiN con atención espacial depthwise

Uso:
    from leyolo_medium_custom import build_custom_leyolo_medium
    model = build_custom_leyolo_medium(num_classes=80)

Integración con Ultralytics (fine-tuning desde pesos oficiales):
    Ver sección al final del archivo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUES BASE (igual que LeYOLO original)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Convolución depthwise + pointwise, bloque base de LeYOLO."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class InvertedBottleneck(nn.Module):
    """Inverted Bottleneck con expansion ratio=2, como en LeYOLO."""
    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super().__init__()
        mid_ch = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # Expansión pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # Depthwise espacial
            nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # Proyección pointwise
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


# ─────────────────────────────────────────────────────────────────────────────
# MODIFICACIÓN 1: SE BLOCK — Atención de canal en el backbone
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Aprende a ponderar la importancia relativa de cada canal.
    Parámetros adicionales: ~2 * (in_ch / reduction)^2  → coste mínimo.
    Ganancia esperada en mAP: +0.5 a +1.5 puntos.
    """
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        squeezed = max(1, in_ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, squeezed, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(squeezed, in_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class InvertedBottleneckSE(nn.Module):
    """Inverted Bottleneck + SE Block (Modificación 1)."""
    def __init__(self, in_ch, out_ch, stride=1, expansion=2, se_reduction=4):
        super().__init__()
        self.ib = InvertedBottleneck(in_ch, out_ch, stride, expansion)
        self.se = SEBlock(out_ch, se_reduction)
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.se(self.ib.conv(x))
        return x + out if self.use_residual else out


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE LeYOLO-Medium con SE Blocks
# ─────────────────────────────────────────────────────────────────────────────

class LeYOLOMediumBackbone(nn.Module):
    """
    Backbone LeYOLO-Medium con SE Blocks añadidos tras cada etapa.
    Produce features en tres escalas: P3 (1/8), P4 (1/16), P5 (1/32).
    Canales: P3=96, P4=192, P5=384  (configuración Medium)
    """
    def __init__(self):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),  # /2
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )

        # Etapa 1 → /4
        self.stage1 = nn.Sequential(
            InvertedBottleneckSE(24, 32, stride=2),
            InvertedBottleneckSE(32, 32),
            InvertedBottleneckSE(32, 32),
        )

        # Etapa 2 → /8  (P3)
        self.stage2 = nn.Sequential(
            InvertedBottleneckSE(32, 96, stride=2),
            InvertedBottleneckSE(96, 96),
            InvertedBottleneckSE(96, 96),
        )

        # Etapa 3 → /16 (P4)
        self.stage3 = nn.Sequential(
            InvertedBottleneckSE(96, 192, stride=2),
            InvertedBottleneckSE(192, 192),
            InvertedBottleneckSE(192, 192),
            InvertedBottleneckSE(192, 192),
        )

        # Etapa 4 → /32 (P5)
        self.stage4 = nn.Sequential(
            InvertedBottleneckSE(192, 384, stride=2),
            InvertedBottleneckSE(384, 384),
            InvertedBottleneckSE(384, 384),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # [B, 96,  H/8,  W/8]
        p4 = self.stage3(p3)  # [B, 192, H/16, W/16]
        p5 = self.stage4(p4)  # [B, 384, H/32, W/32]
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────────────────
# MODIFICACIÓN 2: BiFPN NECK — Fusión bidireccional con pesos aprendibles
# ─────────────────────────────────────────────────────────────────────────────

class BiFPNLayer(nn.Module):
    """
    Una capa de BiFPN (Bidirectional Feature Pyramid Network).

    Flujo:
      Top-down:  P5 → P4_td → P3_out
      Bottom-up: P3_out → P4_out → P5_out

    Los pesos de fusión son aprendibles y se normalizan con softmax
    (Fast Normalized Fusion de EfficientDet).

    Ganancia esperada frente a FPN estándar: +1 a +2 mAP puntos.

    NOTA: is_first=True solo en la primera capa — las proyecciones de canales
    (96/192/384 → channels) se aplican una única vez. Las capas siguientes
    reciben ya los canales uniformes y no necesitan proyectar.
    """
    def __init__(self, channels=96, is_first=False):
        super().__init__()
        self.ch = channels
        self.is_first = is_first

        # Proyecciones de entrada: solo en la primera capa BiFPN
        if is_first:
            self.proj_p3 = nn.Conv2d(96,  channels, 1, bias=False)
            self.proj_p4 = nn.Conv2d(192, channels, 1, bias=False)
            self.proj_p5 = nn.Conv2d(384, channels, 1, bias=False)

        # Convoluciones de fusión (depthwise separable para mantener eficiencia)
        self.fuse_p4_td  = DepthwiseSeparableConv(channels, channels)
        self.fuse_p3_out = DepthwiseSeparableConv(channels, channels)
        self.fuse_p4_out = DepthwiseSeparableConv(channels, channels)
        self.fuse_p5_out = DepthwiseSeparableConv(channels, channels)

        # Pesos aprendibles para cada fusión (fast normalized fusion)
        self.w_p4_td  = nn.Parameter(torch.ones(2))  # [p5_up, p4]
        self.w_p3_out = nn.Parameter(torch.ones(2))  # [p4_td, p3]
        self.w_p4_out = nn.Parameter(torch.ones(3))  # [p4, p4_td, p3_up]
        self.w_p5_out = nn.Parameter(torch.ones(2))  # [p5, p4_up]

        self.eps = 1e-4

    def _fuse(self, weights, *features):
        """Fusión con pesos normalizados (Fast Normalized Fusion)."""
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        return sum(wi * fi for wi, fi in zip(w, features))

    def forward(self, p3, p4, p5):
        # Proyectar a canales uniformes solo en la primera capa
        if self.is_first:
            p3 = self.proj_p3(p3)
            p4 = self.proj_p4(p4)
            p5 = self.proj_p5(p5)

        # ── Top-down ──────────────────────────────────────────────────────────
        # P4_td = fuse(P5↑, P4)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.fuse_p4_td(self._fuse(self.w_p4_td, p5_up, p4))

        # P3_out = fuse(P4_td↑, P3)
        p4_td_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_out = self.fuse_p3_out(self._fuse(self.w_p3_out, p4_td_up, p3))

        # ── Bottom-up ─────────────────────────────────────────────────────────
        # P4_out = fuse(P4, P4_td, P3_out↓)
        p3_dn = F.max_pool2d(p3_out, kernel_size=2, stride=2)
        p4_out = self.fuse_p4_out(self._fuse(self.w_p4_out, p4, p4_td, p3_dn))

        # P5_out = fuse(P5, P4_out↓)
        p4_dn = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.fuse_p5_out(self._fuse(self.w_p5_out, p5, p4_dn))

        return p3_out, p4_out, p5_out


class BiFPNNeck(nn.Module):
    """
    Neck compuesto por N capas BiFPN apiladas.
    Más capas = más capacidad, a costa de más FLOPs.
    Recomendado: num_layers=2 para Medium (equilibrio óptimo).
    """
    def __init__(self, channels=96, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(channels, is_first=(i == 0))  # solo la primera proyecta canales
            for i in range(num_layers)
        ])

    def forward(self, p3, p4, p5):
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────────────────
# MODIFICACIÓN 3: ATTENTION HEAD — DNiN con atención espacial
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Atención espacial ligera: aprende qué regiones del mapa de features
    son más relevantes para la detección.
    Coste: 1 convolución depthwise adicional.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class DNiNAttentionHead(nn.Module):
    """
    Cabeza de detección DNiN (Decoupled Network-in-Network) mejorada
    con atención espacial antes de las ramas de clasificación y regresión.

    Arquitectura:
        features → SpatialAttention
                 ├─ cls_branch → num_classes scores
                 └─ reg_branch → 4 bbox coords

    Parámetros:
        in_ch       : canales de entrada (= channels del BiFPN)
        num_classes : número de clases del dataset
        num_anchors : anclas por celda (1 para anchor-free como LeYOLO)
    """
    def __init__(self, in_ch=96, num_classes=80, num_anchors=1):
        super().__init__()
        self.spatial_attn = SpatialAttention(in_ch)

        # Rama de clasificación
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),  # depthwise
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, num_classes * num_anchors, 1),
        )

        # Rama de regresión bbox
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),  # depthwise
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, 4 * num_anchors, 1),
        )

    def forward(self, x):
        x = self.spatial_attn(x)
        cls_out = self.cls_branch(x)
        reg_out = self.reg_branch(x)
        return cls_out, reg_out


# ─────────────────────────────────────────────────────────────────────────────
# MODELO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

class LeYOLOMediumCustom(nn.Module):
    """
    LeYOLO-Medium Personalizado con las 3 modificaciones:
      ✓ Backbone con SE Blocks
      ✓ BiFPN Neck (2 capas)
      ✓ Cabeza DNiN con atención espacial

    Produce predicciones en 3 escalas (P3, P4, P5).
    Compatible con pérdidas estilo Ultralytics/YOLOv8.
    """
    def __init__(self, num_classes=80, bifpn_channels=96, bifpn_layers=2):
        super().__init__()
        self.backbone = LeYOLOMediumBackbone()
        self.neck     = BiFPNNeck(channels=bifpn_channels, num_layers=bifpn_layers)
        self.head_p3  = DNiNAttentionHead(bifpn_channels, num_classes)
        self.head_p4  = DNiNAttentionHead(bifpn_channels, num_classes)
        self.head_p5  = DNiNAttentionHead(bifpn_channels, num_classes)

    def forward(self, x):
        # Backbone → 3 escalas
        p3, p4, p5 = self.backbone(x)

        # Neck BiFPN
        p3, p4, p5 = self.neck(p3, p4, p5)

        # Cabezas de detección
        cls3, reg3 = self.head_p3(p3)
        cls4, reg4 = self.head_p4(p4)
        cls5, reg5 = self.head_p5(p5)

        # Formato compatible con Ultralytics: [reg+cls] concatenados por escala
        # Ultralytics espera primero regresión (4 canales) y luego clasificación (num_classes)
        return [
            torch.cat([reg3, cls3], dim=1),  # [B, 4+num_classes, 80, 80] → objetos pequeños
            torch.cat([reg4, cls4], dim=1),  # [B, 4+num_classes, 40, 40] → objetos medianos
            torch.cat([reg5, cls5], dim=1),  # [B, 4+num_classes, 20, 20] → objetos grandes
        ]


def build_custom_leyolo_medium(num_classes=80, bifpn_channels=96, bifpn_layers=2):
    """Factory function para construir el modelo personalizado."""
    return LeYOLOMediumCustom(num_classes, bifpn_channels, bifpn_layers)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES: CONTEO DE PARÁMETROS Y FLOPS
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:     {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN CON ULTRALYTICS (fine-tuning desde pesos oficiales)
# ─────────────────────────────────────────────────────────────────────────────

ULTRALYTICS_INTEGRATION = """
# ── Opción A: Fine-tuning completo desde pesos LeYOLO-Medium oficiales ──────
#
# 1. Carga el modelo oficial como punto de partida:
#
from ultralytics import YOLO
base_model = YOLO("weights/LeYOLOMedium.pt")
#
# 2. Extrae los pesos del backbone y cópialos al modelo personalizado:
#
custom = build_custom_leyolo_medium(num_classes=TU_NUM_CLASES)
state  = base_model.model.state_dict()
#
# Cargar solo las capas compatibles (backbone):
compatible = {k: v for k, v in state.items() if k in custom.state_dict()
              and custom.state_dict()[k].shape == v.shape}
custom.load_state_dict(compatible, strict=False)
print(f"Capas cargadas desde pesos oficiales: {len(compatible)}")
#
# 3. Entrenar con tu dataset:
#
import torch
optimizer = torch.optim.AdamW(custom.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
#
# ── Opción B: Integración directa en Ultralytics ────────────────────────────
#
# Registrar los bloques personalizados en el módulo nn de Ultralytics:
#
# En ultralytics/nn/modules/__init__.py añadir:
#   from .leyolo_custom import SEBlock, BiFPNLayer, SpatialAttention, DNiNAttentionHead
#
# Luego definir el modelo en un .yaml propio basado en LeYOLOMedium.yaml
# sustituyendo los bloques C2f/SPPF por los personalizados.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TEST RÁPIDO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LeYOLO-Medium Custom — Test de arquitectura")
    print("=" * 60)

    model = build_custom_leyolo_medium(num_classes=80)
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy)

    print("\n▶ Shapes de salida (formato Ultralytics):")
    for i, tensor in enumerate(out):
        scale = ["P3 (80x80)", "P4 (40x40)", "P5 (20x20)"][i]
        print(f"  {scale}: {tuple(tensor.shape)}  → [B, 4+classes, H, W]")

    print("\n▶ Parámetros del modelo:")
    count_params(model)

    print("\n▶ Modificaciones activas:")
    print("  ✓ SE Blocks en backbone (reduction=4)")
    print("  ✓ BiFPN Neck (2 capas, 96 canales)")
    print("  ✓ DNiN Head con atención espacial")
    print("\n  Integración Ultralytics: ver variable ULTRALYTICS_INTEGRATION")
