"""HuggingFace Hub integration for PanSR.

``PanSRHF`` wraps the detectron2 PanSR model in a :class:`huggingface_hub.PyTorchModelHubMixin`
so weights + config can be shared via the Hub:

    # Load
    from pansr.hub import PanSRHF
    model = PanSRHF.from_pretrained("lojze/pansr-lars-swin-l")

    # Publish (after building from a local checkpoint, see tools/export_to_hub.py)
    model.push_to_hub("lojze/pansr-lars-swin-l")

The full (resolved) detectron2 config is stored as a YAML string in ``config.json`` and the
model weights are stored as ``model.safetensors`` (handled by the mixin). Rebuilding the model
requires the PanSR package to be importable (which registers the architecture and the LaRS
metadata used at inference time).
"""
from torch import nn

from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from .config import add_pansr_config

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError as e:  # pragma: no cover
    raise ImportError("huggingface_hub is required for pansr.hub; `pip install huggingface_hub safetensors`.") from e


def cfg_from_yaml(cfg_yaml: str) -> CfgNode:
    """Rebuild a fully-populated PanSR cfg from a dumped YAML string."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_pansr_config(cfg)
    cfg.merge_from_other_cfg(CfgNode.load_cfg(cfg_yaml))
    cfg.MODEL.WEIGHTS = ""  # weights are restored by the Hub mixin, not from a file
    return cfg


class PanSRHF(nn.Module, PyTorchModelHubMixin):
    """A Hub-serializable PanSR model. ``cfg_yaml`` is the dumped detectron2 config."""

    def __init__(self, cfg_yaml: str):
        super().__init__()
        self.cfg = cfg_from_yaml(cfg_yaml)
        self.cfg.freeze()
        self.model = build_model(self.cfg)

    def forward(self, batched_inputs):
        return self.model(batched_inputs)

    @classmethod
    def from_detectron2_cfg(cls, cfg: CfgNode) -> "PanSRHF":
        """Construct a wrapper from a detectron2 cfg (e.g. when exporting a local checkpoint)."""
        return cls(cfg_yaml=cfg.dump())
