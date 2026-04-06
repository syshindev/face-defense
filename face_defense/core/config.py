from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Union[str, Path], overrides: Optional[List[str]] = None,) -> DictConfig:
    # Load a YAML config and apply CLI-style overrides
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)

    # Inject standard path variables for interpolation
    path_vars = OmegaConf.create({
        "project_root": str(_PROJECT_ROOT),
        "configs": str(_PROJECT_ROOT / "configs"),
        "weights": str(_PROJECT_ROOT / "weights"),
        "data": str(_PROJECT_ROOT / "data"),
        "outputs": str(_PROJECT_ROOT / "outputs"),
        "third_party": str(_PROJECT_ROOT / "third_party"),
    })
    cfg = OmegaConf.merge(path_vars, cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def merge_configs(*configs: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    # Merge multiple configs with later ones taking precedence
    return OmegaConf.merge(*configs)


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    # Convert a DictConfig to a plain Python dict
    return OmegaConf.to_container(cfg, resolve=True)
