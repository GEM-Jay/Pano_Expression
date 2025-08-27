# train.py
from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict
# ERP：横向环绕卷积启用工具（我们在 utils/erp_wrap.py 里已实现）
from utils.erp_wrap import enable_horizontal_circular_padding


def parse_erp_cfg(model_cfg):
    """
    读取 ERP 环绕卷积配置。
    支持三种写法：
      1) model: { erp_wrap: true/false }
      2) model: { erp_wrap: { enable: true, only_3x3: true, v_mode: "reflect", v_value: 0.0 } }
      3) 不写则默认启用，仅替换 3×3，v_mode="reflect"
    """
    erp_wrap_cfg = model_cfg.get("erp_wrap", True)

    # 默认
    params = {
        "enable": True,
        "only_3x3": True,
        "v_mode": "reflect",
        "v_value": 0.0,
    }

    if isinstance(erp_wrap_cfg, bool):
        params["enable"] = erp_wrap_cfg
    elif isinstance(erp_wrap_cfg, (dict, OmegaConf)):
        params["enable"] = bool(erp_wrap_cfg.get("enable", True))
        params["only_3x3"] = bool(erp_wrap_cfg.get("only_3x3", True))
        # 兼容两种命名
        params["v_mode"] = str(erp_wrap_cfg.get("v_mode",
                                 erp_wrap_cfg.get("vertical_pad_mode", "reflect")))
        params["v_value"] = float(erp_wrap_cfg.get("v_value",
                                  erp_wrap_cfg.get("vertical_pad_value", 0.0)))
    # 其它类型 => 使用默认

    return params


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_rdeic.yaml")
    args = parser.parse_args()

    # 读取主配置（包含 data / model / lightning 三段）
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)

    # ----------------------- Data & Model -----------------------
    # data：按照 configs/train_rdeic.yaml 里的 data 配置构造（LightningDataModule 或等价对象）
    data_module = instantiate_from_config(config.data)

    # model：从独立的 model 配置文件构造
    model_cfg = OmegaConf.load(config.model.config)
    model = instantiate_from_config(model_cfg)

    # （可选）恢复初始权重
    if config.model.get("resume"):
        ckpt = torch.load(config.model.resume, map_location="cpu")
        load_state_dict(model, ckpt, strict=False)
        print(f"[Load] Resume weights from: {config.model.resume}")

    # ----------------------- ERP 环绕启用 -----------------------
    # 在加载完权重后再做替换：from_conv 会复制已有权重到新模块
    erp_params = parse_erp_cfg(config.model)
    if erp_params["enable"]:
        replaced = enable_horizontal_circular_padding(
            model,
            only_3x3=erp_params["only_3x3"],
            v_mode=erp_params["v_mode"],
            v_value=erp_params["v_value"],
        )
        print(f"[ERP] Horizontal-circular padding enabled: replaced {replaced} Conv2d "
              f"(only_3x3={erp_params['only_3x3']}, v_mode='{erp_params['v_mode']}').")
    else:
        print("[ERP] Horizontal-circular padding disabled.")

    # ----------------------- Trainer & Fit -----------------------
    # callbacks
    callbacks = []
    for cb_cfg in config.lightning.get("callbacks", []):
        callbacks.append(instantiate_from_config(cb_cfg))

    # loggers（如果配置里提供的话）
    loggers = []
    for lg_cfg in config.lightning.get("loggers", []):
        loggers.append(instantiate_from_config(lg_cfg))

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers if len(loggers) > 0 else None,
        **config.lightning.trainer,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
