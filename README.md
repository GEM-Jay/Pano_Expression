# 训练配置
注意配置里，batch大小最好能整除数据集，否则会训练崩溃。
## 配置清单
压缩：
```yaml
#configs/dataset/lic_train.yaml
dataset:
  params:
      resize_cfg:
      enabled: True
      target_w: 1024
data_loader:
  batch_size: 24
```
```yaml
#configs/dataset/lic_valid.yaml
data_loader:
  batch_size: 12
```
```yaml
#configs/model/rdeic.yaml
params:
  is_refine: false
  l_guide_weight: 0.1  
  l_bpp_weight: 4 
  
  vae_blend_enabled: false  
  train_blend_net: false 
  extend_k_pix: 0 
```
```yaml
#configs/train_rdeic.yaml
model:
  resume: ~
  
lightning:
  trainer:
    devices: [?]
    default_root_dir: ./logs/independent/起个名字
    
  callbacks:
    - target: model.callbacks.ModelCheckpoint
      params:
         monitor: "val/psnr"
         mode: "max"
         save_top_k: 5
         save_last: true
         auto_insert_metric_name: false
      filename: "ep{epoch:03d}-gs{global_step}-psnr{val_psnr:.3f}"
      every_n_epochs: 1
```
diffusion训练阶段：
```yaml
#configs/dataset/lic_train.yaml
dataset:
  params:
    resize_cfg:
      enabled: True
      target_w: 512

data_loader:
  batch_size: 12
```
```yaml
#configs/dataset/lic_valid.yaml
data_loader:
  batch_size: 8
```
```yaml
#configs/model/rdeic.yaml
params:
  is_refine: true
  l_guide_weight: 0.1 (和训练阶段差不多即可） 
  l_bpp_weight: 4 （同上）
  
  vae_blend_enabled: true  
  train_blend_net: true 
  extend_k_pix: 64 
```
```yaml
#configs/train_rdeic.yaml
model:
  resume: /data/xyy/Lucas/PanULIC/weight/step1_0.002/last.ckpt（真实第一阶段模型位置）
  
lightning:
  trainer:
    devices: [?]
    default_root_dir: ./logs/independent/起个名字
  callbacks:
    - target: model.callbacks.ModelCheckpoint
      params:
         monitor: "val/score_combo"
         mode: "max"
         save_top_k: 5
         save_last: true
         auto_insert_metric_name: false
         filename: "ep{epoch:03d}-gs{global_step}-combo{val_score_combo:.3f}"
         every_n_epochs: 1         # 仅在验证后基于指标保存
```

## 第一阶段（压缩）
第一步：修改训练配置
```yaml
# configs/train_rdeic.yaml
trainer:
    accelerator: gpu
    devices: [3] —— 选择gpu
    default_root_dir: ./logs/independent/seamless_step1_2 —— 修改log文件名字
```
控制第一阶段压缩模型的比特率。
```yaml
# configs/model/rdeic.yaml
params:
  is_refine: false
  learning_rate: 2e-5
  l_guide_weight: 1.2
  l_bpp_weight: 3
```
训练配置，第一阶段不做任何的处理只做随机滚动
```yaml
#configs/dataset/lic_train.yaml
dataset:
  target: dataset.licdataset.LICDataset
  params:
    file_list: ./datalists/train.list
    out_size: 0          # 不裁剪整图
    crop_type: none
    use_hflip: True       # ERP 不做水平翻转
    use_rot: False         # ERP 不旋转

    resize_cfg:
      enabled: True
      target_w: 1024     # 与 valid 保持一致；显存允许可改为 768/1024/1536/2048（必须是 64 的倍数）

batch_transform:
  target: dataset.batch_transform.RollThenExtendBatchTransform
  params:
    roll_max_frac: 0.2
    extend_frac: 0   # 拓宽 ~10% （按 W 对齐到 64 的倍数）
    align: 0
    keys: ["jpg"]

data_loader:
  batch_size: 32
  num_workers: 8
  shuffle: true
  drop_last: true
  pin_memory: true

```
验证阶段配置
```yaml
#configs/dataset/lic_valid.yaml
dataset:
  target: dataset.licdataset.LICDataset
  params:
    file_list: ./datalists/valid.list
    out_size: 0
    crop_type: none
    use_hflip: False
    use_rot: False

    resize_cfg:            # 与训练集统一，避免分布偏移
      enabled: true
      target_w: 2048        # 与训练配置一致

data_loader:
  batch_size: 16
  num_workers: 8
  shuffle: false
  drop_last: false
  pin_memory: true

batch_transform:
  target: dataset.batch_transform.TailExtendBatchTransform
  params:
    extend_frac: 0
    align: 0
    keys: ["jpg"]
```
## 第二阶段（Diffusion）
在第二阶段的pipeline：

整图 latent 编码 → 量化得到 c_latent → 在 latent 里做“左侧头部复制扩宽” → 在扩宽 latent 上跑扩散（带 ControlNet）→ VAE 解码出宽度 W0+K_pix 的图 → 像素域只在 [0:K] 左带里用右尾巴 [W0:W0+K] 做单侧融合 → 再裁回原宽 W0 做 MSE/LPIPS 和指标。

第二阶段配置项，先将模式改到第二阶段：
```yaml
#configs/model/rdeic.yaml
params:
  #is_refine: 开启第二阶段，fixed_step建议保持5
  is_refine: true
```

添加第一个检查点的模型（已经训练好的压缩模型）：
```yaml
#configs/train_rdeic.yaml
data:
  target: dataset.data_module.DataModule
  params:
    # 训练配置代码
    train_config: ./configs/dataset/lic_train.yaml
    # 评估配置代码
    val_config: ./configs/dataset/lic_valid.yaml

model:
  config: ./configs/model/rdeic.yaml
   # 启动二阶段的检查点，一阶段设为 "~"
  resume: /data/xyy/Lucas/PanULIC/weight/step1_0.002/last.ckpt
  
  trainer:
    devices: [5]    —— 记得改空闲gpu
    default_root_dir: ./logs/independent/seamless_step2_0.002   ——命名为第二阶段
```

修改extend_k_pix（做latent的拓宽）：
```yaml
#configs/model/rdeic.yaml
params:
  # 5 个无缝相关开关
  vae_blend_enabled: true  # 是否在VAE解码的时候开启融合、第一阶段关闭
  train_blend_net: true   # 不联训则用默认余弦窗，true为开启，false为关、第一阶段关闭
  extend_k_pix: 64  #第一阶段写成0，这是在第二阶段的latent侧拓宽，会对应取等价的latent宽度
```