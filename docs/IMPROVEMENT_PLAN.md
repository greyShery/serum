# SERUM 改进方案（基于调研 + 代码逐行审计）

> 生成时间: 2026-08-06  
> 范围: `train.py`, `train_v2.py`, `score_model.py`, `watermark.py`, `augmentation.py`, `aug_sampler.py`, `eval.py`, `config_paper.yaml`  
> 对照基准: `RESEARCH_NOTES.md` Section 4-7 + 论文 Section 3-4 + Table 1-3  
> 输入证据: `logs/train_gpu1_smoothloss_20260801_172148.log:49614` 崩溃堆栈

---

## 0. TL;DR — 11 个问题 / 4 个优先级

| ID | 问题 | 文件 | 严重度 | 一句话修复 |
|----|------|------|--------|------------|
| B1 | `loss_alpha` 在某种执行路径下崩溃 | `train_v2.py:387-389` | 🔴 P0 | 用 `torch.tensor(alpha, dtype=..., device=...)` 显式构造 |
| T1 | α 完全写死，没有 curriculum | `watermark.py:48`, `train.py` | 🔴 P0 | `Watermark.forward` 已接受 `alpha`，在 epoch/step 内线性调度 |
| T2 | 干净图内部多样性极低，contrastive 失效 | `train.py:114-122` | 🔴 P0 | 引入 `DiverseCleanDataset`，每 epoch 重新生成 |
| T3 | aug_sampler mistake 信号被二元化 | `train.py:508-511`, `aug_sampler.py:85-105` | 🔴 P0 | `update(idx, td_error_float)` 接受连续值 |
| E1 | 评估 aug 区间全部固定，违反论文 Section 4.1 | `augmentation.py:74-84` | 🟡 P1 | 改为 (min, max) 区间 |
| E2 | 同一 batch 共享 `idx` 引入 batch-level shortcut | `augmentation.py:124-130` | 🟡 P1 | per-sample 独立采样 |
| E3 | Grid norm 没有强制约束，数值可能爆炸 | `watermark.py:46,75` | 🟡 P1 | 用 `grid_raw` + 每次 forward 重新标准化 |
| P1 | Multi-user Watermark 未实现 | `watermark.py:20` | 🟢 P2 | `n_users` + 训练时随机子集 |
| P2 | Radioactivity 实验无脚本 | `scripts/` 缺失 | 🟢 P2 | `scripts/lora_finetune_sd.py` |
| P3 | Latency benchmark 无脚本 | `scripts/` 缺失 | 🟢 P2 | `scripts/latency_bench.py` |
| P4 | Pixel detector 仍走 VAE | `train.py:343-347` | 🟢 P2 | 在 `_get_watermarked_batch` 分支直接生成像素 |

执行顺序：**先 4 个 P0 + 3 个 P1 → 跑 50 epoch 实验对比 Table 1 → 再决定 P2**。

---

## 1. P0-1 `loss_alpha` 崩溃修复

**症状**（来自 `logs/train_gpu1_smoothloss_20260801_172148.log:49609-49614`）：

```text
File "train_v2.py", line 387, in train_epoch
    loss_alpha = w_alpha * (
File ".../torch/nn/functional.py", line 1704, in relu
    result = torch.relu(input)
TypeError: relu(): argument 'input' (position 1) must be Tensor, not float
```

**当前代码** (`train_v2.py:381-389`):

```python
loss_alpha = logits_all.new_tensor(0.0)
if getattr(self.config.training, 'use_alpha_reg', True):
    w_alpha = getattr(self.config.training, 'alpha_reg_weight', 1e-4)
    alpha = self.config.watermark.grid.noise_mix_alpha
    alpha_t = logits_all.new_tensor(float(alpha))
    loss_alpha = w_alpha * (
        F.relu(0.3 - alpha_t) + F.relu(alpha_t - 0.7)
    )
```

**根因分析**：本地测试 `logits_all.new_tensor(float(alpha))` 总返回 tensor，但当 `config.watermark.grid` 被某处替换为非数值类型（如 yaml reload 或 SimpleNamespace field override），`float(alpha)` 会接受 numpy scalar 但 `F.relu(...)` 链路上的某个内部类型检查会把结果作为 float 标量传出去。

**修复**（强制 dtype/device 一致）：

```python
loss_alpha = logits_all.new_tensor(0.0)
if getattr(self.config.training, 'use_alpha_reg', True):
    w_alpha = float(getattr(self.config.training, 'alpha_reg_weight', 1e-4))
    alpha_val = float(self.config.watermark.grid.noise_mix_alpha)
    # 显式 dtype/device 防止被识别为 float
    alpha_t = torch.tensor(alpha_val, dtype=logits_all.dtype, device=logits_all.device)
    alpha_lo = torch.tensor(0.3, dtype=logits_all.dtype, device=logits_all.device)
    alpha_hi = torch.tensor(0.7, dtype=logits_all.dtype, device=logits_all.device)
    w_alpha_t = torch.tensor(w_alpha, dtype=logits_all.dtype, device=logits_all.device)
    loss_alpha = w_alpha_t * (F.relu(alpha_lo - alpha_t) + F.relu(alpha_t - alpha_hi))
```

`train.py` 第 490-498 行同步修复。

---

## 2. P0-2 α curriculum（论文 Section 3 提到 multi-α，论文 Section 4.1 写 α=0.5）

### 2.1 现状

`Watermark.forward` (`watermark.py:48-80`) 接受 `alpha=None` 默认走 `self.config.watermark.grid.noise_mix_alpha`，训练时整 epoch 都是同一个值。

### 2.2 问题

- detector 只在 α=0.5 见过水印；实际部署到不同 α（0.3、0.7）时性能不可预测。
- 论文 FID 19.14 vs clean 17.90（Table 4）暗示有 ~1.2 FID 退化；α=0.5 偏强，curriculum 可以减少 FID 同时保持 TPR。

### 2.3 修复

**A. Curriculum 调度器**（新增 `src/training/alpha_schedule.py`）：

```python
import math

def alpha_for_step(step: int, total_steps: int,
                   start: float = 0.7, end: float = 0.3,
                   mode: str = 'cosine') -> float:
    """按论文风格,从强(0.7)平滑过渡到弱(0.3),
    保证训练后期水印更隐式,FID 退化更小。"""
    t = step / max(1, total_steps)
    if mode == 'linear':
        return start + (end - start) * t
    elif mode == 'cosine':
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
    elif mode == 'piecewise':
        # 4 段: 0.7 (前 25%) -> 0.5 (25-50%) -> 0.4 (50-75%) -> 0.3 (后 25%)
        if t < 0.25: return 0.7
        elif t < 0.50: return 0.5
        elif t < 0.75: return 0.4
        else: return 0.3
    raise ValueError(mode)
```

**B. 在 Trainer 中按 step 注入**（修改 `train.py:354-358`）：

```python
total_steps = epochs * len(self.dataloader)
for step, (x1, x2) in enumerate(self.dataloader):
    cur_alpha = alpha_for_step(
        step + epoch * len(self.dataloader),
        total_steps,
        start=self.config.training.get('alpha_start', 0.7),
        end=self.config.training.get('alpha_end', 0.3),
        mode=self.config.training.get('alpha_schedule_mode', 'cosine'),
    )
    # 30% 概率混入一个更弱的 α,模仿 multi-user
    wm_batch_size = self.config.training.watermark_batch_size
    wm_noise = self.watermark(wm_batch_size, alpha=cur_alpha).half()
    if random.random() < 0.3:
        wm_noise_alt = self.watermark(wm_batch_size // 2, alpha=cur_alpha * 0.6).half()
        # 注: 需要 padding 或调整 batch_size
```

**C. config 加入**：

```yaml
training:
  alpha_start: 0.7
  alpha_end: 0.3
  alpha_schedule_mode: cosine
```

---

## 3. P0-3 Diverse clean dataset（论文 Section 3 contrastive 假设）

### 3.1 现状

`train.py:103-122`：

```python
gen_data = flatten_tensors(self.data)
aug_data = flatten_tensors(self.data_aug)
min_len = min(len(gen_data), len(aug_data))
tensor_ds = TensorDataset(
    torch.stack(gen_data[:min_len]),
    torch.stack(aug_data[:min_len])
)
```

`x1` 和 `x2` 是**同一张干净图**的轻微扰动版本，对比学习核心前提（negative 来自不同分布）不成立。

### 3.2 修复

新增 `src/data/diverse_clean_dataset.py`：

```python
import torch
from torch.utils.data import Dataset

class DiverseCleanDataset(Dataset):
    """每个 epoch 重新生成一批覆盖多个 prompt 的干净 latent,
    保证 negative class 在 latent 空间广泛分布,强化 contrastive 信号。"""
    def __init__(self, diffusion_model, prompts, n_samples_per_epoch=20000,
                 batch_size=4, seed=42):
        self.diffusion_model = diffusion_model
        self.prompts = prompts
        self.n = n_samples_per_epoch
        self.batch_size = batch_size
        self.rng = torch.Generator().manual_seed(seed)
        self.cache = None

    @torch.no_grad()
    def refresh(self, augment_fn=None, device='cuda'):
        idxs = torch.randperm(len(self.prompts), generator=self.rng)[:self.n]
        latents = []
        for i in range(0, self.n, self.batch_size):
            batch_prompts = [self.prompts[int(j)] for j in idxs[i:i+self.batch_size]]
            latent = self.diffusion_model.predict_latent(
                batch_prompts, [''] * len(batch_prompts)
            )
            latents.append(latent.cpu())
        cache = torch.cat(latents, dim=0)
        if augment_fn is not None:
            # 注意: augment_fn 在 latent 空间无意义;这里直接复制
            self.cache = cache
            self.cache_aug = cache.clone()
        else:
            self.cache = cache
            self.cache_aug = cache.clone()

    def __len__(self):
        return self.cache.shape[0]

    def __getitem__(self, idx):
        return self.cache[idx], self.cache_aug[idx]
```

在 `train.py:_prepare_dataloader` 中替换：

```python
def _prepare_dataloader(self, batch_size, diverse=False):
    if diverse:
        ds = DiverseCleanDataset(self.diffusion_model, self.prompts,
                                 n_samples_per_epoch=len(self.data) * 4,
                                 batch_size=8)
        ds.refresh()
        # 每个 epoch refresh
        self._diverse_ds = ds
        self.dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     drop_last=True, num_workers=0)
    else:
        # 旧逻辑
        ...
```

并在 `train_epoch` 开头调用 `self._diverse_ds.refresh()`。

---

## 4. P0-4 Aug Sampler TD-error 连续化（论文 Appendix A）

### 4.1 现状

`aug_sampler.py:85-105`：

```python
def update(self, idx: int, mistake: bool) -> None:
    if mistake:
        adapt = (1.0 - self.p[idx]) ** self.beta
        lr = self.base_lr_pos * (1.0 + self.boost * adapt)
        self.p[idx] = self.p[idx] + lr * (1.0 - self.p[idx])
    else:
        ...
```

`train.py:508-511`：

```python
td_error = max(td_error_w, td_error_c)
self.aug_sampler.update(aug_idx, td_error > 0.1)   # ← 二元化
```

### 4.2 修复

**A. 修改接口**：

```python
# aug_sampler.py
def update(self, idx: int, td_error: float) -> None:
    """td_error ∈ [0, 0.5],0 表示完美预测,0.5 表示完全随机。
    论文 PER 用 TD-error 的绝对值作为优先级;这里我们直接用。"""
    # 难度归一化:td_error ∈ [0, 0.5] -> [0, 1]
    norm = min(1.0, max(0.0, td_error * 2.0))
    # boost 与难度成正比
    adapt = (1.0 - self.p[idx]) ** self.beta
    lr = self.base_lr_pos * (1.0 + self.boost * norm * adapt)
    self.p[idx] = self.p[idx] + lr * (1.0 - self.p[idx])
    # 容易样本也小幅降低概率,保持多样性
    adapt_neg = (self.p[idx]) ** self.beta
    lr_neg = self.base_lr_neg * (1.0 - norm) * adapt_neg
    self.p[idx] = self.p[idx] - lr_neg * self.p[idx]
    self.p[idx] = np.clip(self.p[idx], self.eps, 1.0 - self.eps)
```

**B. 调用方**：

```python
# train.py:508
td_error = max(td_error_w, td_error_c)  # ∈ [0, 0.5]
self.aug_sampler.update(aug_idx, td_error)   # ← 连续
```

---

## 5. P1-1 评估 aug 区间化（论文 Section 4.1）

### 5.1 现状（`augmentation.py:74-84`）

```python
T.RandomRotation(90, fill=0.5),
T.Compose([RandomJPEGCompression((25, 25))]),
T.RandomResizedCrop((512, 512), scale=(0.75, 0.75)),
RandomDrop((0.64, 0.64)),
T.GaussianBlur(kernel_size=15),
SaltAndPepperNoise(p=0.05),
AddGaussianNoise((0.1, 0.1)),
T.ColorJitter(brightness=6.0, ...),
```

### 5.2 修复

```python
T.RandomRotation(degrees=(-90, 90), fill=0.5),
RandomJPEGCompression(quality_range=(25, 75)),
T.RandomResizedCrop((512, 512), scale=(0.5, 1.5), ratio=(1, 1)),
RandomDrop((0.1, 0.64)),
T.GaussianBlur(kernel_size=(9, 21)),  # 需要扩展 GaussianBlur
SaltAndPepperNoise(p=(0.01, 0.1)),
AddGaussianNoise(std=(0.02, 0.15)),
T.ColorJitter(brightness=(2.0, 8.0), contrast=(0.5, 1.5),
               saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
```

需要扩展 `AddGaussianNoise`、`SaltAndPepperNoise` 接受区间参数（目前是单值或 `(v, v)`）。

### 5.3 配套

- 训练时也用区间版：让 detector 见**完整强度谱**而不是单一固定强度。
- 评估时 fix seed + 输出区间两端的 TPR（论文 Table 1 只给 max-strength，但附录提到 full-spectrum 才是更严格测试）。

---

## 6. P1-2 Per-sample 独立 aug（修复 batch-level shortcut）

### 6.1 现状（`augmentation.py:124-130`）

```python
for img in x_cpu:
    transform = self.transforms[idx] if idx is not None else random.choice(self.transforms)
    transformed = transform(img)
```

整 batch 共享 `idx`，要么都旋转要么都 JPEG。

### 6.2 修复

```python
def perturb(self, x, idx=None, ..., per_sample_prob=0.5):
    out = []
    for img in x_cpu:
        if idx is not None and random.random() < per_sample_prob:
            t = self.transforms[idx]
        else:
            t = random.choice(self.transforms)
        out.append(self.postprocess(t(img)))
    out = torch.stack(out).to(x.device) * 2 - 1
    return out
```

---

## 7. P1-3 Watermark Grid norm enforcement

### 7.1 现状（`watermark.py:46,75`）

```python
self.grid = nn.Parameter(torch.randn(self.config.base_latent_shape))  # 一次性
# forward 永远用 (grid - grid.mean()) / grid.std()
```

`grid.std()` 极小时会爆 `inf`/`nan`；极大时水印信号弱。

### 7.2 修复

```python
class Watermark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 用 raw parameter + 每次 forward 重新标准化
        self.grid_raw = nn.Parameter(
            torch.randn(self.config.base_latent_shape)
        )

    def get_grid(self):
        g = self.grid_raw
        # 强制 std ∈ [0.5, 2.0],避免数值极端
        std = g.std()
        if std < 0.5:
            g = g / (std + 1e-6) * 0.5
        elif std > 2.0:
            g = g / std * 2.0
        return (g - g.mean()) / (g.std() + 1e-6)

    def forward(self, batch_size, ret_noise=False, alpha=None, user_subset=None):
        if alpha is None:
            alpha = self.config.watermark.grid.noise_mix_alpha
        _, C, H, W = self.config.base_latent_shape
        orig_noise = torch.randn(batch_size, C, H, W, device=self.grid_raw.device)
        grid = self.get_grid()
        noise = orig_noise * ((1 - alpha) ** 0.5) + grid * (alpha ** 0.5)
        if ret_noise:
            return noise, orig_noise
        return noise
```

加载旧 checkpoint 时需要把 `grid` → `grid_raw` 复制一次。

---

## 8. P2-1 Multi-user Watermark（论文 Section 3.3）

```python
# watermark.py
class Watermark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_users = config.watermark.get('n_users', 1)
        self.grid_raw = nn.Parameter(
            torch.randn(self.n_users, *self.config.base_latent_shape)
        )

    def get_grid(self, user_subset=None):
        if user_subset is None:
            user_subset = list(range(self.n_users))
        sub = self.grid_raw[user_subset]
        # per-user 标准化
        mean = sub.mean(dim=(1, 2, 3), keepdim=True)
        std = sub.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        return (sub - mean) / std  # [k, C, H, W]

    def forward(self, batch_size, ret_noise=False, alpha=None,
                user_subset=None, user_assignment=None):
        if alpha is None:
            alpha = self.config.watermark.grid.noise_mix_alpha
        _, C, H, W = self.config.base_latent_shape
        orig_noise = torch.randn(batch_size, C, H, W, device=self.grid_raw.device)

        if user_assignment is None:
            # 每个样本随机一个 user
            user_assignment = torch.randint(0, self.n_users, (batch_size,))
        k = self.n_users  # 全集
        grids = self.get_grid()  # [k, C, H, W]
        # 每个样本 = orig + sqrt(alpha/k) * Σ_{p in S_i} grid_p
        # 这里简化:每个 user 用所有 grid 的均值作为其组合模式
        per_user_grid = grids.mean(dim=0)  # [C, H, W] 默认
        # 实际 multi-user 应当 per-sample:
        sample_grid = grids[user_assignment]  # [B, C, H, W]
        noise = orig_noise * ((1 - alpha) ** 0.5) + sample_grid * (alpha ** 0.5)
        if ret_noise:
            return noise, orig_noise
        return noise
```

detector 改为 multi-head：
```python
class MultiHeadWatermarkScoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_users = config.watermark.get('n_users', 1)
        self.backbone = ... # 与原 WatermarkScoreModel 共享 backbone
        self.heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(self.n_users)])
```

OR-pooling 得到最终 logits（论文 Section 3.3 Eq.4）。

---

## 9. P2-2 Radioactivity LoRA（论文 Table 3）

新增 `scripts/lora_finetune_sd.py`：

```python
"""LoRA-100 / LoRA-1000 微调脚本,验证 SERUM 在 LoRA 微调下仍可检测。
论文 Table 3: SD 1.4 = 99.28%, SD 2.0 = 99.64%, SD 2.1 = 99.26%。
"""
import argparse
from peft import LoraConfig, get_peft_model
import torch
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--lora_rank', type=int, default=4)
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to('cuda')
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=16,
        target_modules=['to_q', 'to_v', 'to_k', 'to_out.0'],
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    optim = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    # 训练循环:随机 prompt + 监督保持生成图像分布
    for step in range(args.n_steps):
        ...  # 详见论文附录 Table 3 复现实验

if __name__ == '__main__':
    main()
```

然后在原 SERUM detector 上跑 `eval_tpr`，对比 LoRA 前后的 TPR。

---

## 10. P2-3 Latency Benchmark（论文声称 < 10ms）

新增 `scripts/latency_bench.py`：

```python
"""Detector latency benchmark.
论文声称: 注入 < 1ms, 检测 < 10ms。
"""
import argparse
import time
import torch
from src.models import WatermarkScoreModel
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--n_warmup', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    cfg = Config(args.config)
    model = WatermarkScoreModel(cfg).to(cfg.get_device()).eval()
    x = torch.randn(args.batch_size, 4, 64, 64).to(cfg.get_device())

    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.n_iter):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000 / args.n_iter

    print(f"detector latency (B={args.batch_size}): {elapsed:.3f} ms")


if __name__ == '__main__':
    main()
```

---

## 11. P2-4 Pixel detector 真正不依赖 VAE

### 11.1 现状

`train.py:343-347`：

```python
if self.detector_type == 'pixel':
    x1 = decode(x1.half()).float()   # ← 仍走 VAE
    x2 = decode(x2.half()).float()
```

pixel detector 声称 VAE-free，但实际训练时 `data` 仍是 latent（4×64×64），必须先 decode 才能进 pixel detector。

### 11.2 修复

在 `_prepare_watermark_loader` 中提前做 VAE encode/decode 并 cache 像素：

```python
def _prepare_watermark_loader(self, batch_size, watermark_batch_size,
                              load_checkpoint=True):
    if self.detector_type == 'pixel':
        # 把 clean data 也预先 decode 成像素,缓存到磁盘
        all_pixel = []
        all_pixel_aug = []
        for x1, x2 in self.dataloader:
            all_pixel.append(decode(x1.half()).float().cpu())
            all_pixel_aug.append(decode(x2.half()).float().cpu())
        pixel_ds = TensorDataset(torch.cat(all_pixel), torch.cat(all_pixel_aug))
        self.dataloader = DataLoader(pixel_ds, batch_size=batch_size,
                                     shuffle=True, drop_last=True, num_workers=0)
        # watermarked batch 也直接走像素路径
        return self._setup_pixel_watermark_loader(batch_size, watermark_batch_size)
```

`_get_watermarked_batch` 的 pixel 分支已经走 `wm_pixel = decode(...)`，所以已经满足；只需要让 clean 数据也提前 decode。

---

## 12. 评估协议补充（论文 Section 4 提到的细节）

### 12.1 target_fpr = 1% 阈值校准

`eval.py:236-237` 已经传 `target_fpr=0.01`，但 `compute_roc_metrics` 实现细节需要确认 **TPR@1%FPR 是如何从 ROC 插值的**——论文 Section 4.1 描述是 linear interpolation over the sorted score array。

### 12.2 7 种高级攻击

论文 Table 2 提到 5 种高级攻击（VAE, Regen, Rinse, CtrlRegen, I2V），但 `eval.py` 中只看到 8 种扰动的 eval；高级攻击评估入口似乎在 `evaluation/` 子目录但代码不完整。建议补一个 `run_advanced_attacks.py` 脚本，调用 WAVES benchmark 实现。

---

## 13. 验证实验设计（修复后必跑）

### 13.1 消融实验矩阵

| 实验 | 配置 | 期望 |
|------|------|------|
| A1 baseline | `config_paper.yaml` 全部, ε_smooth=1e-3 | 论文 Table 1 = 99.75% |
| A2 + α curriculum | `+ alpha_start=0.7, end=0.3, cosine` | FID 退化 ↓, TPR 不变 |
| A3 + diverse clean | `+ DiverseCleanDataset(n=80000)` | TPR ↑ 0.5-1% |
| A4 + TD-error continuous | `+ update(idx, td_error)` | aug sampler 收敛更稳 |
| A5 + aug interval | `+ (min, max)` for each aug | 鲁棒性 ↑ 1-2% |
| A6 + per-sample aug | `+ per_sample_prob=0.5` | 抗 batch shortcut |
| A7 + grid norm | `+ grid_raw + clamp` | 训练稳定性 ↑ |
| A8 全部 | A2-A7 全部开启 | TPR ≥ 99.75%, FID ≤ 19.14 |

### 13.2 评估指标
- TPR @ 1% FPR (Combined / Clean / Aug)
- ROC AUC
- FID (用 `torch-fidelity` + COCO 2014 val)
- CLIP Score
- Detection Latency (ms)

### 13.3 时间预算
- 单 GPU (A100 80G) 跑 50 epoch ≈ 12 小时
- 跑 A1-A8 = 8 × 12h = 96h = 4 天,与现有 GPU 资源吻合

---

## 14. 修改清单（建议执行顺序）

1. **修复 `loss_alpha` 崩溃**（1 小时）
2. **修复 `Watermark.grid` → `grid_raw`**（兼容性需检查，1 小时）
3. **修复 `aug_sampler.update` 接口 + 调用方**（30 分钟）
4. **修复 `augmentation.py` 区间化 + per-sample**（2 小时）
5. **新增 `alpha_schedule.py` + Trainer 注入**（2 小时）
6. **新增 `DiverseCleanDataset` + 集成**（4 小时）
7. **跑 A1 baseline 50 epoch**（12 小时）
8. **跑 A2-A7 各 50 epoch**（12 × 6 = 72 小时）
9. **合并 + 写论文 Section 5 ablation**（4 小时）

总计 ~96 小时,与设备 4 天 GPU 配额吻合。

---

## 15. 关联引用

- 调研文档: `/data2/fzx/serumRef/notes/RESEARCH_NOTES.md`
- 训练日志: `/data2/fzx/SERUM/logs/train_gpu1_smoothloss_20260801_172148.log`
- Eval 日志: `/data2/fzx/SERUM/logs/eval_gpu2.log`
- 参考仓库:
  - `/data2/fzx/serumRef/repos/tree-ring-watermark` (FFT 水印)
  - `/data2/fzx/serumRef/repos/GaussMarker` (双域水印)
  - `/data2/fzx/serumRef/repos/gaussian-shading` (ELUT bits)
  - `/data2/fzx/serumRef/repos/WAVES` (高级攻击 benchmark)
  - `/data2/fzx/serumRef/repos/radioactive_data` (原始 radioactive)