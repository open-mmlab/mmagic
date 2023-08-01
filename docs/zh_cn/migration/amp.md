# 混合精度训练的迁移

在 0.x 版中，MMEditing 并不支持对整体前向过程的混合精度训练。相反，用户必须使用 `auto_fp16` 装饰器来适配特定子模块，然后再将子模块的参数转化成 fp16。这样就可以拥有对模型参数的更细粒度的控制，但是该方法使用起来很繁琐，而且用户需要自己处理一些操作，比如训练过程中损失函数的缩放

MMagic 1.x 版使用了 MMEngine 提供的  `AmpOptimWrapper`，在 `AmpOptimWrapper.update_params` 中，梯度缩放和 `GradScaler` 更新将被自动执行，且在 `optim_context` 上下文管理其中，`auto_cast`被应用到整个前向过程中。

具体来说，0.x 版和 1.x 版之间的差异如下所示：

<table class="docutils">
<thead>
  <tr>
    <th> 0.x 版 </th>
    <th> 1.x 版 </th>
  </tr>
</thead>
<tbody>
<tr>
<td valign="top">

```python
# 配置
runner = dict(fp16_loss_scaler=dict(init_scale=512))
```

```python
# 代码
import torch.nn as nn
from mmedit.models.builder import build_model
from mmedit.core.runners.fp16_utils import auto_fp16


class DemoModule(nn.Module):
    def __init__(self, cfg):
        self.net = build_model(cfg)

    @auto_fp16
    def forward(self, x):
        return self.net(x)

class DemoModel(nn.Module):

    def __init__(self, cfg):
        super().__init__(self)
        self.demo_network = DemoModule(cfg)

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        # 从 data_batch 中获取数据
        inputs = data_batch['img']
        output = self.demo_network(inputs)

        optimizer.zero_grad()
        loss, log_vars = self.get_loss(data_dict_)

        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # 添加 fp16 支持
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_disc, optimizer,
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer)
            loss_scaler.step(optimizer)
        else:
            optimizer.step()
```

</td>

<td valign="top">

```python
# 配置
optim_wrapper = dict(
    constructor='OptimWrapperConstructor',
    generator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # 使用 amp 封装器
        loss_scale='dynamic'),
    discriminator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # 使用 amp 封装器
        loss_scale='dynamic'))
```

```python
# 代码
import torch.nn as nn
from mmagic.registry import MODULES
from mmengine.model import BaseModel


class DemoModule(nn.Module):
    def __init__(self, cfg):
        self.net = MODULES.build(cfg)

    def forward(self, x):
        return self.net(x)

class DemoModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(self)
        self.demo_network = DemoModule(cfg)

    def train_step(self, data, optim_wrapper):
        # 从 data_batch 中获取数据
        data = self.data_preprocessor(data, True)
        inputs = data['inputs']

        with optim_wrapper.optim_context(self.discriminator):
            output = self.demo_network(inputs)
        loss_dict = self.get_loss(output)
        # 使用 `BaseModel` 提供的 parse_loss
        loss, log_vars = self.parse_loss(loss_dict)
        optimizer_wrapper.update_params(loss)

        return log_vars
```

</td>

</tr>
</tbody>
</table>

若要避免用户操作配置文件，MMagic 在 `train.py` 里提供了 `--amp` 选项，其可以让用户在不修改配置文件的情况下启动混合精度训练，用户可以使用以下命令启动混合精度训练：

```bash
bash tools/dist_train.sh CONFIG GPUS --amp

# 对 slurm 用户
bash tools/slurm_train.sh PARTITION JOB_NAME CONFIG WORK_DIR --amp
```
