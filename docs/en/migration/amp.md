# Migration of AMP Training

In 0.x, MMEditing do not support AMP training for the entire forward process.
Instead, users must use `auto_fp16` decorator to warp the specific submodule and convert the parameter of submodule to fp16.
This allows for fine-grained control of the model parameters, but is more cumbersome to use.
In addition, users need to handle operations such as scaling of the loss function during the training process by themselves.

MMagic 1.x use `AmpOptimWrapper` provided by MMEngine.
In `AmpOptimWrapper.update_params`, gradient scaling and `GradScaler` updating is automatically performed.
And in `optim_context` context manager, `auto_cast` is applied to the entire forward process.

Specifically, the difference between the 0.x and 1.x is as follows:

<table class="docutils">
<thead>
  <tr>
    <th> 0.x version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
# config
runner = dict(fp16_loss_scaler=dict(init_scale=512))
```

```python
# code
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
        # get data from data_batch
        inputs = data_batch['img']
        output = self.demo_network(inputs)

        optimizer.zero_grad()
        loss, log_vars = self.get_loss(data_dict_)

        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
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
# config
optim_wrapper = dict(
    constructor='OptimWrapperConstructor',
    generator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # use amp wrapper
        loss_scale='dynamic'),
    discriminator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # use amp wrapper
        loss_scale='dynamic'))
```

```python
# code
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
        # get data from data_batch
        data = self.data_preprocessor(data, True)
        inputs = data['inputs']

        with optim_wrapper.optim_context(self.discriminator):
            output = self.demo_network(inputs)
        loss_dict = self.get_loss(output)
        # use parse_loss provide by `BaseModel`
        loss, log_vars = self.parse_loss(loss_dict)
        optimizer_wrapper.update_params(loss)

        return log_vars
```

</td>

</tr>
</thead>
</table>

To avoid user modifications to the configuration file, MMagic provides the `--amp` option in `train.py`, which allows the user to start AMP training without modifying the configuration file.
Users can start AMP training by following command:

```bash
bash tools/dist_train.sh CONFIG GPUS --amp

# for slurm users
bash tools/slurm_train.sh PARTITION JOB_NAME CONFIG WORK_DIR --amp
```
