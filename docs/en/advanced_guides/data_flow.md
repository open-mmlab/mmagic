# Data flow

- [Data Flow](#data-flow)
  - [Overview of dataflow](#overview-of-data-flow)
  - [Data flow between dataset and model](#data-flow-between-dataset-and-model)
    - [Data from dataloader](#data-from-dataloader)
    - [Data from data preprocessor](#data-from-data-preprocessor)
  - [Data flow between model output and visualizer](#data-flow-between-model-output-and-visualizer)

## Overview of dataflow

The [Runner](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/runner.md) is an "integrator" in MMEngine. It covers all aspects of the framework and shoulders the responsibility of organizing and scheduling nearly all modules, that means the dataflow between all modules also controlled by the `Runner`. As illustrated in the [Runner document of MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html), the following diagram shows the basic dataflow. In this chapter, we will introduce the dataflow and data format convention between the internal modules managed by the [Runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).

<div align="center">
<img src="https://github.com/open-mmlab/mmagic/assets/36404164/fc6ab53c-8804-416d-94cd-332c533a07ad" height="150" />
</div>

In the above diagram, at each training iteration, dataloader loads images from storage and transfer to data preprocessor, data preprocessor would put images to the specific device and stack data to batch, then model accepts the batch data as inputs, finally the outputs of the model would be compute the loss. Since model parameters are freezed when doing evaluation, the model output would be transferred to [Evaluator](./evaluation.md#ioumetric) to compute metrics or seed the data to visualize in [Visualizer](../user_guides/visualization.md).

## Data flow between dataset and model

In this section, we will introduce the data flow passing in the dataset in MMagic. About [dataset](https://mmagic.readthedocs.io/en/latest/howto/dataset.html) and \[transforms\] pipeline (https://mmagic.readthedocs.io/en/latest/howto/transforms.html) related tutorials can be found in the development of guidelines.The data flow between dataloader and model can be generally split into four parts:

1. Read the original information of `XXDataset` collected datasets, and carry out data conversion processing through data transform pipeline;

2. use `PackInputs` to pack data from previous transformations into a dictionar;

3. use `collate_fn` to stack a list of tensors into a batched tensor;

4. use `data preprocessor` to move all these data to target device, e.g. GPUS, and unzip the dictionary from the dataloader
   into a tuple, containing the input images and meta info (`DataSample`).

### Data from transform pipeline

In MMagic, different types of 'XXDataset' load the data (LQ) and label (GT), and perform data transformation in different data preprocessing pipelines, and finally package the processed data into a dictionary through `PackInputs`, which contains all the data required for training and testing iterations.

<table class="docutils">
<thead>
  <tr>
    <th> base_edit_model.py </th>
    <th> base_conditional_gan.py </th>
<tbody>
<tr>
<td valign="top">

```python
@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base model for image and video editing.
    """
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)
```

</td>

<td valign="top">

```python
@MODELS.register_module()
class BaseConditionalGAN(BaseGAN):
    """Base class for Conditional GAM models.
    """
    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs, data_samples)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        pass
        ...
```

</td>

</tr>
</thead>
</table>

For example, in the `BaseEditModel` and `BaseConditionalGAN` models, key input including `img` and `noise` are required. At the same time, the corresponding fields should also be exposed in the configuration file,[cyclegan_lsgan-id0-resnet-in_1xb1-80kiters_facades.py](../../../configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-80kiters_facades.py) as an example,

### Data from dataloader

After receiving a list of dictionary from dataset, `collect_fn` in dataloader will gather `inputs` in each dict
and stack them into a batched tensor. In addition, `data_sample` in each dict will be also collected in a list.
Then, it will output a dict, containing the same keys with those of the dict in the received list. Finally, dataloader
will output the dict from the `collect_fn`. Detailed documentation can be reference [DATASET AND DATALOADER](https://mmengine.readthedocs.io/en/latest/tutorials/dataset.html)。

### Data from data preprocessor

Data preprocessor is the last step to process the data before feeding into the model. It will apply image normalization, convert BGR to RGB and move all data to the target device, e.g. GPUs. After above steps, it will output a tuple, containing a list of batched images, and a list of data samples. Detailed documentation can be reference [data_preprocessor](./data_preprocessor.md)。

## Data flow between model output and visualizer

MMEngine agreed [Abstract Data Element](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/data_element.md) for data transfer Where [data sample](./structures.md) as a more advanced encapsulation can hold more categories of label data. In MMagic, `ConcatImageVisualizer` for visual comparison also controls the visual content through the `add_datasample` function. The specific configuration is as follows.

```python
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
```
