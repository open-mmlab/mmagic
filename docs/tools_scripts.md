## Useful tools

We provide lots of useful tools under `tools/` directory.

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

For example,
```shell
python tools/get_flops.py configs/resotorer/srresnet.py --shape 40 40
```
You will get the result like this.

```
==============================
Input shape: (3, 40, 40)
Flops: 4.07 GMac
Params: 1.52 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 250, 250).
(2) Some operators are not counted into FLOPs like GN and custom operators.
You can add support for new operators by modifying [`mmcv/cnn/utils/flops_counter.py`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py).

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/example_exp/latest.pth example_model_20200202.pth
```

The final output filename will be `example_model_20200202-{hash id}.pth`.

### Convert to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron). Besides, we also support comparing the output results between Pytorch and ONNX model.

```bash
python tools/pytorch2onnx.py
    ${CFG_PATH} \
    ${CHECKPOINT_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    --trimap-path ${TRIMAP_PATH} \
    --output-file ${OUTPUT_ONNX} \
    --show \
    --verify \
    --dynamic-export
```

Description of arguments:

- `config` : The path of a model config file.
- `checkpoint` : The path of a model checkpoint file.
- `model_type` :The model type of the config file, options: `inpainting`, `mattor`, `restorer`, `synthesizer`.
- `image_path` : path to input image file.
- `--trimap-path` : path to input trimap file, used in mattor model.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--opset-version` : ONNX opset version, default to 11.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--dynamic-export`: Determines whether to export ONNX model with dynamic input and output shapes. If not specified, it will be set to `False`.

**Note**: This tool is still experimental. Some customized operators are not supported for now. And we only support `mattor` and `restorer` for now.

#### List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|  Model   |                                                                               Config                                                                                | Dynamic Shape | Batch Inference | Note  |
| :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :---: |
|  ESRGAN  |       [esrgan_x4c64b23g32_g1_400k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py)       |       Y       |        Y        |       |
|  ESRGAN  | [esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) |       Y       |        Y        |       |
|  SRCNN   |            [srcnn_x4k915_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py)             |       Y       |        Y        |       |
|   DIM    |      [dim_stage3_v16_pln_1x1_1000k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py)       |       Y       |        Y        |       |
|   GCA    |                 [gca_r34_4x10_200k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/gca/gca_r34_4x10_200k_comp1k.py)                  |       N       |        Y        |       |
| IndexNet |         [indexnet_mobv2_1x16_78k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k.py)         |       Y       |        Y        |       |

**Notes**:

- *All models above are tested with Pytorch==1.6.0 and onnxruntime==1.5.1*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmedit`.

### Evaluate ONNX Models with ONNXRuntime (experimental)

We prepare a tool `tools/deploy_test.py` to evaluate ONNX models with ONNX Runtime backend.

#### Prerequisite

- Install onnx and onnxruntime-gpu

  ```shell
  pip install onnx onnxruntime-gpu
  ```

#### Usage

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${ONNX_FILE} \
    --out ${OUTPUT_FILE} \
    --save-path ${SAVE_PATH} \
    ----cfg-options ${CFG_OPTIONS} \
```

#### Description of all arguments

- `config`: The path of a model config file.
- `model`: The path of an ONNX model file.
- `--out`: The path of output result file in pickle format.
- `--save-path`: The path to store images and if not given, it will not save image.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.

#### Results and Models

<table border="1" class="docutils">
	<tr>
	    <th align="center">Model</th>
	    <th align="center">Config</th>
	    <th align="center">Dataset</th>
	    <th align="center">Metric</th>
	    <th align="center">PyTorch</th>
	    <th align="center">ONNX Runtime</th>
	</tr>
    <tr>
	    <td align="center" rowspan="6">ESRGAN</td>
	    <td align="center" rowspan="6">
            <code>esrgan_x4c64b23g32_g1_400k_div2k.py</code>
        </td>
	    <td align="center" rowspan="2">Set5</td>
        <td align="center">PSNR</td>
        <td align="center">28.2700</td>
        <td align="center">28.2619</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7778</td>
        <td align="center">0.7784</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">24.6328</td>
        <td align="center">24.6290</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.6491</td>
        <td align="center">0.6494</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">26.6531</td>
        <td align="center">26.6532</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7340</td>
        <td align="center">0.7340</td>
    </tr>
    <tr>
	    <td align="center" rowspan="6">ESRGAN</td>
	    <td align="center" rowspan="6">
            <code>esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</code>
        </td>
	    <td align="center" rowspan="2">Set5</td>
        <td align="center">PSNR</td>
        <td align="center">30.6428</td>
        <td align="center">30.6307</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8559</td>
        <td align="center">0.8565</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">27.0543</td>
        <td align="center">27.0422</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7447</td>
        <td align="center">0.7450</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">29.3354</td>
        <td align="center">29.3354</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8263</td>
        <td align="center">0.8263</td>
    </tr>
    <tr>
	    <td align="center" rowspan="6">SRCNN</td>
	    <td align="center" rowspan="6">
            <code>srcnn_x4k915_g1_1000k_div2k.py</code>
        </td>
	    <td align="center" rowspan="2">Set5</td>
        <td align="center">PSNR</td>
        <td align="center">28.4316</td>
        <td align="center">28.4120</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8099</td>
        <td align="center">0.8106</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">25.6486</td>
        <td align="center">25.6367</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7014</td>
        <td align="center">0.7015</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">27.7460</td>
        <td align="center">27.7460</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7854</td>
        <td align="center">0.78543</td>
    </tr>
</table>

**Notes**:

- All ONNX models are evaluated with dynamic shape on the datasets and images are preprocessed according to the original config file.
- This tool is still experimental, and we only support `restorer` for now.
