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

|  Model   |                                                                              Config                                                                              | Dynamic Shape | Batch Inference | Note |
| :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :--: |
|  ESRGAN  |     [esrgan_x4c64b23g32_g1_400k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py)      |       Y       |        Y        |      |
|  ESRGAN  | [esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) |       Y       |        Y        |      |
|  SRCNN   |           [srcnn_x4k915_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py)           |       Y       |        Y        |      |
|   DIM    |     [dim_stage3_v16_pln_1x1_1000k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py)     |       Y       |        Y        |      |
|   GCA    |                [gca_r34_4x10_200k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/gca/gca_r34_4x10_200k_comp1k.py)                |       N       |        Y        |      |
| IndexNet |       [indexnet_mobv2_1x16_78k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k.py)        |       Y       |        Y        |      |

**Notes**:

- *All models above are tested with Pytorch==1.6.0 and onnxruntime==1.5.1*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmedit`.

### Convert ONNX to TensorRT (experimental)

We also provide a script to convert [ONNX](https://github.com/onnx/onnx) model to [TensorRT](https://github.com/NVIDIA/TensorRT) format. Besides, we support comparing the output results between ONNX and TensorRT model.

```bash
python tools/onnx2tensorrt.py
    ${CFG_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    ${INPUT_ONNX} \
    --trt-file ${OUT_TENSORRT} \
    --max-shape INT INT INT INT \
    --min-shape INT INT INT INT \
    --workspace-size INT \
    --fp16 \
    --show \
    --verify \
    --verbose
```

Description of arguments:

- `config` : The path of a model config file.
- `model_type` :The model type of the config file, options: `inpainting`, `mattor`, `restorer`, `synthesizer`.
- `img_path` : The path to input image file.
- `onnx_file` : The path to input ONNX file.
- `--trt-file` : The path of output TensorRT model. If not specified, it will be set to `tmp.trt`.
- `--max-shape` : Maximum shape of model input.
- `--min-shape` : Minimum shape of model input.
- `--workspace-size`: Max workspace size in GiB. If not specified, it will be set to 1 GiB.
- `--fp16`: Determines whether to export TensorRT with fp16 mode. If not specified, it will be set to `False`.
- `--show`: Determines whether to show the output of ONNX and TensorRT. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--verbose`: Determines whether to verbose logging messages while creating TensorRT engine. If not specified, it will be set to `False`.

**Note**: This tool is still experimental. Some customized operators are not supported for now. We only support `restorer` for now. While generating ONNX file of SRCNN, replace 'bicubic' with 'bilinear' in SCRNN model [here](https://github.com/open-mmlab/mmediting/blob/764e6065e315b7d0033762038fcbf0bb1c570d4d/mmedit/models/backbones/sr_backbones/srcnn.py#L40). For TensorRT does not support bicubic interpolation by now and final performance will be weaken by about 4%.

#### List of supported models exportable to TensorRT

The table below lists the models that are guaranteed to be exportable to TensorRT engine and runnable in TensorRT.

| Model  |                                                      Config                                                       | Dynamic Shape | Batch Inference |                         Note                          |
| :----: | :---------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :---------------------------------------------------: |
| ESRGAN | [esrgan_x4c64b23g32_g1_400k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py) |       Y       |        Y        |                                                       |
| ESRGAN | [esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) |       Y       |        Y        |                                                       |
| SRCNN  | [srcnn_x4k915_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py) |       Y       |        Y        | 'bicubic' upsampling must be replaced with 'bilinear' |

**Notes**:

- *All models above are tested with Pytorch==1.8.1,  onnxruntime==1.7.0 and tensorrt==7.2.3.4*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmedit`.

### Evaluate ONNX and TensorRT Models (experimental)

We provide methods to evaluate TensorRT and ONNX models in `tools/deploy_test.py`.

#### Prerequisite

To evaluate ONNX and TensorRT models, onnx, onnxruntime and TensorRT should be installed first. Install `mmcv-full` with ONNXRuntime custom ops and TensorRT plugins follow [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) and [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md).

#### Usage

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${BACKEND} \
    --out ${OUTPUT_FILE} \
    --save-path ${SAVE_PATH} \
    ----cfg-options ${CFG_OPTIONS} \
```

#### Description of all arguments

- `config`: The path of a model config file.
- `model`: The path of a TensorRT or an ONNX model file.
- `backend`: The backend for testing, choose tensorrt or onnxruntime.
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
      <th align="center">TensorRT FP32</th>
      <th align="center">TensorRT FP16</th>
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
        <td align="center">28.2619</td>
        <td align="center">28.2616</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7778</td>
        <td align="center">0.7784</td>
        <td align="center">0.7784</td>
        <td align="center">0.7783</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">24.6328</td>
        <td align="center">24.6290</td>
        <td align="center">24.6290</td>
        <td align="center">24.6274</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.6491</td>
        <td align="center">0.6494</td>
        <td align="center">0.6494</td>
        <td align="center">0.6494</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">26.6531</td>
        <td align="center">26.6532</td>
        <td align="center">26.6532</td>
        <td align="center">26.6532</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7340</td>
        <td align="center">0.7340</td>
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
        <td align="center">30.6307</td>
        <td align="center">30.6305</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8559</td>
        <td align="center">0.8565</td>
        <td align="center">0.8565</td>
        <td align="center">0.8566</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">27.0543</td>
        <td align="center">27.0422</td>
        <td align="center">27.0422</td>
        <td align="center">27.0411</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7447</td>
        <td align="center">0.7450</td>
        <td align="center">0.7450</td>
        <td align="center">0.7449</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">29.3354</td>
        <td align="center">29.3354</td>
        <td align="center">29.3354</td>
        <td align="center">29.3339</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8263</td>
        <td align="center">0.8263</td>
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
        <td align="center">27.2144</td>
        <td align="center">27.2127</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.8099</td>
        <td align="center">0.8106</td>
        <td align="center">0.7782</td>
        <td align="center">0.7781</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">Set14</td>
        <td align="center">PSNR</td>
        <td align="center">25.6486</td>
        <td align="center">25.6367</td>
        <td align="center">24.8613</td>
        <td align="center">24.8599</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7014</td>
        <td align="center">0.7015</td>
        <td align="center">0.6674</td>
        <td align="center">0.6673</td>
    </tr>
    <tr>
        <td align="center" rowspan="2">DIV2K</td>
        <td align="center">PSNR</td>
        <td align="center">27.7460</td>
        <td align="center">27.7460</td>
        <td align="center">26.9891</td>
        <td align="center">26.9862</td>
    </tr>
    <tr>
        <td align="center">SSIM</td>
        <td align="center">0.7854</td>
        <td align="center">0.78543</td>
        <td align="center">0.7605</td>
        <td align="center">0.7604</td>
    </tr>
</table>

**Notes**:

- All ONNX and TensorRT models are evaluated with dynamic shape on the datasets and images are preprocessed according to the original config file.
- This tool is still experimental, and we only support `restorer` for now.
