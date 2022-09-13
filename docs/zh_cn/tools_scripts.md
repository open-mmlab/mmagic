## 实用工具

我们在 `tools/` 目录下提供了很多有用的工具。

### 获取 FLOP 和参数量（实验性）

我们提供了一个改编自 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 的脚本来计算模型的 FLOP 和参数量。

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

例如，

```shell
python tools/get_flops.py configs/resotorer/srresnet.py --shape 40 40
```

你会得到以下的结果。

```
==============================
Input shape: (3, 40, 40)
Flops: 4.07 GMac
Params: 1.52 M
==============================
```

**注**：此工具仍处于实验阶段，我们不保证数字正确。 您可以将结果用于简单的比较，但在技术报告或论文中采用它之前，请仔细检查它。

(1) FLOPs 与输入形状有关，而参数量与输入形状无关。默认输入形状为 (1, 3, 250, 250)。
(2) 一些运算符不计入 FLOP，如 GN 和自定义运算符。
你可以通过修改 [`mmcv/cnn/utils/flops_counter.py`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) 来添加对新运算符的支持。

### 发布模型

在将模型上传到 AWS 之前，您可能需要
(1) 将模型权重转换为 CPU tensors, (2) 删除优化器状态，和
(3) 计算模型权重文件的哈希并将哈希 ID 附加到文件名。

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如，

```shell
python tools/publish_model.py work_dirs/example_exp/latest.pth example_model_20200202.pth
```

最终输出文件名将是 `example_model_20200202-{hash id}.pth`.

### 转换为 ONNX（实验性）

我们提供了一个脚本将模型转换为 [ONNX](https://github.com/onnx/onnx) 格式。 转换后的模型可以通过 [Netron](https://github.com/lutzroeder/netron) 等工具进行可视化。此外，我们还支持比较 Pytorch 和 ONNX 模型之间的输出结果。

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

参数说明：

- `config` : 模型配置文件的路径。
- `checkpoint` : 模型模型权重文件的路径。
- `model_type` : 配置文件的模型类型，选项： `inpainting`, `mattor`, `restorer`, `synthesizer`。
- `image_path` : 输入图像文件的路径。
- `--trimap-path` : 输入三元图文件的路径，用于 mattor 模型。
- `--output-file`: 输出 ONNX 模型的路径。默认为 `tmp.onnx`。
- `--opset-version` : ONNX opset 版本。默认为 11。
- `--show`: 确定是否打印导出模型的架构。默认为 `False`。
- `--verify`: 确定是否验证导出模型的正确性。默认为 `False`。
- `--dynamic-export`: 确定是否导出具有动态输入和输出形状的 ONNX 模型。默认为 `False`。

**注**：此工具仍处于试验阶段。目前不支持某些自定义运算符。我们现在只支持 `mattor` 和 `restorer`。

#### 支持导出到 ONNX 的模型列表

下表列出了保证可导出到 ONNX 并可在 ONNX Runtime 中运行的模型。

|   模型   |                                                                                配置                                                                                 | 动态形状 | 批量推理 | 备注 |
| :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :------: | :--: |
|  ESRGAN  |       [esrgan_x4c64b23g32_g1_400k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py)       |    Y     |    Y     |      |
|  ESRGAN  | [esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) |    Y     |    Y     |      |
|  SRCNN   |            [srcnn_x4k915_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py)             |    Y     |    Y     |      |
|   DIM    |      [dim_stage3_v16_pln_1x1_1000k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py)       |    Y     |    Y     |      |
|   GCA    |                 [gca_r34_4x10_200k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/gca/gca_r34_4x10_200k_comp1k.py)                  |    N     |    Y     |      |
| IndexNet |         [indexnet_mobv2_1x16_78k_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k.py)         |    Y     |    Y     |      |

**注**：

- *以上所有模型均使用 Pytorch==1.6.0 和 onnxruntime==1.5.1*
- 如果您遇到上面列出的模型的任何问题，请创建一个 issue，我们会尽快处理。对于列表中未包含的型号，请尝试自行解决。
- 由于此功能是实验性的并且可能会快速更改，请始终尝试使用最新的 `mmcv` 和 `mmedit`。

### 将 ONNX 转换为 TensorRT（实验性）

我们还提供了将 [ONNX](https://github.com/onnx/onnx) 模型转换为 [TensorRT](https://github.com/NVIDIA/TensorRT) 格式的脚本。 此外，我们支持比较 ONNX 和 TensorRT 模型之间的输出结果。

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

参数说明：

- `config` : 模型配置文件的路径。
- `model_type` :配置文件的模型类型，选项： `inpainting`, `mattor`, `restorer`, `synthesizer`。
- `img_path` : 输入图像文件的路径。
- `onnx_file` : 输入 ONNX 文件的路径。
- `--trt-file` : 输出 TensorRT 模型的路径。默认为 `tmp.trt`。
- `--max-shape` : 模型输入的最大形状。
- `--min-shape` : 模型输入的最小形状。
- `--workspace-size`: 以 GiB 为单位的最大工作空间大小。默认为 1 GiB。
- `--fp16`: 确定是否以 fp16 模式导出 TensorRT。默认为 `False`。
- `--show`: 确定是否显示 ONNX 和 TensorRT 的输出。默认为 `False`。
- `--verify`: 确定是否验证导出模型的正确性。默认为 `False`。
- `--verbose`: 确定在创建 TensorRT 引擎时是否详细记录日志消息。默认为 `False`。

**注**：此工具仍处于试验阶段。 目前不支持某些自定义运算符。 我们现在只支持 `restorer`。 在生成 SRCNN 的 ONNX 文件时，将 SCRNN 模型中的 'bicubic' 替换为 'bilinear' \[此处\](https://github.com/open-mmlab/mmediting/blob/764e6065e315b7d0033762038fcbf0bb1c570d4d/mmedit.bones/modelsrnn py#L40）。 因为 TensorRT 目前不支持 bicubic 插值，最终性能将下降约 4%。

#### 支持导出到 TensorRT 的模型列表

下表列出了保证可导出到 TensorRT 引擎并可在 TensorRT 中运行的模型。

|  模型  |                                                                     配置                                                                      | 动态形状 | 批量推理 |                 备注                  |
| :----: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :------: | :-----------------------------------: |
| ESRGAN | [esrgan_x4c64b23g32_g1_400k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py) |    Y     |    Y     |                                       |
| ESRGAN | [esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) |    Y     |    Y     |                                       |
| SRCNN  | [srcnn_x4k915_g1_1000k_div2k.py](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py)  |    Y     |    Y     | 'bicubic' 上采样必须替换为 'bilinear' |

**注**：

- *以上所有模型均使用 Pytorch==1.8.1、onnxruntime==1.7.0 和 tensorrt==7.2.3.4 进行测试*
- 如果您遇到上面列出的模型的任何问题，请创建一个问题，我们会尽快处理。 对于列表中未包含的型号，请尝试自行解决。
- 由于此功能是实验性的并且可能会快速更改，因此请始终尝试使用最新的 `mmcv` 和 `mmedit`。

### 评估 ONNX 和 TensorRT 模型（实验性）

我们在 `tools/deploy_test.py` 中提供了评估 TensorRT 和 ONNX 模型的方法。

#### 先决条件

要评估 ONNX 和 TensorRT 模型，应先安装 onnx、onnxruntime 和 TensorRT。遵循 [mmcv 中的 ONNXRuntime](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) 和 \[mmcv 中的 TensorRT 插件\](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md%EF%BC%89%E4%BD%BF%E7%94%A8 ONNXRuntime 自定义操作和 TensorRT 插件安装 `mmcv-full`。

#### 用法

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${BACKEND} \
    --out ${OUTPUT_FILE} \
    --save-path ${SAVE_PATH} \
    ----cfg-options ${CFG_OPTIONS} \
```

#### 参数说明：

- `config`: 模型配置文件的路径。
- `model`: TensorRT 或 ONNX 模型文件的路径。
- `backend`: 用于测试的后端，选择 tensorrt 或 onnxruntime。
- `--out`: pickle 格式的输出结果文件的路径。
- `--save-path`: 存储图像的路径，如果没有给出，则不会保存图像。
- `--cfg-options`: 覆盖使用的配置文件中的一些设置，`xxx=yyy` 格式的键值对将被合并到配置文件中。

#### 结果和模型

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

**注**：

- 所有 ONNX 和 TensorRT 模型都使用数据集上的动态形状进行评估，图像根据原始配置文件进行预处理。
- 此工具仍处于试验阶段，我们目前仅支持 `restorer`。
