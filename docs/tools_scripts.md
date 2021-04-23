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
