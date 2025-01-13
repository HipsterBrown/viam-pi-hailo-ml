# `pi-hailo-ml` module

This [module](https://docs.viam.com/registry/modular-resources/) implements the [`rdk:service:vision` API](https://docs.viam.com/appendix/apis/services/vision/) for the [Raspberry Pi AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit) and [Raspberry Pi AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus).

With this model, you can perform efficient AI inferencing, such as real-time image classification and object detection, while using minimum CPU resources compared to inferencing directly on the CPU.

## Requirements

This module assumes you are using the [AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit) or [AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus) attached to the Raspberry Pi 5:

It also assumes you are using at least version Bookworm of the Raspberry Pi OS.

## Configure your pi-hailo-ml Vision service

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add `hipsterbrown:pi-hailo-ml` to your machine](https://docs.viam.com/configure/#services).

### Attributes

The following attributes are available for `hipsterbrown:vision:hailo-rt` vision service:

| Name    | Type   | Required?    | Default | Description |
| ------- | ------ | ------------ | ------- | ----------- |
| `model` | string * | Optional     | 'yolov8s_h8l' | Which pre-compiled ML model to use |
| `labels_path` | string | Optional | 'assets/coco.txt' | Path to plain text file with the list of associated image labels for the model |
| `default_minimum_confidence` | number | Optional | 0.55 | Number between 0 and 1 as a minimum percentage of confidence for the returned outputs from the model |

_* This value must be one of the included pre-compiled models, see more below._

**Model Selection:**

While this module tries to select the best general model, feel free to experiment with the various included models to see if another works better for your use case.

You can learn more about the available models in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

The following models can be used:

- yolov6n_h8
- yolov6n_h8l
- yolov8s_h8
- yolov8s_h8l
- yolox_s_leaky_h8l_rpi


### Example configuration

```json
{
    "model": "yolox_s_leaky_h8l_rpi"
}
```

### Next steps

Compile other ML models you've trained or found from a model zoo like [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending). https://github.com/hailo-ai/hailo-rpi5-examples?tab=readme-ov-file#hailo-dataflow-compiler-dfc

_Guide coming soon._

Add support for custom model paths to use any supported models from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

## Troubleshooting

**Mis-matching Hailo firmware and drivers**

If you see an error related to the Hailo firmware (`hailofw` or `hailo-dkms`) version not matching the runtime drivers (`hailort` or `python3-hailort`), install the correct version of the software from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/).
You should search for "Vision Processors", selecting the following form options:

- "Archive" instead of "Latest releases"
- "HailoRT" in "Software Sub-Package"
- "ARM64" for "Architecture"
- "Linux" for "OS"
- Whichever Python version is on your device, most likely 3.11

Download the "HailoRT - PCIe driver Ubuntu package (deb)", "HailoRT - Python package (whl)", and "HailoRT - Ubuntu package (deb)" of the version you want, then copy them over to your device.

Install the `.deb` packages using `apt`:

```console
sudo apt install ./path/to/package.deb
```

Install the Python `.whl` package using the system's `pip` command:

```console
sudo python3 -m pip install ./path/to/package.whl --break-system-packages
```

The `--break-system-package` flag is used to override the recommendation about installing packages in virtual environments and replace the system's version of the `pythonr-hailort` package.

You may need to remove this module from your machine configuration, save, then re-add it to use the new system packages on your device.
