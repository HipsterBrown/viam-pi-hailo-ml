# `pi-hailo-ml` module

This [module](https://docs.viam.com/registry/modular-resources/) implements the [`rdk:service:mlmodel` API](https://docs.viam.com/appendix/apis/services/mlmodel/) for the [Raspberry Pi AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit) and [Raspberry Pi AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus).

With this model, you can perform efficient AI inferencing, such as real-time image classification and object detection, while using minimum CPU resources compared to inferencing directly on the CPU.

## Requirements

This module assumes you are using the [AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#ai-kit) or [AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html#ai-hat-plus) attached to the Raspberry Pi 5:

It also assumes you are using at least version Bookworm of the Raspberry Pi OS.

## Configure your pi-hailo-ml MLModel service

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add `hipsterbrown:pi-hailo-ml` to your machine](https://docs.viam.com/configure/#services).

### Attributes

The following attributes are available for `hipsterbrown:pi-hailo-ml:hailo-rt` mlmodel service:

| Name    | Type   | Required?    | Default | Description |
| ------- | ------ | ------------ | ------- | ----------- |
| `model` | string * | Optional     | 'mobilenet_v2' (classification) or 'ssd_mobilenetv2_fpnlite_320x320_pp' (object detection) ** | Which pre-compiled ML model to use with the selected task |
| `labels_path` | string | Optional | 'assets/imagenet_labels.txt' (classification) or 'assets/coco_labels.txt' (object detection) ** | Path to plain text file with the list of associated image labels for the model |
| `default_minimum_confidence` | number | Optional | 0.55 | Number between 0 and 1 as a minimum percentage of confidence for the returned outputs from the model |

_* This value must be one of the included pre-compiled models, see more below._

_** The default value depends on the configured task._

**Model Selection:**

While this module tries to select the best general model for each task, feel free to experiment with the various included models to see if another works better for your use case.

The following models can be used:

- efficientnet_bo
- efficientnet_lite0
- efficientnetv2_b0
- efficientnetv2_b1
- efficientnetv2_b2
- higherhrnet_coco
- levit_128s
- mnasnet1.0
- mobilenet_v2
- mobilevit_xs
- mobilevit_xxs
- regnetx_002
- regnety_002
- regnety_004
- resnet18
- shufflenet_v2_x1_5
- squeezenet1.0


### Example configuration

```json
{
    "model": "nanodet_plus_416x416_pp"
}
```

### Next steps

Compile other ML models you've trained or found from a model zoo like [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending). https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment

_Guide coming soon._

## Troubleshooting

