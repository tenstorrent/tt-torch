
## Supported Models
The following models can be currently run through tt-torch as of Feb 3rd, 2025. Please note, there is a known bug causing incorrect output for some models. The [PCC](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is displayed at the end of each test below. This issue will be addressed soon.

| Model Name | Variant | Pytest Command |
| ---------- | ------- | -------------- |
| Albert | Masked LM Base | tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[full-albert/albert-base-v2-eval] |
| | Masked LM Large | tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[full-albert/albert-large-v2-eval] |
| | Masked LM XLarge | tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[full-albert/albert-xlarge-v2-eval] |
| | Masked LM XXLarge | tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[full-albert/albert-xxlarge-v2-eval] |
| | Sequence Classification Base | tests/models/albert/test_albert_sequence_classification.py::test_albert_sequence_classification[full-textattack/albert-base-v2-imdb-eval] |
| | Token Classification Base | tests/models/albert/test_albert_token_classification.py::test_albert_token_classification[full-albert/albert-base-v2-eval] |
| Autoencoder | (linear) | tests/models/autoencoder_linear/test_autoencoder_linear.py::test_autoencoder_linear[full-eval] |
| DistilBert | base uncased | tests/models/distilbert/test_distilbert.py::test_distilbert[full-distilbert-base-uncased-eval] |
| Llama | 7B | tests/models/llama/test_llama.py::test_llama[full-eval] |
| MLPMixer || tests/models/mlpmixer/test_mlpmixer.py::test_mlpmixer[full-eval] |
| MNist || pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval] |
| MobileNet V2 || tests/models/MobileNetV2/test_MobileNetV2.py::test_MobileNetV2[full-eval] |
|| TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-mobilenet_v2] |
| MobileNet V3 | Small TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-mobilenet_v3_small] |
|| Large TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-mobilenet_v3_large] |
| OpenPose || tests/models/openpose/test_openpose_v2.py::test_openpose_v2[full-eval] |
| Preciever_IO || tests/models/perceiver_io/test_perceiver_io.py::test_perceiver_io[full-eval] |
| ResNet | 18 | tests/models/resnet/test_resnet.py::test_resnet[full-eval] |
|| 18 TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnet18] |
|| 34 TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnet34] |
||  50 | tests/models/resnet50/test_resnet50.py::test_resnet[full-eval] |
|| 50 TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnet50] |
|| 101 TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnet101] |
|| 152 TorchVision | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnet152] |
| Wide ResNet | 50 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-wide_resnet50_2] |
|| 101 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-wide_resnet101_2] |
| ResNext |  50 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnext50_32x4d] |
||  101_32x8d | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnext101_32x8d] |
||  101_64x4d | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-resnext101_64x4d] |
| Regnet | y 400 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_400mf] |
|| y 800 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_800mf] |
|| y 1 6 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_1_6gf] |
|| y 3 2 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_3_2gf] |
|| y 8 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_8gf] |
|| y 16 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_16gf] |
|| y 32 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_y_32gf] |
|| x 400 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_400mf] |
|| x 800 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_800mf] |
|| x 1 6 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_1_6gf] |
|| x 3 2 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_3_2gf] |
|| x 8 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_8gf] |
|| x 16 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_16gf] |
|| x 32 | tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[full-regnet_x_32gf] |
| Yolo | V3 | tests/models/yolov3/test_yolov3.py::test_yolov3[full-eval] |
