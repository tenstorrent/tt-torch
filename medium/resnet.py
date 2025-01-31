# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("ambityga/imagenet100")

# print("Path to dataset files:", path)
# /home/achoudhury/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8
VAL = (
    "/home/achoudhury/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/val.X/"
)

from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
import torch
import torchvision as tv
import onnx
import onnxruntime as ort
from onnxruntime import quantization

# synset to target
synset_to_target = {}
f = open("synset_words.txt", "r")
index = 0
for line in f:
    parts = line.split(" ")
    synset_to_target[parts[0]] = index
    index = index + 1
f.close()

# dataset
preprocess = tv.transforms.Compose(
    [
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def tar_transform(synset):
    return synset_to_target[synset]


class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_paths = sorted(
            glob.glob(img_dir + "*/*.JPEG"),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        synset = img_path.split("/")[-2]
        label = synset
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


ds = ImageNetValDataset(f"{VAL}/", transform=preprocess, target_transform=tar_transform)

# slice the dataset

offset = 500
calib_ds = torch.utils.data.Subset(ds, list(range(offset)))
val_ds = torch.utils.data.Subset(ds, list(range(offset, offset * 2)))


# dataloader
batch_size = 64
dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# load the model
model_pt = torch.hub.load(
    "pytorch/vision:v0.10.0", "resnet18", weights=tv.models.ResNet18_Weights.DEFAULT
)
model_pt.eval()

# trace the model
dummy_in = torch.randn(1, 3, 224, 224, requires_grad=True)

dummy_out = model_pt(dummy_in)


# convert to onnx
# export fp32 model to onnx
model_fp32_path = "resnet18_fp32.onnx"

torch.onnx.export(
    model_pt,  # model
    dummy_in,  # model input
    model_fp32_path,  # path
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=14,  # the ONNX version to export the model to
    do_constant_folding=True,  # constant folding for optimization
    input_names=["input"],  # input names
    output_names=["output"],  # output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

# verify the onnx model
model_onnx = onnx.load(model_fp32_path)
onnx.checker.check_model(model_onnx)

# pytorch tensor to numpy array
def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# prepare models
ort_provider = ["CPUExecutionProvider"]
if torch.cuda.is_available():
    model_pt.to("cuda")
    ort_provider = ["CUDAExecutionProvider"]

ort_sess = ort.InferenceSession(model_fp32_path, providers=ort_provider)

# test models
correct_pt = 0
correct_onnx = 0
tot_abs_error = 0

for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
    ort_outs = ort_sess.run(None, ort_inputs)[0]

    ort_preds = np.argmax(ort_outs, axis=1)
    correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

    if torch.cuda.is_available():
        img_batch = img_batch.to("cuda")
        label_batch = label_batch.to("cuda")

    with torch.no_grad():
        pt_outs = model_pt(img_batch)

    pt_preds = torch.argmax(pt_outs, dim=1)
    correct_pt += torch.sum(pt_preds == label_batch)

    tot_abs_error += np.sum(np.abs(to_numpy(pt_outs) - ort_outs))

print("\n")

print(
    f"pt top-1 acc = {100.0 * correct_pt/len(val_ds)} with {correct_pt} correct samples"
)
print(
    f"onnx top-1 acc = {100.0 * correct_onnx/len(val_ds)} with {correct_onnx} correct samples"
)

mae = tot_abs_error / (1000 * len(val_ds))
print(f"mean abs error = {mae} with total abs error {tot_abs_error}")

# prep quantization
model_prep_path = "resnet18_prep.onnx"

quantization.shape_inference.quant_pre_process(
    model_fp32_path, model_prep_path, skip_symbolic_shape=False
)

# calibration data readre
class QuntizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(
            torch_ds, batch_size=batch_size, shuffle=False
        )

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return (
            pt_tensor.detach().cpu().numpy()
            if pt_tensor.requires_grad
            else pt_tensor.cpu().numpy()
        )

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
            return {self.input_name: self.to_numpy(batch[0])}
        else:
            return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


qdr = QuntizationDataReader(
    calib_ds, batch_size=64, input_name=ort_sess.get_inputs()[0].name
)

# quantize
q_static_opts = {"ActivationSymmetric": False, "WeightSymmetric": True}
if torch.cuda.is_available():
    q_static_opts = {"ActivationSymmetric": True, "WeightSymmetric": True}

model_int8_path = "resnet18_int8.onnx"
quantized_model = quantization.quantize_static(
    model_input=model_prep_path,
    model_output=model_int8_path,
    calibration_data_reader=qdr,
    extra_options=q_static_opts,
)
# load quantized model
ort_int8_sess = ort.InferenceSession(model_int8_path, providers=ort_provider)


# test the models
correct_int8 = 0
correct_onnx = 0
tot_abs_error = 0

for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
    ort_outs = ort_sess.run(None, ort_inputs)[0]

    ort_preds = np.argmax(ort_outs, axis=1)
    correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

    ort_int8_outs = ort_int8_sess.run(None, ort_inputs)[0]

    ort_int8_preds = np.argmax(ort_int8_outs, axis=1)
    correct_int8 += np.sum(np.equal(ort_int8_preds, to_numpy(label_batch)))

    tot_abs_error += np.sum(np.abs(ort_int8_outs - ort_outs))


print("\n")

print(
    f"onnx top-1 acc = {100.0 * correct_onnx/len(val_ds)} with {correct_onnx} correct samples"
)
print(
    f"onnx int8 top-1 acc = {100.0 * correct_int8/len(val_ds)} with {correct_int8} correct samples"
)

mae = tot_abs_error / (1000 * len(val_ds))
print(f"mean abs error = {mae} with total abs error {tot_abs_error}")
