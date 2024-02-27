# segment_anything_onnx_models
onnx_models for segment_anything

1. change in segment-anything\scripts\export_onnx_model.py

parser.add_argument(
   "--return-single-mask",
    default=False,
    #action="store_true",

2. convert pth model to onnx (b model for example)
python scripts/export_onnx_model.py --checkpoint sam_vit_b_01ec64.pth --model-type vit_b --output vit_b.onnx

3.use to extract multiple masks
```
import cv2,time
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=checkpoint)

onnx_model_path='vit_b.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)
sam.to(device='cuda')

start_time = time.time()

img = cv2.imread("near.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
print(len(masks))

print("--- %s seconds ---" % (time.time() - start_time))
```
