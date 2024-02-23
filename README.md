# ONNX Age Recongnition Lab

[Source repo](https://github.com/IT-HUSET/onnx-age-estimation-demo)

In this lab we build a system to estimate age from an image. This is done directly
in the browser using a framework called `onnxruntime-web`. It allows us to perform inference
with a deep learning model that has been pretrained to estimate age.

To estimate age:

- user uploads the image
- pre-process image to meet model's requirements
- perform inference with onnx model

## Development

1. Download the pre-trained model from [onnx' GitHub](https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/models/age_googlenet.onnx).

2. Install dependencies with `npm i`

3. Run the app locally with `npm start`
