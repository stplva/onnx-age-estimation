# ONNX Age Recongnition Lab

[Source repo](https://github.com/IT-HUSET/onnx-age-estimation-demo)

In this lab, we build a system to estimate age from an image. This is done directly
in the browser using a framework called `onnxruntime-web`. It allows us to perform inference
with a deep learning model that has been pre-trained to estimate age.

To estimate age:

- user uploads the image
- pre-process image to meet the model's requirements
- perform inference with onnx model

## Demo

[Live Demo](https://stplva.github.io/onnx-age-estimation/)

## Development

1. Install dependencies with `npm i`
2. Run the app locally with `npm start`
3. Deploy to GitHub Pages with `npm run deploy`
