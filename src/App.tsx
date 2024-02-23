// Based on https://github.com/IT-HUSET/onnx-age-estimation-demo

import { InferenceSession, Tensor } from "onnxruntime-web";
import { useEffect, useRef, useState } from "react";
import {
  AGE_INTERVALS,
  INPUT_HEIGHT,
  INPUT_WIDTH,
  TRAINING_INPUT_DATA_MEAN,
} from "./utils/constants";
import { argmax, interleavedToPlanar, removeAlpha } from "./utils";

import "./App.css";

type FileWithUrl = File & { url: string };

let model: InferenceSession;

function App() {
  const [selectedImage, setSelectedImage] = useState<FileWithUrl | null>(null);

  const inputImage = useRef<HTMLImageElement>(null);
  const [preprocessed, setPreprocessed] = useState<Float32Array>();
  const [estimatedAge, setEstimatedAge] = useState<string>();
  const [detailedOutput, setDetailedOutput] = useState<Float32Array>();

  const initializeModel = async () => {
    model = await InferenceSession.create("age_googlenet.onnx", {
      executionProviders: ["webgl"],
    });
  };

  useEffect(() => {
    if (!model) {
      initializeModel();
    }
  }, []);

  const resetResult = () => {
    setEstimatedAge(undefined);
    setDetailedOutput(undefined);
    setPreprocessed(undefined);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const image = e.target.files[0] as FileWithUrl;
      image.url = URL.createObjectURL(image);
      setSelectedImage(image);
      resetResult();
    }
  };

  const preprocess = () => {
    const canvas = document.createElement("canvas");

    const img_w = inputImage.current!.width;
    const img_h = inputImage.current!.height;
    canvas.width = INPUT_WIDTH;
    canvas.height = INPUT_HEIGHT;

    // 1. Scale the picture to 224x224
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(
      inputImage.current!,
      0,
      0,
      img_w,
      img_h,
      0,
      0,
      INPUT_WIDTH,
      INPUT_HEIGHT
    );

    // 2. Extract the pixel data from our canvas element
    const array = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    // 3. Remove the alpha-channel
    const without_alpha = removeAlpha(array);

    // 4. and 5. Convert to floating point array and subtract the mean of the training data
    const f32array = Float32Array.from(
      without_alpha,
      (x) => x - TRAINING_INPUT_DATA_MEAN
    );

    // 6. Convert to planear-format
    const channelSeparated = interleavedToPlanar(f32array);

    setPreprocessed(channelSeparated);
  };

  const estimateAge = async () => {
    const tensor = new Tensor(preprocessed!, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);

    const results = await model.run({ input: tensor });
    const output = results["loss3/loss3_Y"].data as Float32Array;

    const highestProbabilityIndex = argmax(output as Float32Array);
    const ageInterval = AGE_INTERVALS[highestProbabilityIndex];

    setEstimatedAge(ageInterval);
    setDetailedOutput(output);
  };

  return (
    <div className="wrapper">
      <h1>Age Estimator</h1>
      {selectedImage && (
        <img
          id="inputImage"
          src={selectedImage.url}
          crossOrigin="anonymous"
          ref={inputImage}
          onLoad={preprocess}
        />
      )}

      <div className="upload--wrapper">
        <input
          id="file"
          type="file"
          className="upload--file"
          accept="image/*"
          onChange={handleFileChange}
        />
        <label htmlFor="file" className="upload--label">
          Choose a file
        </label>
        {selectedImage && (
          <span className="upload--file-name">
            {selectedImage.name} {(selectedImage.size / 1000000).toFixed(2)}Mb
          </span>
        )}
      </div>

      <div className="results--wrapper">
        <button
          id="estimateAge"
          type="button"
          onClick={estimateAge}
          className="results--btn"
          disabled={!selectedImage}
        >
          Estimate Age
        </button>
        {estimatedAge && <span>Estimated Age: {estimatedAge}</span>}
      </div>
      {detailedOutput && (
        <div className="details--wrapper">
          <details>
            <summary>Detailed age recognition results</summary>
            {AGE_INTERVALS.map((interval, index) => {
              return (
                <p key={interval}>
                  <b>{interval}</b>: {detailedOutput[index]}
                </p>
              );
            })}
          </details>
        </div>
      )}
    </div>
  );
}

export default App;
