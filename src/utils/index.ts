/*
 * Remove the alpha channel from an interleaved RGBA imagedata array.
 */
export const removeAlpha = (array: Uint8ClampedArray) => {
  const result = new Uint8ClampedArray((array.length / 4) * 3);
  for (let i = 0; i < array.length; i += 4) {
    result[(i / 4) * 3 + 0] = array[i + 0]; // R
    result[(i / 4) * 3 + 1] = array[i + 1]; // G
    result[(i / 4) * 3 + 2] = array[i + 2]; // B
  }
  return result;
};

/*
 * Convert from interleaved RGB to planar RGB.
 */
export const interleavedToPlanar = (array: Float32Array) => {
  const plane_size = array.length / 3;
  const result = new Float32Array(array.length);
  for (let i = 0; i < plane_size; i++) {
    result[i + plane_size * 0] = array[i * 3 + 0];
    result[i + plane_size * 1] = array[i * 3 + 1];
    result[i + plane_size * 2] = array[i * 3 + 2];
  }
  return result;
};

export const argmax = (array: Float32Array) => {
  let max = array[0];
  let max_index = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > max) {
      max = array[i];
      max_index = i;
    }
  }
  return max_index;
};
