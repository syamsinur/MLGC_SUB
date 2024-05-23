const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()

    const prediction = model.predict(tensor);

    const classes = ['Cancer', 'Non-cancer'];

    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let suggestion;

    if (label === 'Cancer') {
      suggestion = "Segera periksa ke dokter!"
    }
  
    if (label === 'Non-cancer') {
      suggestion = "Anda sehat!"
    }
  

    return { label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
