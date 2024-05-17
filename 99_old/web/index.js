// // Adds the CPU backend.
// import '@tensorflow/tfjs-backend-cpu';
// // Import @tensorflow/tfjs-core
// import * as tf from '@tensorflow/tfjs-core';
// Import @tensorflow/tfjs-tflite.
// import * as tflite from '@tensorflow/tfjs-tflite';
require('@tensorflow/tfjs-backend-cpu');
tf = require('@tensorflow/tfjs-core');
tflite = require('@tensorflow/tfjs-tflite');

testData = require("./test_data.json");

const modelUrl = 'model_80_dropout.tflite';

// Load the model
const model = tflite.loadTFLiteModel(modelUrl);

const inputData = tf.tensor2d(testData);

const output = model.predict(inputData);

output.print();

model.dispose();
