#include "NNetworkBasic.h"



float sigmoid(float x) {
	return 1.f / float(1 + pow(2.72, -x));
}
float neurotan(float x) {
	return 2.f / float(1 + pow(2.72, -x)) - 1;
}
double sigmoiderivative(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}
double tanderivative(double x)
{
	return 0.5 * (1 + neurotan(x)) * (1 - neurotan(x));
}
float calculateScalingFactor(int kHiddenLayerNeurons, int kInputLayerNeurons) {
	return 0.7 * pow((float)kHiddenLayerNeurons, 1.f / (float)kInputLayerNeurons);
}



void NNLayer::initLayerWeights(NNLayer preLayer, float scalingFactor) {
	biasWeights = new float[layerSize];
	weights = new float* [layerSize];
	acceleration = new float* [layerSize];

	for (int i = 0; i < layerSize; i++) weights[i] = new float[preLayer.layerSize], acceleration[i] = new float[preLayer.layerSize];
	for (int j = 0; j < layerSize; j++) {
		for (int i = 0; i < preLayer.layerSize; i++) {
			weights[j][i] = float(rand() % 100) / 100.f - 0.5;
		}
		float sum = 0;
		for (int i = 0; i < preLayer.layerSize; i++) {
			sum += pow(weights[j][i], 2);
		}
		for (int i = 0; i < preLayer.layerSize; i++) {
			weights[j][i] = scalingFactor * weights[j][i] / sqrt(sum);
			acceleration[j][i] = 0;
		}
		biasWeights[j] = float(rand() % int(scalingFactor * 200)) / 100.f - scalingFactor;
	}
}
void NNLayer::initLayer(NNLayer preLayer, float scalingFactor) {
	values = new float[layerSize];
	biasWeights = new float[layerSize];
	weights = new float* [layerSize];
	acceleration = new float* [layerSize];
	error = new float[layerSize];

	for (int i = 0; i < layerSize; i++) weights[i] = new float[preLayer.layerSize], acceleration[i] = new float[preLayer.layerSize], values[i] = 0;

	for (int j = 0; j < layerSize; j++) {
		for (int i = 0; i < preLayer.layerSize; i++) {
			weights[j][i] = float(rand() % 100) / 100.f - 0.5;
		}
		float sum = 0;
		for (int i = 0; i < preLayer.layerSize; i++) {
			sum += pow(weights[j][i], 2);
		}
		for (int i = 0; i < preLayer.layerSize; i++) {
			weights[j][i] = scalingFactor * weights[j][i] / sqrt(sum);
			acceleration[j][i] = 0;
		}
		biasWeights[j] = float(rand() % int(scalingFactor * 200)) / 100.f - scalingFactor;
	}
}
void NNLayer::calculateLayer(NNLayer preLayer, bool sigmoidal) {
		for (int j = 0; j < layerSize; j++) {
			float sum = 0;
			if (sigmoidal == true) {
				for (int i = 0; i < preLayer.layerSize; i++) {
					sum += sigmoid(preLayer.values[i]) * weights[j][i];
				}
			}
			else {
				for (int i = 0; i < preLayer.layerSize; i++) {
					sum += neurotan(preLayer.values[i]) * weights[j][i];
				}
			}
			values[j] = sum + biasWeights[j];
		}
}
void NNLayer::layerBackPropogationErrorCalculating(float* rightResults, NNLayer postLayer, bool outLayer, bool sigmoidal) {
	for (int i = 0; i < layerSize; i++) error[i] = 0;

	if (outLayer == true) {

		if (sigmoidal == true) {
			for (int output = 0; output < layerSize; output++) {
				error[output] = (rightResults[output] - sigmoid(values[output])) * sigmoiderivative(values[output]);
			}
		}
		else {
			for (int output = 0; output < layerSize; output++) {
				error[output] = (rightResults[output] - neurotan(values[output])) * tanderivative(values[output]);
			}
		}
	}
	else {
		if (sigmoidal == true) {
			for (int input = 0; input < layerSize; input++) {
				for (int output = 0; output < postLayer.layerSize; output++) {
					error[input] += postLayer.weights[output][input] * postLayer.error[output];
				}
				error[input] *= sigmoiderivative(values[input]);
			}
		}
		else {
			for (int input = 0; input < layerSize; input++) {
				for (int output = 0; output < postLayer.layerSize; output++) {
					error[input] += postLayer.weights[output][input] * postLayer.error[output];
				}
				error[input] *= tanderivative(values[input]);
			}
		}
	}
}
void NNLayer::layerBackPropogationWeightsCalculating(float trainSpeed, float momentumCoef, NNLayer preLayer, bool sigmoidal, bool postInpLayer) {
	if (postInpLayer == true) {
		for (int output = 0; output < layerSize; output++) {
			for (int input = 0; input < preLayer.layerSize; input++) {
				acceleration[output][input] = acceleration[output][input] * momentumCoef + (1 - momentumCoef) * error[output] * preLayer.values[input] * trainSpeed;

				weights[output][input] += acceleration[output][input];

			}
			biasWeights[output] += error[output] * trainSpeed;
		}
	}
	else {
		if (sigmoidal == true) {
			for (int output = 0; output < layerSize; output++) {
				for (int input = 0; input < preLayer.layerSize; input++) {
					acceleration[output][input] = acceleration[output][input] * momentumCoef + (1 - momentumCoef) * error[output] * sigmoid(preLayer.values[input]) * trainSpeed;

					weights[output][input] += acceleration[output][input];

				}
				biasWeights[output] += error[output] * trainSpeed;
			}
		}
		else {
			for (int output = 0; output < layerSize; output++) {
				for (int input = 0; input < preLayer.layerSize; input++) {
					acceleration[output][input] = acceleration[output][input] * momentumCoef + (1 - momentumCoef) * error[output] * neurotan(preLayer.values[input]) * trainSpeed;

					weights[output][input] += acceleration[output][input];
				}
				biasWeights[output] += error[output] * trainSpeed;
			}
		}
	}
}

float* NeuralNetwork::calculateNeuralNetwork(float* inputData) {
	layers[0].setValues(inputData);
	for (int i = 1; i < nLayers; i++) layers[i].calculateLayer(layers[i - 1], NNSigmoid);

	float* out = new float[layers[nLayers - 1].getLayerSize()];
	out = layers[nLayers - 1].getValues();
	for (int i = 0; i < layers[nLayers - 1].getLayerSize(); i++) {
		if (NNSigmoid == true) out[i] = sigmoid(out[i]);
		else out[i] = neurotan(out[i]);
	}
	return out;
	delete[] out;
}

void NeuralNetwork::trainNeuralNetwork(float* trainInput, float* trainOutput, float trainSpeed, float momentum) {
	//Standard calculation of the neural network result
	layers[0].setValues(trainInput);
	for (int i = 1; i < nLayers; i++) layers[i].calculateLayer(layers[i - 1], NNSigmoid);

	//Calculation of weights by the method of back propagation of the error
	if (nLayers == 2) {
		layers[nLayers - 1].layerBackPropogationErrorCalculating(trainOutput, layers[nLayers - 1], true, NNSigmoid);
		layers[nLayers - 1].layerBackPropogationWeightsCalculating(trainSpeed, momentum, layers[nLayers - 2], true, true);
	}
	else {
		layers[nLayers - 1].layerBackPropogationErrorCalculating(trainOutput, layers[nLayers - 1], true, NNSigmoid);
		for (int i = nLayers - 2; i >= 1; i--) layers[i].layerBackPropogationErrorCalculating(trainOutput, layers[i + 1], false, NNSigmoid);

		layers[1].layerBackPropogationWeightsCalculating(trainSpeed, momentum, layers[0], NNSigmoid, true);
		for (int i = 2; i < nLayers; i++)layers[1].layerBackPropogationWeightsCalculating(trainSpeed, momentum, layers[i - 1], NNSigmoid, false);
	}
}