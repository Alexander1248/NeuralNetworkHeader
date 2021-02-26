#pragma once
#include <math.h>
#include <iostream>
using namespace std;


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



class NNLayer {
private:
	int layerSize;
	float** weights;
	float* values;
	float* error;

	float* biasWeights;
	
public:
	void setLayerSize(int size) { layerSize = size; }
	int getLayerSize() { return layerSize; }

	void setValue(int valueNumber, float val) { values[valueNumber] = val; }
	float getValue(int valueNumber) { return values[valueNumber]; }


	void setValues(float* val) { values = val; }
	float* getValues() { return values; }

	float** getWeights() { return weights; }
	float* getBiasWeights() { return biasWeights; }

	//Initializers
	void initLayerValues() {
		values = new float[layerSize];
		for (int i = 0; i < layerSize; i++) values[i] = 0;
	}
	void initLayerWeights(NNLayer preLayer, float scalingFactor) {
		biasWeights = new float[layerSize];
		weights = new float*[layerSize];
		for (int i = 0; i < layerSize; i++) weights[i] = new float[preLayer.layerSize];
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
			}
			biasWeights[j] = float(rand() % int(scalingFactor * 200)) / 100.f - scalingFactor;
		}
	}
	void initLayer(NNLayer preLayer, float scalingFactor) {
		values = new float[layerSize];
		biasWeights = new float [layerSize];
		weights = new float* [layerSize];
		error = new float[layerSize];

		for (int i = 0; i < layerSize; i++) weights[i] = new float[preLayer.layerSize], values[i] = 0;

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
			}
			biasWeights[j] = float(rand() % int(scalingFactor * 200)) / 100.f - scalingFactor;
		}
	}


	//Layer functions
	void calculateLayer(NNLayer preLayer, bool sigmoidal) {
		for (int j = 0; j < layerSize; j++) {
			float sum = 0;
			for (int i = 0; i < preLayer.layerSize; i++) {
				if (sigmoidal == true) sum += sigmoid(preLayer.values[i]) * weights[j][i];
				else sum += neurotan(preLayer.values[i]) * weights[j][i];
			}
			values[j] = sum + biasWeights[j];
		}
	}

	void layerBackPropogationErrorCalculating(float* rightResults, NNLayer postLayer, bool outLayer, bool sigmoidal) {
		for (int i = 0; i < layerSize; i++) error[i] = 0;

		if (outLayer == true) {
			for (int output = 0; output < layerSize; output++) {
				if (sigmoidal == true) error[output] = (rightResults[output] - sigmoid(values[output])) * sigmoiderivative(values[output]);
				else error[output] = (rightResults[output] - neurotan(values[output])) * tanderivative(values[output]);
			}
		}
		else {
			for (int input = 0; input < layerSize; input++) {
				for (int output = 0; output < postLayer.layerSize; output++) {
					error[input] += postLayer.weights[output][input] * postLayer.error[output];
				}
				if (sigmoidal == true) error[input] *= sigmoiderivative(values[input]);
				else error[input] *= tanderivative(values[input]);
			}
		}
	}
	void layerBackPropogationWeightsCalculating(float trainSpeed, NNLayer preLayer, bool sigmoidal, bool postInpLayer) {
		for (int output = 0; output < layerSize; output++) {
			for (int input = 0; input < preLayer.layerSize; input++) {
				if (postInpLayer == true) {
					if (sigmoidal == true) { weights[output][input] += error[output] * preLayer.values[input] * trainSpeed; }
					else { weights[output][input] += error[output]  * preLayer.values[input] * trainSpeed; }
				}
				else {
					if (sigmoidal == true) { weights[output][input] += error[output] * sigmoid(preLayer.values[input]) * trainSpeed; }
					else { weights[output][input] += error[output]  * neurotan(preLayer.values[input]) * trainSpeed; }
				}
			}
			biasWeights[output] += error[output] * trainSpeed;
		}
	}
}; 

class NeuralNetwork {
private:
	bool NNSigmoid;
	int nLayers;
public:
	NNLayer* layers;

	void initNeuralNetwork(int* layersSize, int nOfLayers,bool sigmoidal) {
		//Filling the parameters of the neural network
		nLayers = nOfLayers;
		NNSigmoid = sigmoidal;

		//Initialization of the neural network structure
		layers = new NNLayer[nLayers];
		for (int i = 0; i < nLayers; i++) layers[i].setLayerSize(layersSize[i]);

		//Initialization of each layer of the neural network
		layers[0].initLayerValues();
		int sum = 0;
		for (int i = 1; i < nLayers - 1; i++) sum += layersSize[i];
		for (int i = 1; i < nLayers; i++) layers[i].initLayer(layers[i - 1],calculateScalingFactor(sum,layersSize[0]));
		
	}

	float* calculateNeuralNetwork(float* inputData) {
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

	void trainNeuralNetwork(float* trainInput, float* trainOutput, float trainSpeed) {
		//Standard calculation of the neural network result
		layers[0].setValues(trainInput);
		for (int i = 1; i < nLayers; i++) layers[i].calculateLayer(layers[i - 1], NNSigmoid);
		
		//Calculation of weights by the method of back propagation of the error
		if (nLayers == 2) {
			layers[nLayers - 1].layerBackPropogationErrorCalculating(trainOutput, layers[nLayers - 1],true, NNSigmoid);
			layers[nLayers - 1].layerBackPropogationWeightsCalculating(trainSpeed, layers[nLayers - 2], true, true);
		}
		else {
			layers[nLayers - 1].layerBackPropogationErrorCalculating(trainOutput, layers[nLayers - 1], true, NNSigmoid);
			for (int i = nLayers - 2; i >= 1; i--) layers[i].layerBackPropogationErrorCalculating(trainOutput, layers[i + 1], false, NNSigmoid);

			layers[1].layerBackPropogationWeightsCalculating(trainSpeed, layers[0], NNSigmoid, true);
			for (int i = 2; i < nLayers; i++)layers[1].layerBackPropogationWeightsCalculating(trainSpeed, layers[i - 1], NNSigmoid, false);
		}
	}
	float calculateErrorPercentage(float* trainOutput) {
		float k = 0;
		for (int i = 0; i < layers[nLayers - 1].getLayerSize(); i++) {
			if (NNSigmoid == true) k += float(pow(sigmoid(layers[nLayers - 1].getValue(i)) - trainOutput[i], 2)) / layers[nLayers - 1].getLayerSize();
			else k += float(pow(neurotan(layers[nLayers - 1].getValue(i)) - trainOutput[i], 2)) / layers[nLayers - 1].getLayerSize();
		}
		return k;
	}
};