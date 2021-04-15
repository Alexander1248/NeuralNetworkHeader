#pragma once
#include <math.h>
#include <iostream>
using namespace std;




class NNLayer {
private:
	int layerSize;
	float** weights;
	float* values;
	float* error;

	float* biasWeights;

	float** acceleration;
	
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
	void initLayerWeights(NNLayer preLayer, float scalingFactor);
	void initLayer(NNLayer preLayer, float scalingFactor);


	//Layer functions
	void calculateLayer(NNLayer preLayer, bool sigmoidal);
	void layerBackPropogationErrorCalculating(float* rightResults, NNLayer postLayer, bool outLayer, bool sigmoidal);
	void layerBackPropogationWeightsCalculating(float trainSpeed, float momentumCoef, NNLayer preLayer, bool sigmoidal, bool postInpLayer);
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

	float* calculateNeuralNetwork(float* inputData);

	void trainNeuralNetwork(float* trainInput, float* trainOutput, float trainSpeed, float momentum);
	float calculateErrorPercentage(float* trainOutput);
};
