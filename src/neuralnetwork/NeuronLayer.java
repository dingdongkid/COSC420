package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
/**
 * Class NeuronLayer. Holds a list of Neurons. Controls their activation.
 * @author Nick
 *
 */
public class NeuronLayer {

	/*
	 * ArrayList holding Neuron processing units in this layer.
	 */
	public List<Neuron> neurons;

	/**
	 * Default constructor for an empty NeuronLayer.
	 */
	public NeuronLayer() {
		this.neurons = new ArrayList<>();
	}

	/**
	 * Constructor for a given list of Neurons.
	 * 
	 * @param neurons ArrayList of preexisting Neuron units.
	 */
	public NeuronLayer(List<Neuron> neurons) {
		this.neurons = neurons;
	}

	/**
	 * Adds a Neuron to the list of Neurons.
	 * 
	 * @param n Neuron to be added.
	 */
	public void addNeuron(Neuron n) {
		neurons.add(n);
	}

	/**
	 * Given an array of inputs, sets the Neuron activations in the layer to the
	 * designated row of inputs, which will be propagated through the network. Used
	 * in the InputLayer NeuronLayer.
	 * 
	 * @param input Array of input patterns.
	 * @param j     Row of input patterns to be set.
	 */
	public void setInputNeurons(double[][] input, int j) {
		for (int i = 0; i < neurons.size(); i++) {
			neurons.get(i).setOutput(input[i][j]);
		}
	}

	/**
	 * Iterate through neurons and calculate outputs. Used in HiddenLayer, as
	 * outputs do not have to be readily accessed later.
	 */
	public void calcHiddenOutputs() {
		for (Neuron n : this.neurons) {
			n.calcOutput();
		}
	}

	/**
	 * Iterate through neurons and calculate outputs. Neuron outputs will be stored in the given array.
	 * 
	 * @param outputArray Array to place outputs into.
	 * @param j Row of input patterns to place outputs into.
	 */
	public void calcOutputOutputs(double[][] outputArray, int j) {
		int i = 0;
		for (Neuron n : this.neurons) {
			outputArray[i][j] = n.calcOutput();
			i++;
		}
	}
}