package neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Class Neuron. The processing units of the neural network. Contains lists of
 * their connections to neurons in other NeuronLayers. Propagation, activation,
 * and error calculation methods for backpropagation are here.
 * 
 * @author Nick
 *
 */
public class Neuron {

	/*
	 * Identifier for Neuron type: (I)nput, (H)idden, (O)utput, or (B)ias.
	 */
	public String name;

	/*
	 * Amount of output from the Neuron.
	 */
	public double output;

	/*
	 * Lists of Connections to associated input and output Neurons.
	 */
	public List<Connection> inputConnections;
	public List<Connection> outputConnections;

	/*
	 * Error term, used in weight change calculations during backpropagation.
	 */
	public double errorTerm = 0;

	/**
	 * Default constructor for an empty neuron.
	 */
	public Neuron() {
		this.inputConnections = new ArrayList<>();
		this.outputConnections = new ArrayList<>();
	}

	/**
	 * Constructor for a named neuron with a specific type.
	 * 
	 * @param s Neuron identifier, allowing the program to see what type of Neuron
	 *          this is.
	 */
	public Neuron(String s) {
		this.name = s;
		this.inputConnections = new ArrayList<>();
		this.outputConnections = new ArrayList<>();
	}

	/**
	 * Calculated the total input into the Neuron by summating the weighted inputs
	 * of all incoming Connections.
	 * 
	 * @return Weighted sum of all inputs.
	 */
	public double propagationRule() {
		double sumInput = 0;
		for (Connection c : this.inputConnections) {
			sumInput += c.getWeightedInput();
		}
		return sumInput;
	}

	/**
	 * Calculates the Neuron's activation, based on a sigmoidal activation function.
	 * Sets neuron activation to output value.
	 * 
	 * @param input Weighted sum of inputs, from propagation rule
	 * @return Activation value.
	 */
	public double activationRule(double input) {
		this.output = 1 / (1 + Math.pow(Math.E, -input));
		return output;
	}

	/**
	 * Calculates Neuron output, based on activation rule and propagation rules.
	 * Input and Bias neurons do not need additional calculations.
	 * 
	 * @return Neuron output.
	 */
	public double calcOutput() {
		if (this.name == "B") {
			return 1;
		} else if (this.name == "I") {
			return this.output;
		} else {
			double totalInput = propagationRule();
			return activationRule(totalInput);
		}
	}

	/**
	 * Sets neuron output to a desired value.
	 * 
	 * @param output Designated output.
	 */
	public void setOutput(double output) {
		this.output = output;
	}

	/**
	 * Calculates the error of an output neuron, with a given teaching input to
	 * compare to.
	 * 
	 * @param teach Teaching input for output comparison.
	 * @return Calculated error term.
	 */
	public double calcError(double teach) {
		this.errorTerm = ((teach - this.output) * this.output * (1 - this.output));
		return this.errorTerm;
	}

	/**
	 * Calculates the error of a hidden neuron, using the error terms of successor
	 * neurons.
	 * 
	 * @return Calculated error term.
	 */
	public double calcError() {
		double sumError = 0;
		for (Connection c : outputConnections) {
			sumError += c.toNeuron.errorTerm * c.getWeight();
		}
		this.errorTerm = this.output * (1 - this.output) * sumError;
		return this.errorTerm;
	}
}
