package neuralnetwork;

import java.util.Random;

/**
 * Class for the connections between Neurons in a neural network. Holds
 * connection weights, and calculates how these weights should be changed
 * between epochs.
 * 
 * @author Nick
 *
 */
public class Connection {

	Random r = new Random();

	/*
	 * Properties of the connection.
	 */
	public Neuron fromNeuron;
	public Neuron toNeuron;
	public double weight;

	/*
	 * Storage location for weight changes. weightChange stores the most recent
	 * weight change. epochChange stores all weight changes for offline weight
	 * change.
	 */

	public double weightChange[];
	public double epochChange = 0;

	/**
	 * Constructor for a given weight.
	 * 
	 * @param i      Source neuron.
	 * @param j      Target neuron.
	 * @param weight Connection weight value.
	 * @param length Number of patterns in input array.
	 */
	public Connection(Neuron i, Neuron j, double weight, int length) {
		this.fromNeuron = i;
		this.toNeuron = j;
		this.weight = weight;
		this.weightChange = new double[length];
	}

	/**
	 * Constructor without a given weight. Weight is initialised to a positive or
	 * negative small value.
	 * 
	 * @param i      Source neuron.
	 * @param j      Target neuron.
	 * @param length Number of patterns in input array.
	 */
	public Connection(Neuron i, Neuron j, int length) {
		this.fromNeuron = i;
		this.toNeuron = j;
		this.weight = 0;
		while (this.weight == 0) {
			this.weight = ((r.nextDouble() * 2) - 1) * .3;
		}
		this.weightChange = new double[length];
	}

	/**
	 * Accessor for Connection weight.
	 * 
	 * @return Value of weight.
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * Mutator for Connection weight.
	 * 
	 * @param weight New weight value.
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}

	/**
	 * Accessor for Connection input.
	 * 
	 * @return Value of input.
	 */
	public double getInput() {
		return fromNeuron.calcOutput();
	}

	/**
	 * Accessor for weighted connection input.
	 * 
	 * @return Value of input, multiplied by weight.
	 */
	public double getWeightedInput() {
		return (fromNeuron.calcOutput() * this.weight);
	}

	/**
	 * Calculates the weight change for the current I/O pair, based on the target
	 * neuron's error term.
	 * 
	 * @param constant Network learning constant.
	 * @param momentum Network momentum constant.
	 * @param pattern  Pattern number.
	 * @return Weight change.
	 */
	public double calcWeightChange(double constant, double momentum, int pattern) {
		double lastChange = this.weightChange[pattern];
		// System.out.println(lastChange);
		double newChange = (lastChange * momentum) + (constant * this.toNeuron.errorTerm * getInput());
		this.weightChange[pattern] = newChange;
		return newChange;
	}

	/**
	 * Stores weight changes during an epoch for offline weight change.
	 * 
	 * @param constant Network learning constant.
	 * @param momentum Network momentum constant.
	 * @param pattern  Pattern number for tracking previous weight change.
	 * @return Total epoch weight change so far.
	 */
	public double changeWeight(double constant, double momentum, int pattern) {
		double change;
		change = calcWeightChange(constant, momentum, pattern);
		this.epochChange += change;
		return epochChange;
	}

	/**
	 * Updates the weights at the end of an epoch. Weight is adjusted according to
	 * accumulated change.
	 */
	public void updateWeight() {
		setWeight(this.weight + this.epochChange);
		this.epochChange = 0;
	}
}
