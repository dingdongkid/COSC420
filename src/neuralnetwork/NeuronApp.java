package neuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Application file for neural network.
 * @author Nick
 *
 */
public class NeuronApp {

	/*
	 * declaring hyper-parameters.
	 */
	int input;
	int hidden;
	int output;
	double learn;
	double momentum;
	double errorCriterion;
	double populationError;

	double[][] inputArray;
	double[][] outputArray;
	double[][] teacherArray;

	int epochs;

	/**
	 * Returns a message outlining commands and current state.
	 *
	 * @return current state of application and what the commands are.
	 */
	private String info() {
		return "\nNeural network constructed. Enter one of the following commands:\n\n"
				+ "(L)earn    - learn weights according to given teaching pattern.\n"
				+ "(T)est     - test population of input patterns, and see activation of all units.\n"
				+ "(W)eights  - show weights of connections between neurons.\n" + "(E)xit     - exit program.\n";
	}

	/**
	 * Entry point of the program. Creates a NeuronApp instance, reads information
	 * from files, and constructs connected network structure.
	 *
	 * @param args The command line arguments are not used.
	 */
	public static void main(String[] args) {

		NeuronApp net = new NeuronApp();
		DecimalFormat f = new DecimalFormat("#.0000");

		System.out.println("Reading text files.");
		net.readParams();
		net.readInput();
		System.out.println();
		net.readTeacher();

		// build input layer
		NeuronLayer inputLayer = new NeuronLayer();
		for (int i = 0; i < net.input; i++) {
			Neuron n = new Neuron("I");
			inputLayer.addNeuron(n);
		}

		// build hidden layer
		NeuronLayer hiddenLayer = new NeuronLayer();
		for (int i = 0; i < net.hidden; i++) {
			Neuron n = new Neuron("H");
			hiddenLayer.addNeuron(n);
		}

		// build output layer
		NeuronLayer outputLayer = new NeuronLayer();
		for (int i = 0; i < net.output; i++) {
			Neuron n = new Neuron("O");
			outputLayer.addNeuron(n);
		}

		// make bias neuron
		Neuron bias = new Neuron("B");

		// make connections
		net.makeConnections(inputLayer, hiddenLayer, net.inputArray[0].length);
		net.makeConnections(hiddenLayer, outputLayer, net.inputArray[0].length);
		net.makeBiasConnections(bias, hiddenLayer, net.inputArray[0].length);
		net.makeBiasConnections(bias, outputLayer, net.inputArray[0].length);

		System.out.println(net.info());
		Scanner input = new Scanner(System.in);
		while (input.hasNextLine()) {
			String command = input.next().toLowerCase();
			switch (command) {
			case "learn":
			case "l":
				/*
				 * Learning process of the network. Learns until the population error is less
				 * than the error criterion, or the arbitrary epoch limit is reached.
				 */
				while (net.populationError >= net.errorCriterion && net.epochs < 500000) {

					for (int j = 0; j < net.inputArray[0].length; j++) { // for every teaching pattern
						net.setInput(inputLayer, j); // set the inputs into the input neurons
						hiddenLayer.calcHiddenOutputs(); // calculate the outputs of the hidden neurons
						outputLayer.calcOutputOutputs(net.outputArray, j); // calculate the outputs of the output
																			// neurons

						int k = 0;
						for (Neuron n : outputLayer.neurons) { // for each of the output neurons
							n.calcError(net.teacherArray[k][j]); // calculate the error term, given the teaching input
							for (Connection c : n.inputConnections) {
								// for each of its input connections, from hidden neurons
								c.changeWeight(net.learn, net.momentum, j);
								// collate the weight change to be made at the end of the epoch
							}
							k++;
						}
						for (Neuron n : hiddenLayer.neurons) { // for each of the hidden neurons
							n.calcError(); // calculate the error term, accounting for connected output neurons
							for (Connection c : n.inputConnections) {
								// for each of its input connections, from input neurons
								c.changeWeight(net.learn, net.momentum, j);
								// collate the weight change to be made at the end of the epoch
							}
						}
					}

					/*
					 * At the end of the epoch, make all of the weight changes simultaneously.
					 */
					for (Neuron n : outputLayer.neurons) {
						for (Connection c : n.inputConnections) {
							c.updateWeight();
						}
					}
					for (Neuron n : hiddenLayer.neurons) {
						for (Connection c : n.inputConnections) {
							c.updateWeight();
						}
					}

					net.epochs++;

					/*
					 * Every 100 epochs, prints the population error and epoch number.
					 */
					if (net.epochs % 100 == 0) {
						net.populationError = net.errorCheck(net.teacherArray, net.outputArray, net.output,
								net.inputArray[0].length);

						System.out.println("Population error: " + net.populationError);
						System.out.println("Number of epochs: " + net.epochs);
						System.out.println();

					}
				}
				if (net.populationError > net.errorCriterion) {
					System.out.println("Failure to reach error criterion.\n Population error: " + net.populationError);
				}
				break;
			case "test":
			case "t":
				/*
				 * Testing the network with the given input, returns output of output neurons.
				 */
				for (int j = 0; j < net.inputArray[0].length; j++) { // for every teaching pattern
					System.out.println("Pattern " + j);
					net.setInput(inputLayer, j); // set the inputs into the input neurons
					System.out.println("Input neuron outputs:");
					for (Neuron n : inputLayer.neurons) {
						System.out.print(f.format(n.output) + " ");
					}

					System.out.println();
					hiddenLayer.calcHiddenOutputs(); // calculate the outputs of the hidden neurons
					System.out.println("Hidden neuron outputs:");
					for (Neuron n : hiddenLayer.neurons) {
						System.out.print(f.format(n.output) + " ");
					}

					System.out.println();
					outputLayer.calcOutputOutputs(net.outputArray, j); // calculate the outputs of the output neurons
					System.out.println("Output neuron outputs:");
					for (Neuron n : outputLayer.neurons) {
						System.out.print(f.format(n.output) + " ");
					}

					System.out.println();
					System.out.println();
				}

				break;
			case "weights":
			case "w":
				/*
				 * Displays the connection weights between neurons.
				 */
				System.out.println("Displaying weights.");
				int i, h, o;
				// input neurons to hidden neurons
				for (i = 0; i < net.input; i++) {
					for (h = 0; h < net.hidden; h++) {
						Double w = inputLayer.neurons.get(i).outputConnections.get(h).weight;
						System.out.println("I[" + i + "] to H[" + h + "]: " + f.format(w));
					}
				}
				System.out.println();
				// hidden neurons to output neurons
				for (h = 0; h < net.input; h++) {
					for (o = 0; o < net.hidden; o++) {
						Double w = inputLayer.neurons.get(h).outputConnections.get(o).weight;
						System.out.println("H[" + h + "] to O[" + o + "]: " + f.format(w));
					}
				}
				System.out.println();
				// bias neuron to hidden and output neurons
				for (h = 0; h < net.hidden; h++) {
					Double w = bias.outputConnections.get(h).weight;
					System.out.println("B to H[" + h + "]: " + f.format(w));
				}
				for (o = 0; o < net.output; o++) {
					Double w = bias.outputConnections.get(h + o).weight;
					System.out.println("B to O[" + o + "]: " + f.format(w));
				}
				break;
			case "exit":
			case "e":
				/*
				 * Terminates the program.
				 */
				System.exit(0);
			default:
				System.err.println(net.info());
				return;
			}
		}
		input.close();
	}

	/**
	 * Method to read the inputs of the param.txt file and set the network
	 * hyperparameters.
	 */
	public void readParams() {
		File param = new File("param.txt");
		try {
			Scanner sc = new Scanner(param);
			input = sc.nextInt();
			hidden = sc.nextInt();
			output = sc.nextInt();
			learn = sc.nextDouble();
			momentum = sc.nextDouble();
			errorCriterion = sc.nextDouble();
			populationError = errorCriterion;
			sc.close();
			System.out.println("Reading parameters.");
			System.out.println("input: " + input);
			System.out.println("hidden: " + hidden);
			System.out.println("output: " + output);
			System.out.println("learn: " + learn);
			System.out.println("momentum: " + momentum);
			System.out.println("errorCriterion: " + errorCriterion);
			System.out.println();
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Method to read the input.txt file and store the input patterns into an array.
	 */
	public void readInput() {
		System.out.println("Reading inputs.");
		File in = new File("in.txt");
		try {
			int count = 0;
			Scanner lineCounter = new Scanner(in);
			while (lineCounter.hasNextLine()) {
				count++;
				lineCounter.nextLine();
			}
			lineCounter.close();

			inputArray = new double[input][count];

			Scanner sc = new Scanner(in);
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < input; j++) {
					inputArray[j][i] = sc.nextDouble();
					System.out.print(inputArray[j][i] + " ");
				}
				System.out.println("");
			}
			sc.close();

		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Method to read the teach.txt file and store the teaching inputs into an
	 * array. Also creates an empty array of the same size to hold outputs.
	 */
	public void readTeacher() {
		System.out.println("Reading teaching inputs.");
		File teach = new File("teach.txt");
		try {
			int count = 0; // reads the length of the teaching input before creating arrays
			Scanner lineCounter = new Scanner(teach);
			while (lineCounter.hasNextLine()) {
				count++;
				lineCounter.nextLine();
			}
			lineCounter.close();

			teacherArray = new double[output][count];
			outputArray = new double[output][count];

			Scanner sc = new Scanner(teach);
			for (int i = 0; i < count; i++) {
				for (int j = 0; j < output; j++) {
					teacherArray[j][i] = sc.nextDouble();
					System.out.print(teacherArray[j][i] + " ");
				}
				System.out.println("");
			}
			sc.close();

		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Method to make connections between all of the neurons in two layers.
	 * 
	 * @param a        Source layer
	 * @param b        Target layer
	 * @param patterns Number of input patterns to keep track of
	 */
	public void makeConnections(NeuronLayer a, NeuronLayer b, int patterns) {
		for (int i = 0; i < a.neurons.size(); i++) {
			for (int j = 0; j < b.neurons.size(); j++) {
				Connection c = new Connection(a.neurons.get(i), b.neurons.get(j), patterns);
				a.neurons.get(i).outputConnections.add(c);
				b.neurons.get(j).inputConnections.add(c);
			}
		}
	}

	/**
	 * Method to connect bias neuron to all neurons in a layer.
	 * 
	 * @param a        Bias neuron
	 * @param b        Target layer
	 * @param patterns Number of input patterns to keep track of
	 */
	public void makeBiasConnections(Neuron a, NeuronLayer b, int patterns) {
		for (int j = 0; j < b.neurons.size(); j++) {
			Connection c = new Connection(a, b.neurons.get(j), 0, patterns);
			a.outputConnections.add(c);
			b.neurons.get(j).inputConnections.add(c);
		}
	}

	/**
	 * Method to set input neuron activations to current input pattern.
	 * 
	 * @param layer Input NeuronLayer
	 * @param j     Pattern number
	 */
	public void setInput(NeuronLayer layer, int j) {
		layer.setInputNeurons(inputArray, j);
	}

	/**
	 * Method to check population error of the network, comparing teaching input to
	 * actual network output.
	 * 
	 * @param teach    Array of teaching inputs
	 * @param output   Array of actual outputs
	 * @param neurons  Number of output neurons
	 * @param patterns Number of input patterns
	 * @return
	 */
	private double errorCheck(double[][] teach, double[][] output, int neurons, int patterns) {
		double error = 0;
		for (int i = 0; i < teach.length; i++) {
			for (int j = 0; j < teach[0].length; j++) {
				error += Math.pow((teach[i][j] - output[i][j]), 2);
			}
		}
		error = error / (neurons * patterns);
		return error;
	}
}
