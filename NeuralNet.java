import java.util.ArrayList;
import java.util.Random;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.awt.Color;
import java.awt.Image;

public class NeuralNet extends SupervisedLearner {
    ArrayList<Layer> layers;
    int totalv;
    
    Matrix feat;
    Matrix lab;
    
    boolean splitdata = false;
    Matrix trainfeat;
    Matrix trainlab;
    Matrix validatefeat;
    Matrix validatelab;
    //HillClimber hc;
    
    double[] features;
    double[] labels;
    double[] pred;
    Matrix V;
    int width = 64;     // Width of image in pixels
    int height = 48;     // Height of image in pixels
    
    NeuralNet(int[] layer_sizes) {
        
        layers = new ArrayList<>(layer_sizes.length);
        totalv = 0;
        
        for(int i = 1; i < layer_sizes.length; i++) {
            layers.add(new Layer(layer_sizes[i-1], layer_sizes[i]));
            totalv += layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i];
        }
        // Random small values set to all weights and bias
        randomSetValues();
        
        // Project 11 requires special weighting and 2 different activation functions
        //piWeights();
        //project11ActivationChange();
        
        // Testing data below
        //double[] test = {0.1,0.1,0,0,0.1,-0.1,0.1,0.1,0,0.1,0.1,0.1,0.1,0.3,-0.1,0.1,-0.2};
        //double[] test = {0.1,0.1,0.1,0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.0,0.1,0.1,0.1,0.1,0.1,0.0,0.1};
        //setValues(test);
        //hc = new HillClimber(this);
    }
    
    // Change number of nodes/layers in network
    void changeLayers(int[] layer_sizes) {
        
        layers = new ArrayList<>(layer_sizes.length);
        totalv = 0;
        
        for(int i = 1; i < layer_sizes.length; i++) {
            layers.add(new Layer(layer_sizes[i-1], layer_sizes[i]));
            totalv += layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i];
        }
        
        // Random small values set to all weights and bias
        randomSetValues();
    }
    
    // Sets first layer to sin activation and second layer to identity activation.
    void project11ActivationChange() {
        layers.get(0).activationfunctionchange = 2;
        layers.get(1).activationfunctionchange = 3;
    }
    
    // Sets first layer to sin activation and second layer to identity activation.
    void projectCISRActivationChange() {
        layers.get(0).activationfunctionchange = 0;
        //layers.get(1).activationfunctionchange = 3;
    }
    
    // Assigns weights of multiples of pi for assignment 11 -- 1 layer pi weights & 1 layer random small weights
    final void piWeights() {
        int multiple = 0;
        int divider = 1;
        int divcount = 0;
        
        int layer = 0;
        int row;
        int numberofweights;
        for(int i = 0; i < totalv / 2; layer++) {
            row = 0;
            numberofweights = layers.get(layer).weights.rows() * layers.get(layer).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(layer).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                layers.get(layer).weights.row(row)[(j % (layers.get(layer).weights.cols()))] = Math.PI * ((multiple%50)+1) * 2;
                i++;
                multiple++;
                
                if(j == numberofweights-1) {
                    layers.get(layer).weights.row(row)[(j % (layers.get(layer).weights.cols()))] = 0.01;
                }
            }
            
            for(int k = 0; k < layers.get(layer).bias.length; k++) {
                if(divcount < 50) {
                    layers.get(layer).bias[k] = Math.PI / divider;
                    divcount++;
                }
                else {
                    divcount = 0;
                    divider = (divider % 2) + 1;
                    layers.get(layer).bias[k] = Math.PI / divider;
                }
                i++;
                
                if(k == layers.get(0).bias.length - 1 && layer == 0) {
                    layers.get(layer).bias[k] = 0;
                }
            }
        }
        
        Random random = new Random();
        layer = 1;
        for(int i = 202; i < totalv; layer++) {
            row = 0;
            numberofweights = layers.get(layer).weights.rows() * layers.get(layer).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(layer).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                layers.get(layer).weights.row(row)[(j % (layers.get(layer).weights.cols()))] = Math.max(0.03, 1.0 / numberofweights) * random.nextGaussian();
                i++;
            }
            
            for(int k = 0; k < layers.get(layer).bias.length; k++) {
                layers.get(layer).bias[k] = Math.max(0.03, 1.0 / layers.get(layer).bias.length) * random.nextGaussian();
                i++;
            }
        }
    }
    
    // New method, skips buffer and generates random values directly into weights and bias
    final void randomSetValues() {
        Random random = new Random();
        int layer = 0;
        int row;
        int numberofweights;
        for(int i = 0; i < totalv; layer++) {
            row = 0;
            numberofweights = layers.get(layer).weights.rows() * layers.get(layer).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(layer).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                layers.get(layer).weights.row(row)[(j % (layers.get(layer).weights.cols()))] = Math.max(0.03, 1.0 / numberofweights) * random.nextGaussian();
                i++;
            }
            
            for(int k = 0; k < layers.get(layer).bias.length; k++) {
                layers.get(layer).bias[k] = Math.max(0.03, 1.0 / layers.get(layer).bias.length) * random.nextGaussian();
                i++;
            }
        }
    }
    
    // Obsolete - replaced with randomSetValues() 10/10/16 [Delete later if no further instances are found]
    void generateRandomValues(double[] v) {
        Random random = new Random();
        int layer = 0;
        int numberofweights;
        for(int q = 0; q < v.length; layer++) {
            numberofweights = layers.get(layer).weights.rows() * layers.get(layer).weights.cols();
            for(int i = 0; i < numberofweights; i++) {
                v[q] = Math.max(0.03, 1.0 / numberofweights) * random.nextGaussian();
                q++;
            }
            
            for(int i = 0; i < layers.get(layer).bias.length; i++) {
                v[q] = Math.max(0.03, 1.0 / layers.get(layer).bias.length) * random.nextGaussian();
                q++;
            }
        }
        setValues(v);
    }
    
    // Sets values for weights and bias
    void setValues(double[] v) {
        int lay = 0;
        int row;
        int numberofweights;
        for(int i = 0; i < v.length; lay++) {
            row = 0;
            numberofweights = layers.get(lay).weights.rows() * layers.get(lay).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(lay).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                layers.get(lay).weights.row(row)[(j % (layers.get(lay).weights.cols()))] = v[i];
                i++;
            }
            
            for(int k = 0; k < layers.get(lay).bias.length; k++) {
                layers.get(lay).bias[k] = v[i];
                i++;
            }
        }
        
    }
    
    // Copies weights and bias to a vector
    void copyToVector(double[] v) {
        int lay = 0;
        int row;
        int numberofweights;
        for(int i = 0; i < v.length; lay++) {
            
            row = 0;
            numberofweights = layers.get(lay).weights.rows() * layers.get(lay).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(lay).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                v[i] = layers.get(lay).weights.row(row)[(j % (layers.get(lay).weights.cols()))];
                i++;
            }
            
            for(int k = 0; k < layers.get(lay).bias.length; k++) {
                v[i] = layers.get(lay).bias[k];
                i++;
            }
        }
    }
    
    // Prints every layer weights and bias
    void printLayerWB() {
        int lay = 0;
        int row;
        int numberofweights;
        for(int i = 0; i < totalv; lay++) {
            row = 0;
            numberofweights = layers.get(lay).weights.rows() * layers.get(lay).weights.cols();
            for(int j = 0; j < numberofweights; j++) {
                
                if(j % (layers.get(lay).weights.cols()) == 0 && j != 0) {
                    row++;
                }
                System.out.println(layers.get(lay).weights.row(row)[(j % (layers.get(lay).weights.cols()))]);
                i++;
            }
            
            for(int k = 0; k < layers.get(lay).bias.length; k++) {
                System.out.println(layers.get(lay).bias[k]);
                i++;
            }
        }
    }
    
    // Prints activation for every layer
    void printActivation() {
        for(int i = 0; i < layers.size(); i++) {
            for(int k = 0; k < layers.get(i).activation.length; k++) {
                System.out.println(layers.get(i).activation[k]);
            }
        }
    }
    
    // Prints blame for every layer
    void printBlame() {
        for(int i = 0; i < layers.size(); i++) {
            for(int k = 0; k < layers.get(i).blame.length; k++) {
                System.out.println(layers.get(i).blame[k]);
            }
        }
    }
    
    // Splits data to train/validate data sets, also has convergence detection
    void splitData (Matrix features, Matrix labels) {
        double testPortion = 0.3;
        int test = (int)(features.rows() * testPortion);
        int train = features.rows() - test;
        
        trainfeat = new Matrix(train, features.cols());
        trainlab = new Matrix(train, labels.cols());
        validatefeat = new Matrix(test, features.cols());
        validatelab = new Matrix(test, labels.cols());
        
        Random rand = new Random();
        int trainSet = 0;   // train set
        int testSet = 0;   // test set
        for(int m = 0; m < features.rows(); m++) {
            if(rand.nextInt(test + train) < train) {
                trainfeat.copyBlock(trainSet, 0, features, m, 0, 1, features.cols());
                trainlab.copyBlock(trainSet, 0, labels, m, 0, 1, labels.cols());
                train--;
                trainSet++;
            }
            else {
                validatefeat.copyBlock(testSet, 0, features, m, 0, 1, features.cols());
                validatelab.copyBlock(testSet, 0, labels, m, 0, 1, labels.cols());
                test--;
                testSet++;
            }
        }
        
        // Change activation function -- 0:tanh, 1:arctan, 2:sin, 3:identity
        for(int i = 0; i < layers.size(); i++) {
            layers.get(i).activationfunctionchange = 0;
        }
        
        double ratio;
        double learning_rate = 0.01;
        double momentum = 0.2;
        
        for(int i = 0; i < 500; i++) {
            // Check for convergence
            if(75 == i % 76) {
                ratio = measureSSE(validatefeat, validatelab) / measureSSE(trainfeat, trainlab);
                //System.out.println(ratio);
                if(ratio >= 0.54 || ratio <= 0.46) {
                    learning_rate += 0.001;
                }
            }
            else {
                //train_batch(feat, lab, 0.001);
                train_stochastic(trainfeat, trainlab, learning_rate, momentum);   
            }
        }
    }
    
    // Main training method
    @Override
    void train(Matrix features, Matrix labels) {
        //train_iNetAddress(features, labels);
        //train_supervised(features,labels);
        //train_time_series(features, labels);
        
        
        if(splitdata == true) {
            splitData(features,labels);
            return;
        }
        
        //Objective ob = new Objective(features, labels, this);
        feat = features;
        lab = labels;
        
        double learning_rate = 0.01;
        double momentum = 0.5;
        
        // Uncomment/comment to switch between batch and stochastic
        for(int i = 0; i < 100; i++) {
            //train_batch(feat, lab, 0.001);
            train_stochastic(features, labels, learning_rate, momentum);
        }
        
        // Used for HillClimber algorithm
        //hc.optimize(50, 10, 0.05);
        //hc.optimize(10, 2, 0.2);
        //hc.iterate();
        
    }
    
    // Predictions are assigned to Vector out
    @Override
    void predict(double[] in, double[] out) {
        //System.out.println(in[0] + " " + in[1]);
        layers.get(0).feedForward(in);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward(layers.get(i-1).activation);
        }
        System.arraycopy(layers.get(layers.size()-1).activation, 0, out, 0, layers.get(layers.size()-1).activation.length);
        //System.out.println(out[0]);
        //System.out.println(layers.get(layers.size()-1).activation[0] + " " + in[0]);
    }
    
    @Override
    void predictPacket(double[] in, double[] out) {
        // Checking if protocol LLMNR is present, if so, it is a Windows OS
        if(in[2] == 4 || in[2] == 5) {
            out[0] = 0;
            return;
        }
        layers.get(0).feedForward(in);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward(layers.get(i-1).activation);
        }
        System.arraycopy(layers.get(layers.size()-1).activation, 0, out, 0, layers.get(layers.size()-1).activation.length);
    }
    
    void predict11(double[] in, double[] out) {
        layers.get(0).feed11Forward(in);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward(layers.get(i-1).activation);
        }
        System.arraycopy(layers.get(layers.size()-1).activation, 0, out, 0, layers.get(layers.size()-1).activation.length);
    }
    
    // First step of backpropagation
    void forwardPropagate(double[] in) {
        layers.get(0).feedForward(in);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward(layers.get(i-1).activation);
        }
    }
    
    // Second step of backpropagation
    void backwardPropagate(double[] in) {
        layers.get(layers.size()-1).feedBackward(in);
        for(int i = layers.size()-1; i > 0; i--) {
            layers.get(i-1).backPropagate(layers.get(i));
        }
    }
    
    // Preparation for third step of backpropagation
    void updateDeltas(double[] in, double learnrate) {
        layers.get(0).update_deltas(in, learnrate);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).update_deltas(layers.get(i-1).activation, learnrate);
        }
    }
    
    // Third step of backpropagation
    void updateWeights(double learning_rate) {
        for(int i = 0; i < layers.size(); i++) {
            layers.get(i).update_weights(learning_rate);
        }
    }
    
    // Manages momentum
    void decayDeltas(double momentum) {
        for(int j = 0; j < layers.size(); j++) {
            layers.get(j).decay_deltas(momentum);
        }
    }
    
    // L1 and L2 regularization
    void decayWeights(double lambda, double learning_rate) {
        for(int j = 0; j < layers.size(); j++) {
            layers.get(j).decay_L1weights(lambda, learning_rate);
            layers.get(j).decay_L2weights(lambda, learning_rate);
        }
    }
    
    @Override
    String name() { return "NeuralNet"; }
    
    double evaluate (double[] in) {
        setValues(in);
        return measureSSE(feat, lab);
    }
    
    // Backpropataion
    void present_pattern(double[] features, double[] labels, double learnrate) {
        //0.03 for stochastic, 0.001 for batch
        
        // Feed features forward through all layers
        forwardPropagate(features);
        
        // Compute Blame
        backwardPropagate(labels);
        
        // Update Deltas
        updateDeltas(features,learnrate);
    }
    
    // Train with one batch (or mini-batch) of data
    void train_batch(Matrix features, Matrix labels, double learning_rate) {
        decayDeltas(0.0);
        
        for(int i = 0; i < features.rows(); i++) {
            present_pattern(features.row(i), labels.row(i), learning_rate);
        }
        
        updateWeights(learning_rate);
    }
    
    // Perform one epoch of training with stochastic gradient descent
    void train_stochastic(Matrix features, Matrix labels, double learning_rate, double momentum) {
        
        // Make an index that represents the row index of features/labels
        int[] m_pIndexes = new int[features.rows()];
        for(int i = 0; i < m_pIndexes.length; i++) {
            m_pIndexes[i] = i;
        }
        
        // Randomize the index
        Random rand = new Random();
        for(int i = features.rows(); i > 1; i--) {
            Vec.swap(m_pIndexes, m_pIndexes, i-1, rand.nextInt(i));
        }
        
        // Use the randomized index to randomly pick rows to use for gradient descent
        for(int i = 0; i < m_pIndexes.length; i++) {
            decayDeltas(momentum);
            present_pattern(features.row(m_pIndexes[i]), labels.row(m_pIndexes[i]), learning_rate);
            updateWeights(learning_rate);
        }
    }
    
    // Project 9
    void train_unsupervised(Matrix X) {
        int n = X.rows();
        int k = 2;          // k refers to the dimensionality of the belief vectors. The crane has 2 degrees of freedom (ball height and crane left/right) so we will have k = 2.
        double learning_rate = 0.1;
        Random rand = new Random();
        double[] V_blame;       // Blame for V
        //pred = new double[2];
        int t;
        
        V = new Matrix(n,k);
        V.setAll(0);
        
        for(int j = 0; j < 10; j++) {
            for(int i = 0; i < 10000000; i++) {
                t = rand.nextInt(X.rows());
                features = V.row(t);
                labels = X.row(t);
                
                forwardPropagate(features);
                backwardPropagate(labels);
                
                // Calculate blame for V
                V_blame = layers.get(0).transposeDotVec(layers.get(0).weights, layers.get(0).blame);
                
                decayDeltas(0.0);
                updateDeltas(features,learning_rate);
                updateWeights(learning_rate);
                
                // Update V
                Vec.add(V.row(t), V_blame);
            }
            learning_rate *= 0.75;
        }
    }
    
    void train_with_images(Matrix X) {
        int degreesOfFreedom = 2;      // Degrees of freedom
        features = new double[degreesOfFreedom+2];
        double learning_rate = 0.1;
        Random rand = new Random();
        int channels = X.cols() / (width * height);     // Number of color channels
        pred = new double[channels];       // Prediction of RGB values
        labels = new double[channels];     // Actual RGB values
        double[] V_blame;                  // Blame for V
        
        int n = X.rows();
        int t;      // Random value at most X.rows()
        int p;      // Random value at most width
        int q;      // Random value at most height
        int s;      // Random pixel and two pixels afterwards
        
        V = new Matrix(n, degreesOfFreedom);
        V.setAll(0);
        
        for(int j = 0; j < 10; j++) {
            for(int i = 0; i < 10000000; i++) {
                t = rand.nextInt(X.rows());
                p = rand.nextInt(width);
                q = rand.nextInt(height);
                
                // features = a vector containing p/width, q/height, and V[t]
                features[0] = (double)p / (double)width;
                features[1] = (double)q / (double)height;
                features[2] = V.row(t)[0];
                features[3] = V.row(t)[1];
                
                s = channels * (width * q + p);
                
                // label = the vector from X[t][s] to X[t][s + (channels - 1)]
                Vec.copyBlockVec(X.row(t), labels, s, channels);
                
                //predict(features,pred);
                forwardPropagate(features);
                
                backwardPropagate(labels);
                
                //Vec.println(layers.get(0).blame);
                V_blame = layers.get(0).transposeDotVec(layers.get(0).weights, layers.get(0).blame);
                
//                for(int m = 0; m < V_blame.length; m++) {
//                    System.out.println(V_blame[m]);
//                }
//                System.out.println("---------------------");
//                System.out.println(pred[0] + " " + pred[1] + " " + pred[2] + "     @@@     " + labels[0] + " " + labels[1] + " " + labels[2]);
//                System.out.println("---------------------");
                
                decayDeltas(0.0);
                updateDeltas(features,learning_rate);
                updateWeights(learning_rate);
                
                
                V.row(t)[0] += V_blame[2] * learning_rate;
                V.row(t)[1] += V_blame[3] * learning_rate;
            }
            learning_rate *= 0.75;
                
            // Keeps track of progress
            System.out.println(j);
        }
        
        V.saveARFF("savedV.arff");
        String weightname;
        for(int i = 0; i < layers.size(); i++) {
            weightname = "layer" + i + "weights.arff";
            layers.get(i).weights.saveARFF(weightname);
            weightname = "layer" + i + "bias.arff";
            Vec.saveArray(layers.get(i).bias, weightname);
        }
    }
    
    void train_supervised(Matrix state, Matrix naction) {
        features = new double[6];
        pred = new double[2];
        labels = new double[2];
        Random rand = new Random();
        double learning_rate = 0.1;
        int t;
//        System.out.println(state.row(0)[0] + " " + state.row(0)[1]);
        
        for(int j = 0; j < 1; j++) {
            for(int i = 0; i < 100000; i++) {
                t = rand.nextInt(state.rows()-1);
                features[0] = state.row(t)[0];
                features[1] = state.row(t)[1];
                features[2] = naction.row(t)[0];
                features[3] = naction.row(t)[1];
                features[4] = naction.row(t)[2];
                features[5] = naction.row(t)[3];
                labels = state.row(t+1);
                
                decayDeltas(0.0);
                //predict(features,pred);
                forwardPropagate(features);
                backwardPropagate(labels);
                updateDeltas(features,learning_rate);
                updateWeights(learning_rate);
            }
            learning_rate *= 0.75;
        }
    }
    
    double L1Regularization(int layer, double lambda) {
        double sum = 0;
        for(int i = 0; i < layers.get(layer).weights.cols(); i++) {
            sum += Math.abs(layers.get(layer).weights.row(0)[i]);
        }
        sum = sum * lambda;
        return (1.0-sum);
    }
    
    double L2Regularization(int layer, double lambda) {
        double sum = 0;
        for(int i = 0; i < layers.get(layer).weights.rows(); i++) {
            sum += Math.pow(layers.get(layer).weights.row(i)[0], 2);
        }
        sum = sum * lambda;
        return (1.0-sum);
    }
    
    // Regularization, changing actication function for 1 unit - Project 11
    void present_11pattern(double[] features, double[] labels, double learnrate) {
        // feed11Forward and feed11Backward change activation function for last unit in first layer
        
        layers.get(0).feed11Forward(features);
        for(int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward(layers.get(i-1).activation);
        }
        
        layers.get(layers.size()-1).feedBackward(labels);
        layers.get(0).feed11Backward(layers.get(1));
        
        updateDeltas(features,learnrate);
    }
    
    void train_time_series(Matrix features, Matrix labels) {
        double learning_rate = 0.01;
        double L1lambda = 0.0001;
        double L2lambda = 0.00023;
        for(int j = 0; j < 5000; j++) {
            for(int i = 0; i < features.rows(); i++) {
                decayDeltas(0.0);
                
                //present_pattern11 changes the activation function of the last node of 1st layer
                present_11pattern(features.row(i),labels.row(i),learning_rate); 
                
                // L1 Regularization by every 5th training seem to give me better results
                if(i % 5 == 0) {
                    layers.get(1).decay_L1weights(L1lambda, learning_rate);
                }

                // L2 Regularization
                //layers.get(1).decay_L2weights(L2lambda, learning_rate);
                
                updateWeights(learning_rate);
            }
        }
        
        Matrix plot = new Matrix(labels.rows(), 2);
        
        double[] featurino = new double[1];
        pred = new double[1];
        for(int i = 0; i < 356; i++) {
            plot.row(i)[0] = (double)i / 256.0;
            featurino[0] = (double)i / 256.0;
            predict11(featurino, pred);         //predict11 changes the activation function of the last node of 1st layer
            plot.row(i)[1] = pred[0];
        }
        plot.saveARFF("project11b.arff");
        
    }
    
    void train_iNetAddress(Matrix destPrefix, Matrix hops) {
        features = new double[64];
        pred = new double[1];
        labels = new double[1];
//        Random rand = new Random();
        double learning_rate = 0.1;
//        int t; // Used for random number
        
        // Make an index that represents the row index of features/labels
//        int[] m_pIndexes = new int[destPrefix.rows()];
//        for(int i = 0; i < m_pIndexes.length; i++) {
//            m_pIndexes[i] = i;
//        }
        
        // Randomize the index
//        for(int i = destPrefix.rows(); i > 1; i--) {
//            Vec.swap(m_pIndexes, m_pIndexes, i-1, rand.nextInt(i));
//        }
        
        for(int j = 0; j < 10; j++) {
            for(int i = 0; i < destPrefix.rows(); i++) {
                features = destPrefix.row(i);
                labels = hops.row(i);
                
                decayDeltas(0.0);
                //predict(features,pred);
                forwardPropagate(features);
                backwardPropagate(labels);
                updateDeltas(features,learning_rate);
                updateWeights(learning_rate);
                
            }
            learning_rate *= 0.75;
        }
    }
    
    
    
    
    
    
    void imagePredict() {
        String imagename;
        for(int i = 0; i < V.rows(); i++) {
            imagename = "makeImage/frame" + i + ".png";
            makeImage(V.row(i), imagename);
        }
    }
   
    int rgbToUint(int r, int g, int b)
    {
        return (0xff000000 | ((r & 0xff) << 16)) | ((g & 0xff) << 8) | ((b & 0xff));
    }

    void makeImage(double[] state, String filename)
    {
        double[] in = new double[4];
        double[] out = new double[3];
        in[2] = state[0];
        in[3] = state[1];
        
        BufferedImage im = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        
        int red;
        int green;
        int blue;

        for(int y = 0; y < height; y++)
        {
            in[1] = (double)y / height;
            for(int x = 0; x < width; x++)
            {
              in[0] = (double)x / width;
              predict(in, out);
              
              red = (int)Math.abs(out[0] * 255);
              green = (int)Math.abs(out[1] * 255);
              blue = (int)Math.abs(out[2] * 255);
              
              if(red > 255) {
                  red = 255;
              }
              if(green > 255) {
                  green = 255;
              }
              if(blue > 255) {
                  blue = 255;
              }
              if(red < 0) {
                  red = 0;
              }
              if(green < 0) {
                  green = 0;
              }
              if(blue < 0) {
                  blue = 0;
              }
              
              int color = rgbToUint(red, green, blue);
              //System.out.println(red + " " + green + " " + blue);
              //System.out.println(in[0] + " " + in[1] + " " + in[2] + " " + in[3]);
              im.setRGB(x, y, color);
            }
        }
        
        File outputFile = new File(String.valueOf(filename));
        try{
            ImageIO.write(im, "png", outputFile); 
        }
        catch (Exception e) {
            System.out.println("Image print fail!");
        }
        
    }
    
     public static int rgbToInt(int red, int green, int blue){
        int rgb = red;
        rgb = (rgb << 8) + green;
        rgb = (rgb << 8) + blue;
        return rgb;
     }
}
