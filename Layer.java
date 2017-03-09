import static java.lang.Math.tanh;
import static java.lang.Math.atan;

public class Layer {
    Matrix weights;
    double[] bias;
    double[] net;
    double[] activation;
    double[] blame;
    Matrix weightDelta;
    double[] biasDelta;
    int activationfunctionchange;
    
    Layer(int inputs, int outputs) {
        
        if(inputs < 0 || outputs < 0) {
            throw new RuntimeException("Inputs/Outputs in Layer Class is less than 0.");
        }
        
        // Initialize all variables
        weights = new Matrix(outputs, inputs);
        bias = new double[outputs];
        net = new double[outputs];
        activation = new double[outputs];
        
        blame = new double[outputs];
        weightDelta = new Matrix(outputs, inputs);
        biasDelta = new double[outputs];
        
        activationfunctionchange = 0;
    }
    
    // We can add additional activation functions - Add here and below
    double choiceOfActivationFunction(double x) {
        double a;
        switch (activationfunctionchange) {
            case 0: // TanH
                a = tanh(x);
                break;
            case 1: // ArcTan
                a = atan(x);
                break;
            case 2: // Sin
                a = Math.sin(x);
                break;
            case 3: // Identity
                a = x;
                break;
            case 4: // Logistic
                a = (1 / (1 + Math.pow(Math.E, -x)));
                break;
            case 5: // SoftPlus
                a = Math.log(1 + Math.pow(Math.E, -x));
                break;
            default:
                a = 0;
                break;
        }
        return a;
    }
    
    // We can add additional activation functions - Add here and above
    double activationDerivative (double x) { 
        double a;
        switch (activationfunctionchange) {
            case 0: // TanH
                a = 1 - Math.pow(x, 2);
                break;
            case 1: // ArcTan
                a = 1 / (Math.pow(x, 2) + 1);
                break;
            case 2: // Sin
                a = Math.cos(x);
                break;
            case 3: // Identity
                a = 1;
                break;
            case 4: // Logistic
                a = (1 / (1 + Math.pow(Math.E, -x))) * (1 - (1 / (1 + Math.pow(Math.E, -x))));
                break;
            case 5: // SoftPlus
                a = (1 / (1 + Math.pow(Math.E, -x)));
                break;
            default:
                a = 0;
                break;
        }
        return a;
    }
    
    // Sending data through the weights/bias and computing with an activation function
    // Matrix weights * Vector features + Vector bias = Vector net
    // Send net through activation function: Vector activation = ActivationFunction(Vector net)
    void feedForward(double[] x) {
        net = matrixDotVec(weights, x);
        Vec.add(net, bias);
        for(int i = 0; i < net.length; i++) {
            activation[i] = choiceOfActivationFunction(net[i]);
        }
    }
    
    // Change activation function for one unit
    void feed11Forward(double[] x) {
        net = matrixDotVec(weights, x);
        Vec.add(net, bias);
        for(int i = 0; i < net.length-1; i++) {
            activation[i] = choiceOfActivationFunction(net[i]);
        }
        activation[net.length-1] = net[net.length-1];
    }
    
    // First step of back propagation - Feed in labels
    void feedBackward(double[] x) {
        for(int i = 0; i < activation.length; i++) {
            blame[i] = (x[i] - activation[i]) * activationDerivative(activation[i]);
        }
    }
    
    // Change derivative of activation function for one unit
    void feed11Backward(Layer from) {
        double[] dotPro = transposeDotVec(from.weights, from.blame);
        for(int i = 0; i < activation.length-1; i++) {
            blame[i] = dotPro[i] * activationDerivative(activation[i]);
        }
        blame[activation.length-1] = dotPro[activation.length-1];
    }
    
    // Recurring finishing step of back propagation - Feed in weights/bias of previous layer
    void backPropagate(Layer from) {
        double[] dotPro = transposeDotVec(from.weights, from.blame);
        for(int i = 0; i < activation.length; i++) {
            blame[i] = dotPro[i] * activationDerivative(activation[i]);
        }
    }
    
    // Decay to forget older data - 0 is do not memorize any data, 1 is memorize most previous data
    void decay_deltas(double momentum) {
        for(int i = 0; i < weightDelta.rows(); i++) {
            for(int j = 0; j < weightDelta.cols(); j++) {
                weightDelta.row(i)[j] *= momentum;
//                if(Math.abs(weightDelta.row(i)[j]) < 0.001) {
//                    System.out.println(weightDelta.row(i)[j]);
//                }
            }
            biasDelta[i] *= momentum;
        }
    }
    
    // Decay weights to L1 regularize
    void decay_L1weights(double lambda, double learning_rate) {
        for(int i = 0; i < weightDelta.rows(); i++) {
            for(int j = 0; j < weightDelta.cols(); j++) {
                if(weights.row(i)[j] > 0) {
                    weights.row(i)[j] -= lambda * learning_rate;
                }
                else {
                    weights.row(i)[j] += lambda * learning_rate;
                }
            }
            if(bias[i] > 0) {
                bias[i] -= lambda * learning_rate;
            }
            else {
                bias[i] += lambda * learning_rate;
            }
        }
    }
    
    // Decay weights to L2 regularize
    void decay_L2weights(double lambda, double eta) {
        for(int i = 0; i < weightDelta.rows(); i++) {
            for(int j = 0; j < weightDelta.cols(); j++) {
                weights.row(i)[j] *= (1.0 - (lambda * eta));
            }
            bias[i] *= (1.0 - (lambda * eta));
        }
    }
    
    // Deltas are used to update weights on the third step of backpropagation
    void update_deltas(double[] x, double learning_rate) {
        for(int i = 0; i < weightDelta.rows(); i++) {
            for(int j = 0; j < weightDelta.cols(); j++) {
                weightDelta.row(i)[j] += blame[i] * x[j];
            }
            biasDelta[i] += blame[i] * learning_rate;
        }
    }
    
    // Third step of backpropagation
    void update_weights(double learning_rate) {
        for(int i = 0; i < weights.rows(); i++) {
            for(int j = 0; j < weights.cols(); j++) {              
                weights.row(i)[j] += weightDelta.row(i)[j] * learning_rate;
            }
            bias[i] += biasDelta[i];
        }
    }
    
    // Multiplying a matrix with a vector
    double[] matrixDotVec(Matrix matrix, double[] vector) {
        double[] v = new double[matrix.rows()];
        int pos = 0;
        for(int y = 0; y < matrix.rows(); y++) {
            for(int x = 0; x < matrix.cols(); x++) {
                v[pos] += matrix.row(y)[x] * vector[x];
            }
            pos++;
        }
        return v;
    }
    
    // Transposing the matrix, then multiplying with a vector
    double[] transposeDotVec(Matrix matrix, double[] vector) {
        double[] v = new double[matrix.cols()];
        int pos = 0;
        for(int y = 0; y < matrix.cols(); y++) {
            for(int x = 0; x < matrix.rows(); x++) {
                v[pos] += matrix.row(x)[y] * vector[x];
            }
            pos++;
        }
        return v;
    }
    
}
