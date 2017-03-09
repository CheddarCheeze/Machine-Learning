// Skeleton class used to help construct objectives
// Provided by Dr. Michael S. Gashler
public class Objective extends Function {
    
    Matrix feat;
    Matrix lab;
    Regressor reg;
    NeuralNet net;
    
    Objective (Matrix features, Matrix labels, Regressor regressor) {
        reg = regressor;
        feat = features;
        lab = labels;
    }
    
    Objective (Matrix features, Matrix labels, NeuralNet neuralnet) {
        net = neuralnet;
        feat = features;
        lab = labels;
    }

    // Needed for model classes, this was just to satisfy abstract size method for this class. No use for it here.
    @Override
    int size() {
        return 1;
    }
    
    @Override
    double evaluate (double[] in) {
        reg.setParams(in);
        return reg.measureSSE(feat, lab);
    }
}
