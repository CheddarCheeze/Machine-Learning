// Learning algorithm
public class Regressor extends SupervisedLearner {
    
    private final Function model;
    public double[] params;
    
    Regressor (Function function) {
        model = function;
    }
    
    // Declare optimization technique here. Use Optimize method to train model
    @Override
    void train(Matrix features, Matrix labels){
        if(labels.cols() != 1)
            throw new RuntimeException("Expected labels to have only one column");
        Objective ob = new Objective(features, labels, this);
        
        HillClimber hc = new HillClimber(ob, model.size());
        hc.optimize(200, 50, 0.01);
        
        //GridSearch gs = new GridSearch(0.0, 3.0, 0.1);
    };
    
    // Pay attention to concatenate. Params/in matters when declaring variables/parameters in optimization classes
    @Override
    void predict(double[] in, double[] out){
        out[0] = model.evaluate(Vec.concatenate(params, in));
    };
    
    @Override // Not used in this class
    void predictPacket(double[] in, double[] out){
        out[0] = model.evaluate(Vec.concatenate(params, in));
    };
    
    // This is done in Objective class
    void setParams (double[] p) { 
        params = p;
    }
    
    @Override
    String name () { return "Regressor";}
}
