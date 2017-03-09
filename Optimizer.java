// Skeleton class used for constructing optimizing algorithm classes
// Provided by Dr. Michael S. Gashler
public abstract class Optimizer {
    
    Objective obj;
    NeuralNet net;
    
    Optimizer () {
    }
    
    Optimizer (Objective object) {
        obj = object;
    }
    
    abstract double iterate();
    
    // Optimizes function and evaluates within supervised learner optimization classes
    double optimize(int burnIn, int window, double thresh)
    {
        for(int i = 1; i < burnIn; i++)
            iterate();
        double error = iterate();
        while(true)
        {
            double prevError = error;
            for(int i = 1; i < window; i++)
                iterate();
            error = iterate();
            if((prevError - error) / prevError < thresh || error == 0.0)
                break;
        }
        return error;
    }
}
