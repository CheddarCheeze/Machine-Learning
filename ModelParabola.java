// Model used for data that most likely has a parabola for line-of-best-fit
public class ModelParabola extends Function{
    
    @Override
    int size () {
        return 2;
    }
    
    // Return parabola equation: y = m * x^2 + b
    @Override
    double evaluate (double[] vals) {
        double m = vals[0];
        double b = vals[1];
        double x = vals[2];
        return (m * (x * x) + b);
    }
}
