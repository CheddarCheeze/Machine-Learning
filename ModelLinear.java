// Model used for data that has a linear pattern
public class ModelLinear extends Function{
    
    @Override
    int size () {
        return 2;
    }
    
    // Return equation for a linear model (linear equation): y = m * x + b
    @Override
    double evaluate (double[] vals) {
        double m = vals[0];
        double b = vals[1];
        double x = vals[2];
        return (m * x + b);
    }
}
