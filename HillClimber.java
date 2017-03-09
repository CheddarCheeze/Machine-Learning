import java.util.Arrays;

public class HillClimber extends Optimizer {
    double[] cur;
    double[] step;
    
    HillClimber() {
        super();
    }
    
    HillClimber(int size) {
        cur = new double[size];
        step = new double[size];
        
        Arrays.fill(cur, 0.0);
        Arrays.fill(step, 0.1);
    }

    // This constructor is called in Regressor class
    HillClimber(Objective ob, int size) {
        super(ob);
        
        // Size is provided in the Objective model class
        cur = new double[size];
        step = new double[size];
        
        // Start arrays filled with values
        Arrays.fill(cur, 0.0);
        Arrays.fill(step, 0.1);
    }
    
    // This constructor is called in NeuralNet class
    HillClimber(NeuralNet neuralnet) {
        
        super.net = neuralnet;
        
        cur = new double[net.totalv];
        step = new double[net.totalv];
        
        // cur will be all weights and bias from Neural Net
        net.copyToVector(cur);
        Arrays.fill(step, 0.1);
    }
    
    // Iterate over every point and step in different directions/sizes to determine the best point
    @Override
    double iterate() {
        
        // We will return the minimum error. Best error must start out high so we can compare and minimize errors.
        double bestError = 100000000.0;
        double tempError;
        // tr is initial cur value before trying different spots
        double tr;
        double evalCur;
        int flag;
        
        for(int i = 0; i < cur.length; i++) {
            
            // Flag will determine which step was the spot of best error for cur[i]
            flag = 0;
            tr = cur[i];        // Save original value in case moving is worse and the original spot was best
            
            // Compute 5 points, initial point, +-1.25, +-0.8 steps. Determine point of least error.
            tempError = net.evaluate(cur);
            cur[i] = cur[i] - 1.25 * step[i];
            evalCur = net.evaluate(cur);
            
            if (evalCur < tempError) {
                tempError = evalCur;
                flag = 1;
            }
            
            cur[i] = tr;
            cur[i] = cur[i] + 1.25 * step[i];
            evalCur = net.evaluate(cur);
            
            if (evalCur < tempError) {
                tempError = evalCur;
                flag = 2;
            }
            
            cur[i] = tr;
            cur[i] = cur[i] - 0.8 * step[i];
            evalCur = net.evaluate(cur);
            
            if (evalCur < tempError) {
                tempError = evalCur;
                flag = 3;
            }
            
            cur[i] = tr;
            cur[i] = cur[i] + 0.8 * step[i];
            evalCur = net.evaluate(cur);
            
            if (evalCur < tempError) {
                tempError = evalCur;
                flag = 4;
            }
            
            
            // Based on above operation, 1.25/0.8/initial spot was least error. Save that spot back to cur and change step size accordingly.
            switch (flag) {
                case 1:
                    cur[i] = tr - 1.25 * step[i];
                    step[i] *= 1.25;
                    break;
                case 2:
                    cur[i] = tr + 1.25 * step[i];
                    step[i] *= 1.25;
                    break;
                case 3:
                    cur[i] = tr - 0.8 * step[i];
                    step[i] *= 0.8;
                    break;
                case 4:
                    cur[i] = tr + 0.8 * step[i];
                    step[i] *= 0.8;
                    break;
                default:
                    cur[i] = tr;
                    break;
            }
            
            // Save best error for final return
            if (tempError < bestError) {
                bestError = tempError;
            }
        }
        //net.setValues(cur);
        return bestError;
    }
}
