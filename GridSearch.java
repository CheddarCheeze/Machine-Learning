public class GridSearch extends Optimizer {
    
    double[] min;
    double[] max;
    double[] step;
    double[] cur;
    
    // Checks every single spot possible. Very accurate but very slow.
    GridSearch(double[] pmin, double[] pmax, double[] pstep) {
        min = pmin;
        max = pmax;
        step = pstep;
        cur = pmin;
    }
    
    // Not correctly coded, pseudocode below. Needs implementation and testing.
    @Override
    double iterate() {
        
        double besterr = super.obj.evaluate(cur);
        double[] bestVec = Vec.copy(cur);
        
        // If cur is the best vector yet found, remember it.
        
        for(int i = 0; i < cur.length; i++) {
            cur[i] += step[i];
            if(step[i] > max[i]) {
                step[i] = min[i];
            }
            else {
                break;
            }
        }
        return besterr;
        /*
        for(double y = ymin; y < ymax; y += ystep) {
            
            for(double x = xmin; x < xmax; x+= xstep) {
                
                double err = measure_SSE();
                if(err < besterr) {
                    besterr = err;
                    bestVec = vec; // Basically x and y into a vector
                }
            }
        }
        */
    }
}
