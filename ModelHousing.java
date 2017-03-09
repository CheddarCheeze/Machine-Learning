// Model used for housing data
public class ModelHousing extends Function{
    
    // 14 parameters, m0-m12 and b
    @Override
    int size () {
        return 14;
    }
    
    @Override
    double evaluate (double[] vals) {
        
        double m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12;
        double x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12;
        double b;
        
        // Order seems to matter, m0-12, then b, then x0-12
        m0 = vals[0];
        m1 = vals[1];
        m2 = vals[2];
        m3 = vals[3];
        m4 = vals[4];
        m5 = vals[5];
        m6 = vals[6];
        m7 = vals[7];
        m8 = vals[8];
        m9 = vals[9];
        m10 = vals[10];
        m11 = vals[11];
        m12 = vals[12];
        b = vals[13];
        x0 = vals[14];
        x1 = vals[15];
        x2 = vals[16];
        x3 = vals[17];
        x4 = vals[18];
        x5 = vals[19];
        x6 = vals[20];
        x7 = vals[21];
        x8 = vals[22];
        x9 = vals[23];
        x10 = vals[24];
        x11 = vals[25];
        x12 = vals[26];
        
        // This return equation was provided on step 15: Calculates the label given these values (features).
        return ((m0*x0)+(m1*x1)+(m2*x2)+(m3*x3)+(m4*x4)+(m5*x5)+(m6*x6)+(m7*x7)+(m8*x8)+(m9*x9)+(m10*x10)+(m11*x11)+(m12*x12)+b);
    }
}
