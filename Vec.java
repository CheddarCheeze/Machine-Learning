
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

/** Provides static methods for operating on arrays of doubles */
// Provided by Dr. Michael S. Gashler
public class Vec
{
	public static void print(double[] vec) {
		System.out.print("[");
		if(vec.length > 0) {
			System.out.print(Double.toString(vec[0]));
			for(int i = 1; i < vec.length; i++) {
				System.out.print("," + Double.toString(vec[i]));
			}
		}
		System.out.print("]");
	}

	public static void println(double[] vec) {
		print(vec);
		System.out.println();
	}

	public static void setAll(double[] vec, double val) {
		for(int i = 0; i < vec.length; i++)
			vec[i] = val;
	}

	public static double squaredMagnitude(double[] vec) {
		double d = 0.0;
		for(int i = 0; i < vec.length; i++)
			d += vec[i] * vec[i];
		return d;
	}

	public static void normalize(double[] vec) {
            double mag = squaredMagnitude(vec);
            if(mag <= 0.0) {
                setAll(vec, 0.0);
                vec[0] = 1.0;
            } else {
                double s = 1.0 / Math.sqrt(mag);
                for(int i = 0; i < vec.length; i++)
                    vec[i] *= s;
            }
	}

	public static void copy(double[] dest, double[] src) {
            if(dest.length != src.length)
                throw new IllegalArgumentException("mismatching sizes");
            System.arraycopy(src, 0, dest, 0, src.length);
	}

	public static double[] copy(double[] src) {
		double[] dest = new double[src.length];
                System.arraycopy(src, 0, dest, 0, src.length);
		return dest;
	}

	public static void add(double[] dest, double[] src) {
		if(dest.length != src.length)
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < dest.length; i++) {
			dest[i] += src[i];
		}
	}

	public static void scale(double[] dest, double scalar) {
		for(int i = 0; i < dest.length; i++) {
			dest[i] *= scalar;
		}
	}

	public static double dotProduct(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++)
			d += a[i] * b[i];
		return d;
	}

	public static double squaredDistance(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++) {
			double t = a[i] - b[i];
			d += t * t;
		}
		return d;
	}

	public static void clip(double[] vec, double min, double max) {
		if(max < min)
			throw new IllegalArgumentException("max must be >= min");
		for(int i = 0; i < vec.length; i++) {
			vec[i] = Math.max(min, Math.min(max, vec[i]));
		}
	}

	public static double[] concatenate(double[] a, double[] b) {
		double[] c = new double[a.length + b.length];
            System.arraycopy(a, 0, c, 0, a.length);
            System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}
        
        public static Matrix outer_product(double[] a, double[] b) {
            Matrix c = new Matrix(a.length, b.length);
            for(int i = 0; i < a.length; i++) {
                for(int j = 0; j < b.length; j++) {
                    c.row(i)[j] = a[i] * b[j];
                }
            }
            return c;
        }
        
        public static void swap(int[] a, int b[], int index1, int index2) {
            int c = a[index1];
            a[index1] = b[index2];
            b[index2] = c;
        }

        public static void swap(double[] a, double b[], int index1, int index2) {
            double c = a[index1];
            a[index1] = b[index2];
            b[index2] = c;
        }
        
        // Used in NeuralNet train_with_images()
        public static void copyBlockVec(double[] source, double[] dest, int start, int end) {
            for(int i = 0; i < end; i++) {
                dest[i] = source[start+i];
            }
        }
        
        public static void addByLearningRate(double[] dest, double[] src, double learning_rate) {
		if(dest.length != src.length)
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < dest.length; i++) {
			dest[i] += src[i] * learning_rate;
		}
	}
        
        public static void saveArray(double[] src, String filename) {
            
            PrintWriter os = null;

            try
            {
                os = new PrintWriter(filename);
                os.println(src.length);
                for(int i = 0; i < src.length; i++) {
                    os.println(src[i]);
                }
            }
            catch (FileNotFoundException e)
            {
                throw new IllegalArgumentException("Error creating file: " + filename + ".");
            }
            finally
            {
                os.close();
            }
        }
        
        public static double[] loadArray(double[] dst, String filename) {
            Scanner s = null;
            
            try
            {
                s = new Scanner(new File(filename));
                dst = new double[s.nextInt()];
                int spot = 0;
                while (s.hasNextDouble())
                {
                    dst[spot] = s.nextDouble();
                    spot++;
                }
            }
            catch (FileNotFoundException e)
            {
                throw new IllegalArgumentException("Failed to open file: " + filename + ".");
            }
            finally
            {
                s.close();
            }
            return dst;
        }
}
