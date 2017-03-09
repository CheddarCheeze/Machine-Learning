import java.util.Arrays;
import java.util.Random;

abstract class SupervisedLearner 
{
    /// Return the name of this learner
    abstract String name();

    /// Train this supervised learner
    abstract void train(Matrix features, Matrix labels);

    /// Make a prediction
    abstract void predict(double[] in, double[] out);
    abstract void predictPacket(double[] in, double[] out);
    
    int countPacketMisclassifications(Matrix features, Matrix labels)
    {
        if(features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
        double[] pred = new double[labels.cols()];
        int mis = 0;
        for(int i = 0; i < features.rows(); i++)
        {
            double[] feat = features.row(i);
            predictPacket(feat, pred);
            double[] lab = labels.row(i);
            for(int j = 0; j < lab.length; j++)
            {
                if(pred[j] != lab[j])
                    mis++;
            }
        }
        return mis;
    }
    
    /// Measures the misclassifications with the provided test data
    int countMisclassifications(Matrix features, Matrix labels)
    {
        if(features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
        double[] pred = new double[labels.cols()];
        int mis = 0;
        for(int i = 0; i < features.rows(); i++)
        {
            double[] feat = features.row(i);
            predict(feat, pred);
            double[] lab = labels.row(i);
            for(int j = 0; j < lab.length; j++)
            {
                // System.out.println("TEST: " + pred[j] + " " + lab[j]);
                if(pred[j] != lab[j])
                    mis++;
            }
        }
        return mis;
    }

    // Measures the Sum-Squared-Error of the matrices
    double measureSSE(Matrix features, Matrix labels) {

        if(features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");

        double[] prediction = new double[labels.rows()];
        double sse = 0.0;

        for(int i = 0; i < features.rows(); i++) {

            double[] feat = features.row(i);
            predict(feat, prediction);

            double[] lab = labels.row(i);

            for(int j = 0; j < lab.length; j++) {
                double d = prediction[j] - lab[j];
                sse += (d * d);
            }

        }
        return sse;
    }

    // Reps are repetitions, folds are splitting the data into training sets and test sets and only training on the training sets and test on test sets, and then rotating which of the folds are training/testing.
    double crossValidate(int reps, int folds, Matrix features, Matrix labels) {

        if(reps < 1 || folds < 2) {
            throw new IllegalArgumentException("Reps must be greater than 0. Folds cannot be less than 2.");
        }

        if(features.rows() != labels.rows()) {
            throw new IllegalArgumentException("Mismatching number of rows.");
        }

        if(folds > features.rows()) {
            throw new IllegalArgumentException("Folds cannot be greater than features rows.");
        }

        double sse = 0;

        for(int i = 0; i < reps; i++) {

            // Copy matrix and then randomly swap rows
            Matrix randomFeatures = new Matrix(features.rows(), features.cols());
            randomFeatures.copyBlock(0, 0, features, 0, 0, features.rows(), features.cols());

            Matrix randomLabels = new Matrix(labels.rows(), labels.cols());
            randomLabels.copyBlock(0, 0, labels, 0, 0, labels.rows(), labels.cols());

            Random rand = new Random();
            for(int j = features.rows(); j >= 2; j--) {
                int index = rand.nextInt(j);
                randomFeatures.swapRows(j-1, index);
                randomLabels.swapRows(j-1, index);
            }


            // Divide top matrices into folds
            int foldcount = randomFeatures.rows() / folds;
            //int foldextra = f.rows() % folds;

            // Divide into folds and copy training sets and test sets into matrices
            for(int j = 0; j < folds; j++) {
                Matrix training_features = new Matrix(randomFeatures.rows() - foldcount, randomFeatures.cols());
                Matrix testFold_Features = new Matrix(foldcount, randomFeatures.cols());
                randomFeatures.copyFoldBlock(testFold_Features, training_features, foldcount, j);

                Matrix training_labels = new Matrix(randomLabels.rows() - foldcount, randomLabels.cols());
                Matrix testFold_Labels = new Matrix(foldcount, randomLabels.cols());
                randomLabels.copyFoldBlock(testFold_Labels, training_labels, foldcount, j);

                train(training_features, training_labels);
                sse += measureSSE(testFold_Features, testFold_Labels);
            }
        }
        // Average the SSE to get a good representation of the true value
        return (sse / reps);
    }
    
    // Feature selection - Leave One Out
    double leaveOneOutCrossValidate(int i, int j, int[] importOut, Matrix trainingFeatures, Matrix trainingLabels, Matrix testFeatures, Matrix testLabels) {
        
        int leaveOneOut = 1;
        
        int[] oneOut = importOut;
        
                
            Matrix trainLOO_Features = new Matrix(trainingFeatures.rows(), (trainingFeatures.cols() - leaveOneOut)-i);
            trainLOO_Features.copyLOOBlock(trainingFeatures, j, oneOut);

            Matrix testLOO_Features = new Matrix(testFeatures.rows(), (testFeatures.cols() - leaveOneOut)-i);
            testLOO_Features.copyLOOBlock(testFeatures, j, oneOut);

            // There is only 1 label + we are only LOO for features selection
//                Matrix trainLOO_Labels = new Matrix(labels.rows(), labels.cols() - leaveOneOut);
//                trainLOO_Labels.copyLOOBlock(trainingLabels, j);
//                
//                Matrix testLOO_Labels = new Matrix(testLabels.rows(), testLabels.cols() - leaveOneOut);

            train(trainLOO_Features, trainingLabels);
            return measureSSE(testLOO_Features, testLabels);

            
            //System.out.println(j + " = " + msse);
        
        //System.out.println("WAT IS THIS " + i + " >>>>> " + position);
        
        //oneOut[i] = position;

        //System.out.println(Arrays.toString(oneOut));
        
    }
    
    // Resure the correct count when determining column to remove
    int correctColumn(int position, int[] oneOut) {
        Arrays.sort(oneOut);
        for(int i = 0; i < oneOut.length; i++) {
            if(position >= oneOut[i] && position < 65) {
                position++;
            }
            //System.out.print(oneOut[i] + " ");
        }
        //System.out.println();
        
        return position;
    }
    
}