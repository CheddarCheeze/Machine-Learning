import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.Arrays;
import java.util.Date;
import java.util.stream.IntStream;

class Main
{
	static void test(SupervisedLearner learner, String challenge)
	{
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testLearner(SupervisedLearner learner)
	{
		test(learner, "hep");
		test(learner, "vow");
		test(learner, "soy");
	}
        
        public static void unittestsProject1() {
            // Print user working directory
            //System.out.println(System.getProperty("user.dir"));
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
                
            
            /* Assignment 1
            // Linear regressor model
            Regressor r = new Regressor(new ModelLinear());
            
            // Load linear sample data
            Matrix tFeatures = new Matrix();
            tFeatures.loadARFF("sample_1.arff");
            Matrix tLabels = new Matrix();
            tLabels.loadARFF("sample_2.arff");
            
            // Train linear model and compute SSE
            r.train(tFeatures,tLabels);
            //Vec.println(r.params);
            double sse = r.measureSSE(tFeatures, tLabels);
            
            // Print linear SSE value
            System.out.println("sse_linear=" + sse);
            
            // Parabola regressor model
            Regressor s = new Regressor(new ModelParabola());
            
            // Train parabola model with linear data and compute SSE. Print SSE value
            s.train(tFeatures, tLabels);
            //Vec.println(s.params);
            double sse_parabola = s.measureSSE(tFeatures, tLabels);
            System.out.println("sse_parabola=" + sse_parabola);
            

            // Load housing data
            Matrix housingFeatures = new Matrix();
            housingFeatures.loadARFF("housing_features.arff");
            Matrix housingLabels = new Matrix();
            housingLabels.loadARFF("housing_labels.arff");
            
            // Housing regressor model
            Regressor t = new Regressor(new ModelHousing());
            
            // Train housing model with housing data and compute SSE
            t.train(housingFeatures, housingLabels);
            //Vec.println(t.params);
            double sse_housing = t.measureSSE(housingFeatures, housingLabels);
            
            // Print housing SSE
            System.out.println("sse_housing=" + sse_housing);
            
            /* End of Assignment 1     */
        }
        
        public static void unittestsProject2() {
            
            int[] numberOfLayers = {13, 8, 1};
            NeuralNet net = new NeuralNet(numberOfLayers);
            
            Filter fx = new Filter(net, new Normalizer(), true);
            Filter fy = new Filter(fx, new Normalizer(), false);
            
            
            // Load housing data
            Matrix trainFeatures = new Matrix();
            trainFeatures.loadARFF("housing_features.arff");
            Matrix trainLabels = new Matrix();
            trainLabels.loadARFF("housing_labels.arff");
            
            
            double testPortion = 0.5;
            int test = (int)(trainFeatures.rows() * testPortion);
            int train = trainFeatures.rows() - test;
            
            Matrix testSet_f = new Matrix(test, trainFeatures.cols());
            Matrix testSet_l = new Matrix(test, trainLabels.cols());
            Matrix trainingSet_f = new Matrix(train, trainFeatures.cols());
            Matrix trainingSet_l = new Matrix(train, trainLabels.cols());
            
            
            // This is Splitting the data then training on one set and testing 1 on train set and 1 on test set
            Random rand = new Random();
            int teSe = 0;
            int trSe = 0;
            for(int m = 0; m < trainFeatures.rows(); m++) {
                if(rand.nextInt(test + train) < test) {
                    testSet_f.copyBlock(teSe, 0, trainFeatures, m, 0, 1, trainFeatures.cols());
                    testSet_l.copyBlock(teSe, 0, trainLabels, m, 0, 1, trainLabels.cols());
                    test--;
                    teSe++;
                }
                else {
                    trainingSet_f.copyBlock(trSe, 0, trainFeatures, m, 0, 1, trainFeatures.cols());
                    trainingSet_l.copyBlock(trSe, 0, trainLabels, m, 0, 1, trainLabels.cols());
                    train--;
                    trSe++;
                }
            
            }
                
            /*  // Uncomment this to get values for graph
            for(int m = 0; m < 300; m++) {
                fy.train(trainingSet_f, trainingSet_l);
                double trainsse = Math.sqrt(fy.measureSSE(trainingSet_f, trainingSet_l)/trainingSet_f.rows());
                //System.out.println(trainsse + " " + ((double)System.nanoTime() * 1e9));
                
                try(FileWriter fw = new FileWriter("trainsse.txt", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)) {
                        out.println(Double.toString(trainsse) + " " + ((double)System.nanoTime() * 1e9));
                        //more code
                    } catch (IOException e) {
                        //exception handling left as an exercise for the reader
                    }
            
            
                double testsse = Math.sqrt(fy.measureSSE(testSet_f, testSet_l)/trainingSet_f.rows());
                //System.out.println(testsse + " " + ((double)System.nanoTime() * 1e9));
            
                try(FileWriter fw = new FileWriter("testsse.txt", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)) {
                        out.println(Double.toString(testsse) + " " + ((double)System.nanoTime() * 1e9));
                        //more code
                    } catch (IOException e) {
                        //exception handling left as an exercise for the reader
                    }
            }
            */
            
           
            
             // This is training and testing on all data
            double rmse = fy.crossValidate(2, 10, trainFeatures, trainLabels);
            rmse = Math.sqrt(rmse/(trainFeatures.rows()));
            System.out.println("rmse=" + rmse);
            
            
            /* // This is used for testing steps 1-3 on assignment 3
            double[] x = {-1, 0.5, 2};
            double[] v = {0.1,0.1,0.1,0.1,0.3,-0.1,0.1,-0.2};
            
            net.setValues(v);
            
            net.layers.get(0).feedForward(x);
            for(int i = 0; i < net.layers.get(0).activation.length; i++) {
                System.out.println(net.layers.get(0).activation[i]);
            }
            /*
            layer.feedForward(x);
            
            for(int i = 0; i < layer.activation.length; i++) {
                System.out.println(layer.activation[i]);
            }
            */
        }

        public static void unittestsProject3() {
            
            // misclassifications with 1 layer
            //===============================================================
            
            //int[] numberOfLayers = {13, 8, 1};          // Housing data
            int[] numberOfLayers = {27,11};           // Vowel sounds
            NeuralNet net = new NeuralNet(numberOfLayers);
            
            Filter fm = new Filter(net, new NomCat(), true);
            Filter fn = new Filter(fm, new NomCat(), false);
            Filter fx = new Filter(fn, new Normalizer(), true);
            Filter fy = new Filter(fx, new Normalizer(), false);
            
            // Load train data
            Matrix vtrain_features = new Matrix();
            vtrain_features.loadARFF("vowel_train_features.arff"); // 528 rows, 12 cols
            Matrix vtrain_labels = new Matrix();
            vtrain_labels.loadARFF("vowel_train_labels.arff"); // 528 rows, 1 col
            
            // Load test data
            Matrix vtest_features = new Matrix();
            vtest_features.loadARFF("vowel_test_features.arff"); //462 rows, 12 cols
            Matrix vtest_labels = new Matrix();
            vtest_labels.loadARFF("vowel_test_labels.arff"); // 462 rows, 1 col
            
            fy.train(vtrain_features, vtrain_labels);
//            double rmse = fy.crossValidate(2, 10, vtrain_features, vtrain_labels);
//            rmse = Math.sqrt(rmse/(vtrain_features.rows()));
//            System.out.println("train_rmse=" + rmse);
            System.out.println("mis1=" + (double)fy.countMisclassifications(vtest_features, vtest_labels)/(double)vtest_features.rows());
            
            
            
            
            // misclassifications with 2 layer
            //===============================================================
            
            int[] numbers = {27,20,11};           // Vowel sounds
            NeuralNet nets = new NeuralNet(numbers);
            
            Filter fms = new Filter(nets, new NomCat(), true);
            Filter fns = new Filter(fms, new NomCat(), false);
            Filter fxs = new Filter(fns, new Normalizer(), true);
            Filter fys = new Filter(fxs, new Normalizer(), false);
            
            fys.train(vtrain_features, vtrain_labels);
            System.out.println("mis2=" + (double)fys.countMisclassifications(vtest_features, vtest_labels)/(double)vtest_features.rows());
            
            
            
            
            // misclassifications with 2 layer, split data and validate, and convergence detection
            //===============================================================
            
            int[] number = {27,20,11};           // Vowel sounds
            NeuralNet netr = new NeuralNet(number);
            netr.splitdata = true;
            
            Filter fmr = new Filter(netr, new NomCat(), true);
            Filter fnr = new Filter(fmr, new NomCat(), false);
            Filter fxr = new Filter(fnr, new Normalizer(), true);
            Filter fyr = new Filter(fxr, new Normalizer(), false);
            
            fyr.train(vtrain_features, vtrain_labels);
            System.out.println("mis3=" + (double)fyr.countMisclassifications(vtest_features, vtest_labels)/(double)vtest_features.rows());
            
            
//            fy.train(vtest_features, vtest_labels);
//            double rmseT = fy.crossValidate(2, 10, vtest_features, vtest_labels);
//            rmseT = Math.sqrt(rmse/(vtest_features.rows()));
//            System.out.println("test_rmse=" + rmseT);
//            System.out.println((double)fy.countMisclassifications(vtest_features, vtest_labels)/(double)vtest_features.rows());
            
            //----------------------------------------------------------------
            
            /*
            
            // Load housing data
            Matrix trainFeatures = new Matrix();
            trainFeatures.loadARFF("housing_features.arff"); // 506 rows, 13 cols
            Matrix trainLabels = new Matrix();
            trainLabels.loadARFF("housing_labels.arff"); // 506 rows, 1 col
            
            double testPortion = 0.5;
            int test = (int)(trainFeatures.rows() * testPortion);
            int train = trainFeatures.rows() - test;
            
            Matrix testSet_f = new Matrix(test, trainFeatures.cols());
            Matrix testSet_l = new Matrix(test, trainLabels.cols());
            Matrix trainingSet_f = new Matrix(train, trainFeatures.cols());
            Matrix trainingSet_l = new Matrix(train, trainLabels.cols());
            
            
            // This is Splitting the data then training on one set and testing 1 on train set and 1 on test set
            Random rand = new Random();
            int teSe = 0;
            int trSe = 0;
            for(int m = 0; m < trainFeatures.rows(); m++) {
                if(rand.nextInt(test + train) < test) {
                    testSet_f.copyBlock(teSe, 0, trainFeatures, m, 0, 1, trainFeatures.cols());
                    testSet_l.copyBlock(teSe, 0, trainLabels, m, 0, 1, trainLabels.cols());
                    test--;
                    teSe++;
                }
                else {
                    trainingSet_f.copyBlock(trSe, 0, trainFeatures, m, 0, 1, trainFeatures.cols());
                    trainingSet_l.copyBlock(trSe, 0, trainLabels, m, 0, 1, trainLabels.cols());
                    train--;
                    trSe++;
                }
            
            }
            
            for(int m = 0; m < 300; m++) {
                fy.train(trainingSet_f, trainingSet_l);
                double trainsse = Math.sqrt(fy.measureSSE(trainingSet_f, trainingSet_l)/trainingSet_f.rows());
                //System.out.println(trainsse + " " + ((double)System.nanoTime() * 1e9));
                
                try(FileWriter fw = new FileWriter("trainsse.txt", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)) {
                        out.println(Double.toString(trainsse) + " " + ((double)System.nanoTime() * 1e9));
                        //more code
                    } catch (IOException e) {
                        //exception handling left as an exercise for the reader
                    }
            
            
                double testsse = Math.sqrt(fy.measureSSE(testSet_f, testSet_l)/trainingSet_f.rows());
                //System.out.println(testsse + " " + ((double)System.nanoTime() * 1e9));
            
                try(FileWriter fw = new FileWriter("testsse.txt", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)) {
                        out.println(Double.toString(testsse) + " " + ((double)System.nanoTime() * 1e9));
                        //more code
                    } catch (IOException e) {
                        //exception handling left as an exercise for the reader
                    }
            }
            */
            
            //-----------------------------------------------------------------
            
            
            
            // This is training and testing on all data
            //double rmse = fy.crossValidate(2, 10, trainFeatures, trainLabels);
            //rmse = Math.sqrt(rmse/(trainFeatures.rows()));
            //System.out.println("rmse=" + rmse);
            // rmse=6.315196272725976
            
            
            
            
            
            // Everything below this section was used for testing
//------------------------------------------------------------------------------
            
            //int[] testLayers = {2, 3, 2};
            //int[] testLayers = {2,2,2,2,2};
            //NeuralNet net = new NeuralNet(testLayers);
            
//            double[] x = {0.3, -0.2};
//            double[] y = {0.1, 0.0};
            
//            Matrix xtest = new Matrix(1, 2);
//            xtest.row(0)[0] = 0.3;
//            xtest.row(0)[1] = -0.2;
//            Matrix ytest = new Matrix(1, 2);
//            ytest.row(0)[0] = 0.1;
//            ytest.row(0)[1] = 0.0;
            
//            Matrix xtest = new Matrix(1, 2);
//            xtest.row(0)[0] = 0.1;
//            xtest.row(0)[1] = 0.1;
//            Matrix ytest = new Matrix(1, 2);
//            ytest.row(0)[0] = 0.1;
//            ytest.row(0)[1] = 0.1;
//            
//            net.train(xtest, ytest);
            
//            double[] dummy = new double[2];
//            net.predict(x, dummy);
//            System.out.println("Activation: ");
//            net.printLayerWB();
            /*
            net.layers.get(net.layers.size()-1).feedBackward(y);
            for(int i = net.layers.size()-1; i > 0; i--) {
                
                net.layers.get(i-1).backProp(net.layers.get(i));
            }
            
            double learning_rate = 0.1;
            
            for(int lay = 0; lay < net.layers.size(); lay++) {
                for(int i = 0; i < net.layers.get(lay).weights.rows(); i++) {
                    for(int j = 0; j < net.layers.get(lay).weights.cols(); j++) {
                        if(lay ==0) {
                            net.layers.get(lay).weights.row(i)[j] += learning_rate * net.layers.get(lay).blame[i] * x[j];
                        }
                        else {
                        net.layers.get(lay).weights.row(i)[j] += learning_rate * net.layers.get(lay).blame[i] * net.layers.get(lay-1).activation[j];
                        }
                    }
                    net.layers.get(lay).bias[i] = net.layers.get(lay).bias[i] + learning_rate * net.layers.get(lay).blame[i];
                }
            }
            //net.printLayerWB();
            net.predict(x, dummy);
            System.out.println("Activation: ");
            net.printActivation();
            */
        }
        
        public static void unittestsProject7() {
            
//            java.util.Date date= new java.util.Date();      // Print times for testing
//            System.out.println(new Timestamp(date.getTime()));
            
            // Number of layers suggested by Dr. Gashler
            int[] numberoflayers = {784,80,30,10};           // Vowel sounds
            NeuralNet net = new NeuralNet(numberoflayers);
            
            // Nominal to categorical data
            Filter fm = new Filter(net, new NomCat(), true);
            Filter fn = new Filter(fm, new NomCat(), false);
            
            // Reduces values for easier computation
            Filter fx = new Filter(fn, new Normalizer(), true);
            Filter fy = new Filter(fx, new Normalizer(), false);
            
            // Load train data
            Matrix vtrain_features = new Matrix();
            vtrain_features.loadARFF("train_feat.arff"); // 528 rows, 12 cols
            Matrix vtrain_labels = new Matrix();
            vtrain_labels.loadARFF("train_lab.arff"); // 528 rows, 1 col
            
            // Load test data
            Matrix vtest_features = new Matrix();
            vtest_features.loadARFF("test_feat.arff"); //462 rows, 12 cols
            Matrix vtest_labels = new Matrix();
            vtest_labels.loadARFF("test_lab.arff"); // 462 rows, 1 col
            
            
            
            for(int q = 1; q <= 10; q++) {
                fy.train(vtrain_features, vtrain_labels);
                System.out.println("mis(" + q + "/10)=" + fy.countMisclassifications(vtest_features, vtest_labels));
//                date= new java.util.Date();
//                System.out.println(new Timestamp(date.getTime()));
            }
        }
        
        public static void unittestsProject9() {
            
            int[] numberOfLayers = {4, 12, 12, 3};
            NeuralNet net = new NeuralNet(numberOfLayers);
            
//            net.V = new Matrix();
//            net.V.loadARFF("savedV.arff");
//            String weightname;
//            for(int i = 0; i < net.layers.size(); i++) {
//                weightname = "layer" + i + "weights.arff";
//                net.layers.get(i).weights.loadARFF(weightname);
//                weightname = "layer" + i + "bias.arff";
//                net.layers.get(i).bias = new double[0];
//                net.layers.get(i).bias = Vec.loadArray(net.layers.get(i).bias, weightname);
//            }
            
            Matrix X = new Matrix();
            X.loadARFF("observations.arff");
            
            // Divide all numbers by 255 to normalize
            for(int i = 0; i < X.rows(); i++) {
                for(int j = 0; j < X.cols(); j++) {
                    X.row(i)[j] = X.row(i)[j] / 255;
                }
            }
            net.train_with_images(X);
            
            
            
            int[] layersForControls = {6, 6, 2};
            NeuralNet neural = new NeuralNet(layersForControls);
            
//            Matrix savedV = new Matrix();
//            savedV.loadARFF("savedV.arff");
            
            Matrix Y = new Matrix();
            Y.loadARFF("actions.arff");
            
            // Nominal to categorical data
            Filter fm = new Filter(neural, new NomCat(), true);
            Filter fn = new Filter(fm, new NomCat(), false);
            
            // Reduces values for easier computation
            Filter fx = new Filter(fn, new Normalizer(), true);
            Filter fy = new Filter(fx, new Normalizer(), false);
            
            //neural.V = new Matrix(Y.rows(), 2);
            fy.train(net.V, Y);
            
            
            //net.imagePredict();
            Normalizer norm = new Normalizer();
            norm.train(net.V);
            
            //double[] vorigin = {-1.203430477982321,-0.7113841263067792};
            double[] vorigin = {net.V.row(0)[0],net.V.row(0)[1]};
            
            //System.out.println(vorigin[0] + " " + vorigin[1]);
            
            
            double[] origin = {vorigin[0],vorigin[1],1.0,0.0,0.0,0.0};
            net.makeImage(vorigin, "makeImage/Test/origin.png");
            //System.out.println(vorigin[0] + " " + vorigin[1]);
            //neural.predict(origin, vorigin);
            
            
            norm.transform(vorigin, vorigin);
            
            
            double[] test = {vorigin[0], vorigin[1],1.0,0.0,0.0,0.0};
            //double[] test = {0.39887820631184817,0.43927230409593593,1.0,0.0,0.0,0.0};
            double[] out = new double[2];
            neural.predict(test, out);
            norm.untransform(out, out);
            net.makeImage(out, "makeImage/Test/0.png");
//            System.out.println(out[0] + " " + out[1]);
            // "a" five times
            for(int i = 1; i <= 4; i++) {
                norm.transform(out, out);
                test[0] = out[0];
                test[1] = out[1];
                neural.predict(test, out);
                
                norm.untransform(out, out);
//            System.out.println(out[0] + " " + out[1]);
                net.makeImage(out, "makeImage/Test/" + i + ".png");
            }
            
            test[2] = 0.0;
            test[4] = 1.0;
            
            // "c" five times
            for(int i = 5; i <= 9; i++) {
                norm.transform(out, out);
                test[0] = out[0];
                test[1] = out[1];
                neural.predict(test, out);
                norm.untransform(out, out);
                net.makeImage(out, "makeImage/Test/" + i + ".png");
            }
            
        }
        
        public static void unittestsProject11() {
            
            int[] numberOfLayers = {1, 101, 1};
            NeuralNet net = new NeuralNet(numberOfLayers);
            net.project11ActivationChange();
            
            Matrix X = new Matrix(256, 1);
            for(int i = 0; i < X.rows(); i++) {
                X.row(i)[0] = (double)i / 256;
            }
            
            Matrix Y = new Matrix();
            Y.loadARFF("labor_stats.arff");
            
            net.train_time_series(X, Y);
            
            
            // Notes:
            // Regularization methods are found in Layer.java
                // L1 Regularization is named decay_L1weights
                // L2 Regulatization is named decay_L2weights
            // L1/L2 Regularization methods are called in NeuralNet.java
                // They can be found in the train_time_series method
                // L1 and L2 are commented out and can be used by uncommenting them.
            // Output goes to a file called project11b.arff
                // Output x is the month (starting at 0) over 256. Predictions start at x = 1.
                // Output y is the unemployment rate by %.
            // Several runs were generated to get the desired results on the chart.
        }
        
        public static void unittests() {
            int[] numberOfLayers = {63, 32, 1};
            NeuralNet net = new NeuralNet(numberOfLayers);
            
            Matrix features = new Matrix();
            features.loadARFF("traceroute_features.arff");
            Matrix labels = new Matrix();
            labels.loadARFF("traceroute_labels.arff");
            
            
            // Reduces values for easier computation
            //Filter fx = new Filter(net, new Normalizer(), true);
            Filter fy;// = new Filter(net, new Normalizer(), false);
            //fy.train(features, labels);
            
            int repetitions = 1;
            double msse = 0;
            double sse = 0;
            int n = features.cols();
            int trainNum = (int)(0.8 * n);
            int testNum = n - trainNum;
            Random rand = new Random();
            
            int leaveOneOut = 1;
            double lowest = 1000000000.0;
            int position = 0;
            
            int[] avgRank = new int[n];
            Arrays.fill(avgRank, 0);
            
            int[] oneOut = new int[n];
            Arrays.fill(oneOut, n+1);

            // Total repetitions
            for(int k = 0; k < repetitions; k++) {

                // Reset used array list
                Arrays.fill(oneOut, n+1);
                
                // Copy matrix and then randomly swap rows
                Matrix randomFeatures = new Matrix(features.rows(), features.cols());
                randomFeatures.copyBlock(0, 0, features, 0, 0, features.rows(), features.cols());

                Matrix randomLabels = new Matrix(labels.rows(), labels.cols());
                randomLabels.copyBlock(0, 0, labels, 0, 0, labels.rows(), labels.cols());

                for(int j = features.rows(); j >= 2; j--) {
                    int index = rand.nextInt(j);
                    randomFeatures.swapRows(j-1, index);
                    randomLabels.swapRows(j-1, index);
                }


                // Split 80/20 training data set and testing data set
                Matrix trainingFeatures = new Matrix(trainNum,features.cols());
                trainingFeatures.copyBlock(0, 0, randomFeatures, 0, 0, trainNum, n);

                Matrix trainingLabels = new Matrix(trainNum,labels.cols());
                trainingLabels.copyBlock(0, 0, randomLabels, 0, 0, trainNum, labels.cols());

                Matrix testFeatures = new Matrix(testNum,features.cols());
                testFeatures.copyBlock(0, 0, randomFeatures, trainNum, 0, testNum, n);

                Matrix testLabels = new Matrix(testNum, labels.cols());
                testLabels.copyBlock(0, 0, randomLabels, trainNum, 0, testNum, labels.cols());

                int[] finalRank = new int[64];
            
                
                // Definition, LOO: Leave One Out
                if(trainingFeatures.rows() != trainingLabels.rows() || testFeatures.rows() != testLabels.rows()) {
                    throw new IllegalArgumentException("Mismatching number of rows.");
                }
                
                // i is the remaining number of columns left
                for(int i = 0; i < n-1; i++) {
                    
                    int[] numLayers = {n-i-1,(int)((n-i)/2), 1};
                    net.changeLayers(numLayers);
//System.out.println(numLayers[0] + " " + numLayers[1] + " " + numLayers[2]);
                    fy = new Filter(net, new Normalizer(), false);
                    
                    
                    lowest = 1000000000.0;
                    position = 0;
                    
                    // j is the column to be taken out
                    for(int j = 0; j < n-i; j++) {
                        msse = fy.leaveOneOutCrossValidate(i, j, oneOut, trainingFeatures, trainingLabels, testFeatures, testLabels);
                        sse += msse;
                        
                        if(msse < lowest) {
                            lowest = msse;
                            position = j;
                        }
                        
                        net.randomSetValues();
                        fy = new Filter(net, new Normalizer(), false);
                        
                    }
                    
                    position = fy.correctColumn(position, oneOut);
                    
                    oneOut[i] = position;
                    
                    finalRank[i] = oneOut[i];
                }

                // Find the remaining feature and put it in the array
                for(int i = 0; i < finalRank.length; i++) {
                    int check = i;
                    if(!IntStream.of(finalRank).anyMatch(x -> x == check)) {
                        finalRank[63] = i;
                        break;
                    }
                }
                
                //System.out.println("Ranking of which features to remove first, test " + (k+1));
                for(int i = 0; i < finalRank.length; i++) {
                    avgRank[finalRank[i]] += i;
                //    System.out.print(finalRank[i] + " ");
                }
                //System.out.println();
                //System.out.println();
                //double mse = fy.leaveOneOutCrossValidate(2, features, labels);
            }
            
            int[] indexes = new int[avgRank.length];
            for(int i = 0; i < indexes.length; i++) {
                indexes[i] = i;
            }
            
            // I picked something quick to code, there is only 64 bits so this would not take long
            bubblesort(indexes, avgRank);
            
            System.out.println("Ranking of features election averaged over " + repetitions + " times.");
            for(int i = 0; i < avgRank.length; i++) {
                //System.out.println("Bit: " + indexes[i] + ", Average Error: " + (double)((double)avgRank[i]/repetitions) + " ");
                System.out.println(indexes[i] + " " + (double)((double)avgRank[i]/repetitions));
            }
            System.out.println();
            
//            double mse = 0;
//            System.out.println("mse=" + mse);
//            double rmse = Math.sqrt(mse/(features.rows()));
//            System.out.println("rmse=" + rmse);
            
            
            
            
//            double[] test = {0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0}; // hops = 3 [3,3] (in order, randomized)
//            double[] pred = new double[1];
//            fy.predict(test, pred);
//            
//            System.out.println(pred[0]);
//            
//            
//            
//            double[] test2 = {0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0}; // hops = 12 [12-13,~10-11.5]
//            fy.predict(test2, pred);
//            
//            System.out.println(pred[0]);
//            
//            
//            
//            double[] test3 = {0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0}; // hops = 16 [~11.8-13.8,~11-13]
//            fy.predict(test3, pred);
//            
//            System.out.println(pred[0]);
        }
        
        public static void bubblesort(int[] index, int[] score) {
            int temp;
            
            for(int i = 0; i < score.length - 1; i++) {
                for(int j = 1; j < score.length - i; j++) {
                    if(score[j-1] > score[j]) {
                        temp = score[j-1];
                        score[j-1] = score[j];
                        score[j] = temp;
                        temp = index[j-1];
                        index[j-1] = index[j];
                        index[j] = temp;
                    }
                }
            }
            
        }
        
        
        public static void packet3OSClassifier() {
            
            int numberOfTests = 100;
            double[] averageTest = new double[numberOfTests];
            double lowest = 100;
            double highest = 0;
            
            for(int i = 0; i < averageTest.length; i++) {
                
                // --------------------Load data--------------------------------
                Matrix features = new Matrix();
                features.loadARFF("packet_features2.arff");
                Matrix labels = new Matrix();
                labels.loadARFF("packet_labels2.arff");

                Matrix trainfeat;
                Matrix trainlab;
                Matrix validatefeat;
                Matrix validatelab;

                Random rand = new Random();
                for(int j = features.rows(); j >= 2; j--) {
                    int index = rand.nextInt(j);
                    features.swapRows(j-1, index);
                    labels.swapRows(j-1, index);
                }

                double testPortion = 0.2;
                int test = (int)(features.rows() * testPortion);
                int train = features.rows() - test;

                trainfeat = new Matrix(train, features.cols());
                trainlab = new Matrix(train, labels.cols());
                validatefeat = new Matrix(test, features.cols());
                validatelab = new Matrix(test, labels.cols());

                //Random rand = new Random();
                int trainSet = 0;   // train set
                int testSet = 0;   // test set
                for(int m = 0; m < features.rows(); m++) {
                    if(rand.nextInt(test + train) < train) {
                        trainfeat.copyBlock(trainSet, 0, features, m, 0, 1, features.cols());
                        trainlab.copyBlock(trainSet, 0, labels, m, 0, 1, labels.cols());
                        train--;
                        trainSet++;
                    }
                    else {
                        validatefeat.copyBlock(testSet, 0, features, m, 0, 1, features.cols());
                        validatelab.copyBlock(testSet, 0, labels, m, 0, 1, labels.cols());
                        test--;
                        testSet++;
                    }
                }
                // !-------------------Load data--------------------------------
                
                // ----------------------Neural Network 1----------------------
                int[] numberOfLayers = {7, 1};
                NeuralNet net = new NeuralNet(numberOfLayers);

                net.projectCISRActivationChange();

                Filter fm = new Filter(net, new NomCat(), true);
                Filter fn = new Filter(fm, new NomCat(), false);
                Filter fy = new Filter(fn, new Normalizer(), true);
                //Filter fy = new Filter(fx, new Normalizer(), false);
                fy.train(trainfeat, trainlab);

                // !----------------------Neural Network 1----------------------
                
                // ----------------------Neural Network 2----------------------
                int[] numberOfLayers2 = {7, 4, 1};
                NeuralNet net2 = new NeuralNet(numberOfLayers2);

                net2.projectCISRActivationChange();

                Filter fm2 = new Filter(net2, new NomCat(), true);
                Filter fn2 = new Filter(fm2, new NomCat(), false);
                Filter fy2 = new Filter(fn2, new Normalizer(), true);
                //Filter fy2 = new Filter(fx2, new Normalizer(), false);
                fy2.train(trainfeat, trainlab);

                // !----------------------Neural Network 2----------------------
                
                // ----------------------Neural Network 3----------------------
                int[] numberOfLayers3 = {7, 5, 3, 1};
                NeuralNet net3 = new NeuralNet(numberOfLayers3);

                net3.projectCISRActivationChange();

                Filter fm3 = new Filter(net3, new NomCat(), true);
                Filter fn3 = new Filter(fm3, new NomCat(), false);
                Filter fy3 = new Filter(fn3, new Normalizer(), true);
                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
                fy3.train(trainfeat, trainlab);

                // !----------------------Neural Network 3----------------------
                
                // ----------------------Neural Network 4----------------------
                int[] numberOfLayers4 = {7, 5, 1};
                NeuralNet net4 = new NeuralNet(numberOfLayers4);

                net4.projectCISRActivationChange();

                Filter fm4 = new Filter(net4, new NomCat(), true);
                Filter fn4 = new Filter(fm4, new NomCat(), false);
                Filter fy4 = new Filter(fn4, new Normalizer(), true);
                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
                fy4.train(trainfeat, trainlab);

                // !----------------------Neural Network 4----------------------
                
                // ----------------------Neural Network 5----------------------
//                int[] numberOfLayers5 = {7, 2, 1};
//                NeuralNet net5 = new NeuralNet(numberOfLayers5);
//
//                net5.projectCISRActivationChange();
//
//                Filter fm5 = new Filter(net5, new NomCat(), true);
//                Filter fn5 = new Filter(fm5, new NomCat(), false);
//                Filter fy5 = new Filter(fn5, new Normalizer(), true);
//                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
//                fy5.train(trainfeat, trainlab);

                // !----------------------Neural Network 5----------------------
                
                // ---------------------Predict OS from test data---------------
                int mis = 0;
                double[] pred = new double[1];
                double[] pred2 = new double[1];
                double[] pred3 = new double[1];
                double[] pred4 = new double[1];
//                double[] pred5 = new double[1];
                for(int k = 0; k < validatefeat.rows(); k++)
                {
                    double[] feat = validatefeat.row(k);
                    fy.predictPacket(feat, pred);
                    fy2.predictPacket(feat, pred2);
                    fy3.predictPacket(feat, pred3);
                    fy4.predictPacket(feat, pred4);
//                    fy5.predictPacket(feat, pred5);
                    double[] lab = validatelab.row(k);
                    int[] counter = new int[4];
                    for(int j = 0; j < lab.length; j++)
                    {
                        counter[(int)pred[0]] += 1;
                        counter[(int)pred2[0]] += 1;
                        counter[(int)pred3[0]] += 1;
                        counter[(int)pred4[0]] += 1;
                        
                        int largest = counter[0];
                        int vote = 0;
                        
                        for(int z = 0; z < counter.length; z++)
                        {
                            if(counter[z] > largest) {
                                largest = counter[z]; 
                                vote =z;
                            }  
                        }
                        
                        if(vote == 0) {
                            vote = 0;
                        }
                        else {
                            vote = 1;
                        }
                
                        if(vote != lab[j]) {
                            mis++;
                            
                System.out.println("pred=" + (int)pred[0] + " " + (int)pred2[0] + " " + (int)pred3[0] + " " + (int)pred4[0]);
                System.out.println("vote=" + vote);
                System.out.println("lab=" + lab[j]);
                        }
                    }
                }
                // !--------------------Predict OS from test data---------------
                
                
                //System.out.println("mis=" + mis + "/" + validatefeat.rows());
                
                // -------------------Save weights------------------------------
//                String weightname;
//                for(int m = 0; m < net.layers.size(); m++) {
//                    weightname = "packet/net1-" + i + "layer" + m + "weights.arff";
//                    net.layers.get(m).weights.saveARFF(weightname);
//                    weightname = "packet/net1-" + i + "layer" + m + "bias.arff";
//                    Vec.saveArray(net.layers.get(m).bias, weightname);
//                }
//                
//                for(int m = 0; m < net2.layers.size(); m++) {
//                    weightname = "packet/net2-" + i + "layer" + m + "weights.arff";
//                    net2.layers.get(m).weights.saveARFF(weightname);
//                    weightname = "packet/net2-" + i + "layer" + m + "bias.arff";
//                    Vec.saveArray(net2.layers.get(m).bias, weightname);
//                }
//                
//                for(int m = 0; m < net3.layers.size(); m++) {
//                    weightname = "packet/net3-" + i + "layer" + m + "weights.arff";
//                    net3.layers.get(m).weights.saveARFF(weightname);
//                    weightname = "packet/net3-" + i + "layer" + m + "bias.arff";
//                    Vec.saveArray(net3.layers.get(m).bias, weightname);
//                }
                // !-------------------Save weights------------------------------
                
                
                averageTest[i] = mis/(double)validatefeat.rows();
                //averageTest[i] = (double)fy.countPacketMisclassifications(validatefeat, validatelab)/(double)validatefeat.rows();
                
//                System.out.println("mis1=" + 
//                    (double)fy.countMisclassifications(validatefeat, validatelab) + " / " + (double)validatefeat.rows());
//                System.out.println("Accuracy=" + 
//                        (1 - (double)fy.countMisclassifications(validatefeat, validatelab)/(double)validatefeat.rows()) + "%");
            }
            
            
            
            // Statistics start here -------------------------------------------
            double sum = 0;
            double sqDiff = 0;
            double median = 0;
            double stDev = 0;
            
            int weightRow = 0;
            int bestRow = 0;
            
            for (int i = 0; i < averageTest.length; i++) {
                sum += averageTest[i];
                if(averageTest[i] < lowest) {
                    lowest = averageTest[i];
                    bestRow = i;
                }
                if(averageTest[i] > highest) {
                    highest = averageTest[i];
                    weightRow = i;
                }
            }
            
            Arrays.sort(averageTest);
            
            double mean = (sum / averageTest.length);
            for (double d : averageTest) {
                sqDiff += Math.pow((mean - d), 2);
            }
            
            if(averageTest.length % 2 == 0) {
                median = (averageTest[averageTest.length/2] + averageTest[(averageTest.length/2)-1]) / 2;
            }
            else {
                median = averageTest[averageTest.length/2];
            }
            
            stDev = Math.sqrt(sqDiff / averageTest.length);
            System.out.println("Average over " + averageTest.length + " tests: " + ((1 - mean) * 100) + "%");
            System.out.println("Highest: " + ((1 - lowest) * 100) + "%");
            System.out.println("Lowest: " + ((1 - highest) * 100) + "%");
            System.out.println("Median: " + ((1 - median) * 100) + "%");
            System.out.println("Stdev: " + (stDev * 100) + "%");
            
            System.out.println("Highest weight row: " + weightRow);
            System.out.println("Best weight row: " + bestRow);
            
            
            // Statistics end here ---------------------------------------------
        }
        
        public static void packetClassifier() {
            
            int numberOfTests = 100;
            double[] averageTest = new double[numberOfTests];
            double lowest = 100;
            double highest = 0;
            
            for(int i = 0; i < averageTest.length; i++) {
                
                // --------------------Load data--------------------------------
                Matrix features = new Matrix();
                features.loadARFF("packet_features.arff");
                Matrix labels = new Matrix();
                labels.loadARFF("packet_labels.arff");

                Matrix trainfeat;
                Matrix trainlab;
                Matrix validatefeat;
                Matrix validatelab;

                Random rand = new Random();
                for(int j = features.rows(); j >= 2; j--) {
                    int index = rand.nextInt(j);
                    features.swapRows(j-1, index);
                    labels.swapRows(j-1, index);
                }

                double testPortion = 0.2;
                int test = (int)(features.rows() * testPortion);
                int train = features.rows() - test;

                trainfeat = new Matrix(train, features.cols());
                trainlab = new Matrix(train, labels.cols());
                validatefeat = new Matrix(test, features.cols());
                validatelab = new Matrix(test, labels.cols());

                //Random rand = new Random();
                int trainSet = 0;   // train set
                int testSet = 0;   // test set
                for(int m = 0; m < features.rows(); m++) {
                    if(rand.nextInt(test + train) < train) {
                        trainfeat.copyBlock(trainSet, 0, features, m, 0, 1, features.cols());
                        trainlab.copyBlock(trainSet, 0, labels, m, 0, 1, labels.cols());
                        train--;
                        trainSet++;
                    }
                    else {
                        validatefeat.copyBlock(testSet, 0, features, m, 0, 1, features.cols());
                        validatelab.copyBlock(testSet, 0, labels, m, 0, 1, labels.cols());
                        test--;
                        testSet++;
                    }
                }
                // !-------------------Load data--------------------------------
                
                // ----------------------Neural Network 1----------------------
                int[] numberOfLayers = {7, 1};
                NeuralNet net = new NeuralNet(numberOfLayers);

                net.projectCISRActivationChange();

                Filter fm = new Filter(net, new NomCat(), true);
                Filter fn = new Filter(fm, new NomCat(), false);
                Filter fy = new Filter(fn, new Normalizer(), true);
                //Filter fy = new Filter(fx, new Normalizer(), false);
                fy.train(trainfeat, trainlab);

                // !----------------------Neural Network 1----------------------
                
                // ----------------------Neural Network 2----------------------
                int[] numberOfLayers2 = {7, 4, 1};
                NeuralNet net2 = new NeuralNet(numberOfLayers2);

                net2.projectCISRActivationChange();

                Filter fm2 = new Filter(net2, new NomCat(), true);
                Filter fn2 = new Filter(fm2, new NomCat(), false);
                Filter fy2 = new Filter(fn2, new Normalizer(), true);
                //Filter fy2 = new Filter(fx2, new Normalizer(), false);
                fy2.train(trainfeat, trainlab);

                // !----------------------Neural Network 2----------------------
                
                // ----------------------Neural Network 3----------------------
                int[] numberOfLayers3 = {7, 5, 1};
                NeuralNet net3 = new NeuralNet(numberOfLayers3);

                net3.projectCISRActivationChange();

                Filter fm3 = new Filter(net3, new NomCat(), true);
                Filter fn3 = new Filter(fm3, new NomCat(), false);
                Filter fy3 = new Filter(fn3, new Normalizer(), true);
                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
                fy3.train(trainfeat, trainlab);

                // !----------------------Neural Network 3----------------------
                
                // ----------------------Neural Network 4----------------------
//                int[] numberOfLayers4 = {7, 1};
//                NeuralNet net4 = new NeuralNet(numberOfLayers4);
//
//                net4.projectCISRActivationChange();
//
//                Filter fm4 = new Filter(net4, new NomCat(), true);
//                Filter fn4 = new Filter(fm4, new NomCat(), false);
//                Filter fy4 = new Filter(fn4, new Normalizer(), true);
//                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
//                fy4.train(trainfeat, trainlab);

                // !----------------------Neural Network 4----------------------
                
                // ----------------------Neural Network 5----------------------
//                int[] numberOfLayers5 = {7, 2, 1};
//                NeuralNet net5 = new NeuralNet(numberOfLayers5);
//
//                net5.projectCISRActivationChange();
//
//                Filter fm5 = new Filter(net5, new NomCat(), true);
//                Filter fn5 = new Filter(fm5, new NomCat(), false);
//                Filter fy5 = new Filter(fn5, new Normalizer(), true);
//                //Filter fy3 = new Filter(fx3, new Normalizer(), false);
//                fy5.train(trainfeat, trainlab);

                // !----------------------Neural Network 5----------------------
                
                // ---------------------Predict OS from test data---------------
                int mis = 0;
                double[] pred = new double[1];
                double[] pred2 = new double[1];
                double[] pred3 = new double[1];
                double[] pred4 = new double[1];
                double[] pred5 = new double[1];
                for(int k = 0; k < validatefeat.rows(); k++)
                {
                    double[] feat = validatefeat.row(k);
                    fy.predictPacket(feat, pred);
                    fy2.predictPacket(feat, pred2);
                    fy3.predictPacket(feat, pred3);
//                    fy4.predictPacket(feat, pred4);
//                    fy5.predictPacket(feat, pred5);
                    double[] lab = validatelab.row(k);
                    for(int j = 0; j < lab.length; j++)
                    {
                        int vote = (int)pred[0] + (int)pred2[0] + (int)pred3[0] + (int)pred4[0] + (int)pred5[0];
                        
                        if(vote == 0) {
                            vote = 0;
                        }
                        else {
                            vote = 1;
                        }
                
                        if(vote != lab[j]) {
                            mis++;
                            
//                System.out.println("pred=" + (int)pred[0] + " " + (int)pred2[0] + " " + (int)pred3[0]);
//                System.out.println("vote=" + vote);
//                System.out.println("lab=" + lab[j]);
                        }
                    }
                }
                // !--------------------Predict OS from test data---------------
                
                
                //System.out.println("mis=" + mis + "/" + validatefeat.rows());
                
                // -------------------Save weights------------------------------
                String weightname;
                for(int m = 0; m < net.layers.size(); m++) {
                    weightname = "packet/net1-" + i + "layer" + m + "weights.arff";
                    net.layers.get(m).weights.saveARFF(weightname);
                    weightname = "packet/net1-" + i + "layer" + m + "bias.arff";
                    Vec.saveArray(net.layers.get(m).bias, weightname);
                }
                
                for(int m = 0; m < net2.layers.size(); m++) {
                    weightname = "packet/net2-" + i + "layer" + m + "weights.arff";
                    net2.layers.get(m).weights.saveARFF(weightname);
                    weightname = "packet/net2-" + i + "layer" + m + "bias.arff";
                    Vec.saveArray(net2.layers.get(m).bias, weightname);
                }
                
                for(int m = 0; m < net3.layers.size(); m++) {
                    weightname = "packet/net3-" + i + "layer" + m + "weights.arff";
                    net3.layers.get(m).weights.saveARFF(weightname);
                    weightname = "packet/net3-" + i + "layer" + m + "bias.arff";
                    Vec.saveArray(net3.layers.get(m).bias, weightname);
                }
                // !-------------------Save weights------------------------------
                
                
                averageTest[i] = mis/(double)validatefeat.rows();
                //averageTest[i] = (double)fy.countPacketMisclassifications(validatefeat, validatelab)/(double)validatefeat.rows();
                
//                System.out.println("mis1=" + 
//                    (double)fy.countMisclassifications(validatefeat, validatelab) + " / " + (double)validatefeat.rows());
//                System.out.println("Accuracy=" + 
//                        (1 - (double)fy.countMisclassifications(validatefeat, validatelab)/(double)validatefeat.rows()) + "%");
            }
            
            
            
            // Statistics start here -------------------------------------------
            double sum = 0;
            double sqDiff = 0;
            double median = 0;
            double stDev = 0;
            
            int weightRow = 0;
            int bestRow = 0;
            
            for (int i = 0; i < averageTest.length; i++) {
                sum += averageTest[i];
                if(averageTest[i] < lowest) {
                    lowest = averageTest[i];
                    bestRow = i;
                }
                if(averageTest[i] > highest) {
                    highest = averageTest[i];
                    weightRow = i;
                }
            }
            
            Arrays.sort(averageTest);
            
            double mean = (sum / averageTest.length);
            for (double d : averageTest) {
                sqDiff += Math.pow((mean - d), 2);
            }
            
            if(averageTest.length % 2 == 0) {
                median = (averageTest[averageTest.length/2] + averageTest[(averageTest.length/2)-1]) / 2;
            }
            else {
                median = averageTest[averageTest.length/2];
            }
            
            stDev = Math.sqrt(sqDiff / averageTest.length);
            System.out.println("Average over " + averageTest.length + " tests: " + ((1 - mean) * 100) + "%");
            System.out.println("Highest: " + ((1 - lowest) * 100) + "%");
            System.out.println("Lowest: " + ((1 - highest) * 100) + "%");
            System.out.println("Median: " + ((1 - median) * 100) + "%");
            System.out.println("Stdev: " + (stDev * 100) + "%");
            
            System.out.println("Highest weight row: " + weightRow);
            System.out.println("Best weight row: " + bestRow);
            
            
            // Statistics end here ---------------------------------------------
        }
        
        public static void packetOSPredict() {
            int[] numberOfLayers1 = {7, 1};
            NeuralNet net = new NeuralNet(numberOfLayers1);

            int best = 0;
            double temp = 0;
            for(int row = 0; row < 100; row++) {
            // ------------------Load all weights-------------------------------
            int bestrow = row;
            String weightname;
            for(int i = 0; i < net.layers.size(); i++) {
                weightname = "packet/net1-" + bestrow + "layer" + i + "weights.arff";
                net.layers.get(i).weights.loadARFF(weightname);
                weightname = "packet/net1-" + bestrow + "layer" + i + "bias.arff";
                net.layers.get(i).bias = new double[0];
                net.layers.get(i).bias = Vec.loadArray(net.layers.get(i).bias, weightname);
            }
            
            int[] numberOfLayers2 = {7, 4, 1};
            NeuralNet net2 = new NeuralNet(numberOfLayers2);

            for(int i = 0; i < net2.layers.size(); i++) {
                weightname = "packet/net2-" + bestrow + "layer" + i + "weights.arff";
                net2.layers.get(i).weights.loadARFF(weightname);
                weightname = "packet/net2-" + bestrow + "layer" + i + "bias.arff";
                net2.layers.get(i).bias = new double[0];
                net2.layers.get(i).bias = Vec.loadArray(net2.layers.get(i).bias, weightname);
            }
            
            int[] numberOfLayers3 = {7, 5, 3, 1};
            NeuralNet net3 = new NeuralNet(numberOfLayers3);

            for(int i = 0; i < net3.layers.size(); i++) {
                weightname = "packet/net3-" + bestrow + "layer" + i + "weights.arff";
                net3.layers.get(i).weights.loadARFF(weightname);
                weightname = "packet/net3-" + bestrow + "layer" + i + "bias.arff";
                net3.layers.get(i).bias = new double[0];
                net3.layers.get(i).bias = Vec.loadArray(net3.layers.get(i).bias, weightname);
            }
            
            // !-----------------Load all weights-------------------------------
            
            
            // ------------------Setup filters----------------------------------
            Matrix features = new Matrix();
            features.loadARFF("packet_features.arff");
            Matrix labels = new Matrix();
            labels.loadARFF("packet_labels.arff");
            
            Filter fm = new Filter(net, new NomCat(features), true);
            Filter fn = new Filter(fm, new NomCat(labels), false);
            Filter fy = new Filter(fn, new Normalizer(features), true);

            Filter fm2 = new Filter(net2, new NomCat(features), true);
            Filter fn2 = new Filter(fm2, new NomCat(labels), false);
            Filter fy2 = new Filter(fn2, new Normalizer(features), true);

            Filter fm3 = new Filter(net3, new NomCat(features), true);
            Filter fn3 = new Filter(fm3, new NomCat(labels), false);
            Filter fy3 = new Filter(fn3, new Normalizer(features), true);

            // !-----------------Setup filters----------------------------------
            
            // ------------------Predict Packets--------------------------------
            int mis = 0;
            double[] pred = new double[1];
            double[] pred2 = new double[1];
            double[] pred3 = new double[1];
            for(int k = 0; k < features.rows(); k++)
            {
                double[] feat = features.row(k);
                fy.predictPacket(feat, pred);
                fy2.predictPacket(feat, pred2);
                fy3.predictPacket(feat, pred3);
                double[] lab = labels.row(k);
                for(int j = 0; j < lab.length; j++)
                {
                    int vote = (int)pred[0] + (int)pred2[0] + (int)pred3[0];

                    if(vote == 0) {
                        vote = 0;
                    }
                    else {
                        vote = 1;
                    }

                    if(vote != lab[j]) {
                        mis++;

//                System.out.println("pred=" + (int)pred[0] + " " + (int)pred2[0] + " " + (int)pred3[0]);
//                System.out.println("vote=" + vote);
//                System.out.println("lab=" + lab[j]);
                    }
                }
            }
            // !-----------------Predict Packets--------------------------------

            //System.out.println("Percentage of correct predictions: " + (1 - (double)mis/features.rows()) * 100 + "%");
            if(((1 - (double)mis/features.rows()) * 100) > temp) {
                best = row;
                temp = (1 - (double)mis/features.rows()) * 100;
            }
            }
            System.out.println("BEST ROW: " + best + " @ " + temp);
        }
        
	public static void main(String[] args)
	{
            // Do Final Project
            //unittests();
            
            // Run packet-analyzer neural network
            packetClassifier();
            //packet3OSClassifier();
            //packetOSPredict();
	}
}
