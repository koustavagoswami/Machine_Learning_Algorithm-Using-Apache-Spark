import scala.Tuple2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.random.StandardNormalGenerator;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;



public class TrainingModel {

	public static void main(String args[])
	{
		/*In the below part we have written a small piece of code which is reading the file which is provided for the movie review
		 * with the label 1 or 0. 1 is positive and 0 is negative. Depending on that label we are dividing the data into two files
		 * one is consisting with all the positive review data and negative data review data in one file removing the label part
		 * I have used to part to read the file in case of positive and negative separately because there can be multiple file
		 * Here it is one file
		 */
		
		/*Start the New file creation code here******************************************************************************************/
		String line = "";
		
		//Negative data file creation part
		try {
			//This part is to read the given file
			FileInputStream fs = new FileInputStream("/Users/koustavagoswami/Downloads/imdb_labelled.txt"); //Please change the file path in case of you where the file is kept
			BufferedReader br = new BufferedReader(new InputStreamReader(fs));
			int count = 0;
			//This will write the file in the same directory of the project and java file 
			FileOutputStream fos = new FileOutputStream("./imdb_labelled_negative" + count + ".txt");
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));             
			while ((line = br.readLine()) != null) {
				String mine = line;              
				if (mine.endsWith("0"))
				{
					//System.out.print("Yes");
					bw.write(mine.substring(0,(mine.length()-1)));
					bw.newLine();
					bw.flush();
					count++;

				}


			}

		} catch (Exception e) {
			System.out.println("Exception: " + e);
		}

		//Postive file creation part
		try {
			FileInputStream fs = new FileInputStream("/Users/koustavagoswami/Downloads/imdb_labelled.txt");////Please change the file path in case of you where the file is kept
			BufferedReader br = new BufferedReader(new InputStreamReader(fs));

			int count = 0;
			//This will write the file in the same directory of the project and java file 
			FileOutputStream fos = new FileOutputStream("./imdb_labelled_positive" + count + ".txt");
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));             
			while ((line = br.readLine()) != null) {
				String mine = line;              
				if (mine.endsWith("1"))
				{
					//System.out.print("Yes");
					bw.write(mine.substring(0,(mine.length()-1)));
					bw.newLine();
					bw.flush();
					count++;

				}


			}

		} catch (Exception e) {
			System.out.println("Exception: " + e);
		}
		/*File Creation code ends her*******************************************************************************************************/

		//I have used Logger to switch off the info log of the spark 
		Logger.getLogger("org.apache").setLevel(Level.OFF);	
		//This part is the initialization of the spark context . I have given the app name as "CountTemperature" and set the master to 
		//Local as it os running locally.
		SparkConf sparkConf = new SparkConf().setAppName("TrainingModel").setMaster("local[4]").set("spark.executor.memory",
				"1g");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		//We are making two JavaRdd of string reading the postive and negative files
		JavaRDD<String> labelledData = sc.textFile("./imdb_labelled_positive0.txt");
		JavaRDD<String> labelledData2 = sc.textFile("./imdb_labelled_negative0.txt");
		
		/*We have used Hashing transform as tf will later be used to create numerical feature vectors from string-form
		features, by generating indices in form of hash values for terms (e.g.,
		words or other tokens) and computing the term frequencies based on
		these indices.*/
		final HashingTF tf = new HashingTF(10000);
		//We are creating two JavaRDD LabelPoint of positive and negative with the help of positive RDD and negative RDD
		//Making the label 1 and 0 respectively 
		JavaRDD<LabeledPoint> positiveExamples = labelledData.map(a -> new LabeledPoint(1, tf.transform(Arrays.asList(a.trim().split(" ")))));
		JavaRDD<LabeledPoint> negativeExamples = labelledData2.map(a -> new LabeledPoint(0, tf.transform(Arrays.asList(a.trim().split(" ")))));

		//This part we are creating the data set JavaRDD with the help positive and negative dataset made above
		//by union the data
		JavaRDD<LabeledPoint> dataSet = positiveExamples.union(negativeExamples);
		
		//This part is to make the training data with the help of sample where 60% will be training data
		JavaRDD<LabeledPoint> training = dataSet.sample(false, 0.6, 11L);
		
		dataSet.cache(); //We are taking the help of cache as we are storing the memory data as will be iterating the training model multiple times
		//Making of test data subtracting the training data from whole data set
		JavaRDD<LabeledPoint> test = dataSet.subtract(training);

		//This is the part where we are making SVM model and giving the training data to predict the label by iterating 1000 times
		SVMModel model = SVMWithSGD.train(training.rdd(), 1000);
		
		//Making the scoreand Labels with the help of test data mapping the first label by the model which has been trained earlier to
		// predict the label and second tuple is the label is test data label
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
		new Tuple2<>(model.predict(p.features()), p.label()));
		
		//This part is commented as it is said to take some few data to print the prediction of test data
		//scoreAndLabels.collect().forEach(a -> System.out.println("The predicted level is :" + a._1() + " for " + a._2()));
		
		//This part is to take 10 data from scoreand Label RDD and making a list
		List<Tuple2<Object, Object>> testPart = scoreAndLabels.collect().subList(0, 10);
		
		//This part is printing the model prediction data on the movie review label and original test data label
		testPart.forEach(a -> System.out.println("The predicted level is :" + a._1 +  " for "  + a._2));
		
		/*This is the part where we are makeing a matrics with scoresand Label data so that we can get the ROC area
		 * This is making a matrics and making the ROC area and printing the area under the ROC for the model which
		 * we have made with the data and review label
		 */
		BinaryClassificationMetrics metrics =
				new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
		double auROC = metrics.areaUnderROC();

		System.out.println("Area under ROC = " + auROC);
		/*This Parts ends here*/

		//Closing and ending the spark session here
		sc.close();
		sc.stop();

	}
}
