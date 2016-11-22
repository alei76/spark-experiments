package drew.ml;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

public class JavaClassificationExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaClassificationExample")
                .master("local")
                .getOrCreate();

        // Load the data from the sequence file provided
        JavaRDD<Tuple2<Text, Text>> data = spark.sparkContext().sequenceFile("src/main/data/newsgroups/20news-vectors.seq", Text.class, Text.class).toJavaRDD();

        // Convert to Row(s)
        JavaRDD<Row> rows = data.map(new Function<Tuple2<Text, Text>, Row>() {
            public Row call(Tuple2<Text, Text> a) {
                String path = a._1.toString();
                String[] parts = path.split("/");
                return RowFactory.create(parts[1], parts[2], a._2.toString());
            }
        });

        // Establish Schema/Dataset
        StructType schema = new StructType(new StructField[]{
                new StructField("group", DataTypes.StringType, false, Metadata.empty()),
                new StructField("id", DataTypes.StringType, false, Metadata.empty()),
                new StructField("message", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> newsgroupData = spark.createDataFrame(rows, schema);

        // Tokenize Message into words
        Tokenizer tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(newsgroupData);
        wordsData.select("group", "id", "words").show();

        // filter it down to 2 classes only
        //wordsData = wordsData.filter(
        //        wordsData.col("group").like("rec.motorcycles")
        //                .or(wordsData.col("group").like("rec.autos")));

        // Display some output
        wordsData.select("group", "id", "words").show();

        // Calculate hash based features for words + TF
        int numFeatures = (int) Math.pow(2.0, 18.0);
        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
        // alternatively, CountVectorizer can also be used to get term frequency vectors

        // Calculate IDF
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);
        Dataset<Row> rescaledData = idfModel.transform(featurizedData);

        // Change the group names to numeric labels for input to the classifier
        StringIndexer groupCounter = new StringIndexer()
                .setInputCol("group")
                .setOutputCol("label");
        StringIndexerModel indexModel = groupCounter.fit(rescaledData);
        Dataset<Row> labeledData = indexModel.transform(rescaledData);

        // Display some output
        labeledData.select("label", "id", "features").show();

        // Do a 60/40 train/test split
        Dataset<Row>[] splits = labeledData.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // Create the trainer and train the models
        NaiveBayes nb = new NaiveBayes();
        NaiveBayesModel model = nb.fit(train);

        // Evaluate accuracy on the test set
        Dataset<Row> result = model.transform(test);
        result.show(100, true);
        Dataset<Row> predictionAndLabels = result.select("prediction", "label");
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy");
        System.out.println("Accuracy = " + evaluator.evaluate(predictionAndLabels));

        spark.stop();
    }

}
