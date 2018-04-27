# Feature Selection for Apache Spark
Different Featureselection methods (3 filters/ 2 selectors based on scores from embedded methods) are provided as Spark MLlib `PipelineStage`s. 
These are: 

## Filters:
1) CorrelationSelector: calculates correlation ("spearman", "pearson"- adjustable through ```.setCorrelationType```) between each feature and label. 
2) GiniSelector: measures impurity difference between before and after a feature value is known. 
3) InfoGainSelector: measures the information gain of a feature with respect to the class. 

## Embedded:
1) ImportanceSelector: takes FeatureImportances from any embedded method, e.g. Random Forest.
2) LRSelector: takes feature weights from (L1) logistic regression. The weights are in a matrix W with dimensions #Labels X #Features. The absolute value is taken from all entries, summed column wise and scaled with the max value. 

## Util
1) VectorMerger: takes several VectorColumns (e.g.  the result of  different feature selection methods) and merges them into one VectorColumn. Unlike the VectorAssembler, VectorMerger uses the metadata of the VectorColumn to remove duplicates. It supports two modes:
   - useFeaturesCol true and featuresCol set: the output column will contain the corresponding column from featuresCol (match by name) that have names appearing in one of the inputCols. Use this, if feature importances were calculated using (e.g.) discretized columns, but selection shall use original values. 
   - useFeaturesCol false: the output column will contain the columns from the inputColumns, but dropping duplicates.

Formulas for metrics:

General: 
- <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineP(X_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(X_j)" title="P(X_j)" /></a> - prior probability of feature X having value <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineX_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_j" title="X_j" /></a>
- <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineP(Y_c&space;|&space;X_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_c&space;|&space;X_j)" title="P(Y_c | X_j)" /></a> - cond. probability that a sample is of class <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineY_c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y_c" title="Y_c" /></a>, given that feature X has value <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineX_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_j" title="X_j" />
- <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineP(Y_c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Y_c)" title="P(Y_c)" /></a> - prior probability that the label Y has value <a href="https://www.codecogs.com/eqnedit.php?latex=\inlineY_c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y_c" title="Y_c" /></a>

1) Correlation: Calculated through ``org.apache.spark.mllib.stat``
2) Gini: 

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Gini(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)^2}}-\sum_c{P(Y_c)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Gini(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)^2}}-\sum_c{P(Y_c)^2}" title="Gini(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)^2}}-\sum_c{P(Y_c)^2}" /></a>
3) Informationgain:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;IG(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)\log{P(Y_c|X_j))}}}-\sum_c{P(Y_c)\log{P(Y_c)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;IG(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)\log{P(Y_c|X_j))}}}-\sum_c{P(Y_c)\log{P(Y_c)}}" title="IG(X)=\sum_j{P(X_j)*\sum_c{P(Y_c|X_j)\log{P(Y_c|X_j))}}}-\sum_c{P(Y_c)\log{P(Y_c)}}" /></a>

## Usage

All selection methods share a common API, similar to `ChiSqSelector`. 

```scala
import org.apache.spark.ml.feature.selection.filter._ 
import org.apache.spark.ml.feature.selection.util._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

val data = Seq(
  (Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
  (Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
  (Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
)

val df = spark.createDataset(data).toDF("features", "label")
  
val igSel = new InfoGainSelector()
             .setFeaturesCol("features")
             .setLabelCol("label")
             .setOutputCol("igSelectedFeatures")
             .setSelectorType("percentile")
           
val corSel = new CorrelationSelector()
               .setFeaturesCol("features")
               .setLabelCol("label")
               .setOutputCol("corrSelectedFeatures")
               .setSelectorType("percentile")
           
val giniSel = new GiniSelector()           
                .setFeaturesCol("features")
                .setLabelCol("label")
                .setOutputCol("giniSelectedFeatures")
                .setSelectorType("percentile")

val merger = new VectorMerger()
              .setInputCols(Array("igSelectedFeatures", "corrSelectedFeatures", "giniSelectedFeatures"))
              .setOutputCol("filtered")
              
val plm = new Pipeline().setStages(Array(igSel, corSel, giniSel, merger)).fit(df)

plm.transform(df).select("filtered").show()

```

   
