/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature.selection

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NominalAttribute}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, _}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{rand, udf}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Abstraction for FeatureSelectors, which selects features to use for predicting a categorical label.
  * The selector supports two selection methods: `numTopFeatures` and `percentile`.
  *  - `numTopFeatures` chooses a fixed number of top features according to the feature importance.
  *  - `percentile` is similar but chooses a fraction of all features instead of a fixed number.
  * By default, the selection method is `numTopFeatures`, with the default number of top features set to 50.
  *
  * @tparam Learner  Specialization of this class.  If you subclass this type, use this type
  *                  parameter to specify the concrete type.
  * @tparam M  Specialization of [[FeatureSelectorModel]]. If you subclass this type, use this type
  *            parameter to specify the concrete type for the corresponding model.
  */
abstract class FeatureSelector[
  Learner <: FeatureSelector[Learner, M],
  M <: FeatureSelectorModel[M]] @Since("2.1.1")
  extends Estimator[M] with FeatureSelectorParams with DefaultParamsWritable {
  /** @group setParam */
  @Since("2.1.1")
  def setNumTopFeatures(value: Int): Learner = set(numTopFeatures, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setPercentile(value: Double): Learner = set(percentile, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setSelectorType(value: String): Learner = set(selectorType, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setFeaturesCol(value: String): Learner = set(featuresCol, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setOutputCol(value: String): Learner = set(outputCol, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setLabelCol(value: String): Learner = set(labelCol, value).asInstanceOf[Learner]

  /** @group setParam */
  @Since("2.1.1")
  def setRandomCutOff(value: Double): Learner = set(randomCutOff, value).asInstanceOf[Learner]

  override def fit(dataset: Dataset[_]): M = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train() and make().
    transformSchema(dataset.schema, logging = true)

    val randomColMaxCategories = 10000

    // Get num features for percentile calculation
    val attrGroup = AttributeGroup.fromStructField(dataset.schema($ {
      featuresCol
    }))
    val numFeatures = attrGroup.size

    val (featureImportances, features) =
      if ($(selectorType) == FeatureSelector.Random) {
        // Append column with random values to dataframe
        val withRandom = dataset.withColumn("random", (rand * randomColMaxCategories).cast(IntegerType))
        val featureVectorWithRandom = new VectorAssembler()
          .setInputCols(Array($(featuresCol), "random"))
          .setOutputCol("FeaturesAndRandom")
          .transform(withRandom)

        // Cache and change features column name, calculate importances and reset.
        val realFeaturesCol = $(featuresCol)
        setFeaturesCol("FeaturesAndRandom")
        val featureImportances = train(featureVectorWithRandom)
        setFeaturesCol(realFeaturesCol)
        val idFromRandomCol = featureImportances.map(_._1).max

        // Take features until reaching random feature. Take overlap from remaining depending on randomCutOff percentage
        val sortedFeatureImportances = featureImportances
          .sortBy { case (_, imp) => -imp } // minus for descending direction!
          .zipWithIndex

        val randomColPos = sortedFeatureImportances.find { case ((fId, fImp), sortId) => fId == idFromRandomCol }.get._2
        val overlap = math.max(0, math.round((featureImportances.length - randomColPos - 1) * $(randomCutOff))).toInt

        (featureImportances.filterNot(_._1 == idFromRandomCol), sortedFeatureImportances
          .take(randomColPos + overlap + 1)
          .map(_._1)
          .filterNot(_._1 == idFromRandomCol))
      } else {
        val featureImportances = train(dataset)

        // Select features depending on selection method
        val features = $(selectorType) match {
          case FeatureSelector.NumTopFeatures => featureImportances
            .sortBy { case (_, imp) => -imp } // minus for descending direction!
            .take($(numTopFeatures))
          case FeatureSelector.Percentile => featureImportances
            .sortBy { case (_, imp) => -imp }
            .take((numFeatures * $(percentile)).toInt) // Take is save, even if numFeatures > featureImportances.length
          case errorType =>
            throw new IllegalStateException(s"Unknown FeatureSelector Type: $errorType")
        }
        (featureImportances, features)
      }

    if (featureImportances.length < numFeatures)
      logWarning(s"Some features were dropped while calculating importance values, " +
        s"since numFeatureImportances < numFeatures (${featureImportances.length} < $numFeatures). This happens " +
        s"e.g. for constant features.")


    // Save name of columns and corresponding importance value
    val nameImportances = featureImportances.map { case (idx, imp) => (
      if (attrGroup.attributes.isDefined && attrGroup.getAttr(idx).name.isDefined)
        attrGroup.getAttr(idx).name.get
      else {
        logWarning(s"The metadata of $featuresCol is empty or does not contain a name for col index: $idx")
        idx.toString
      }
      , imp)
    }

    val indices = features.map { case (idx, _) => idx }

    copyValues(make(uid, indices, nameImportances.toMap).setParent(this))
  }

  override def copy(extra: ParamMap): Learner

  /**
    * Calculate the featureImportances. These shall be sortable in descending direction to select the best features.
    * FeatureSelectors implement this to avoid dealing with schema validation
    * and copying parameters into the model.
    *
    * @param dataset Training dataset
    * @return Array of (feature index, feature importance)
    */
  protected def train(dataset: Dataset[_]): Array[(Int, Double)]

  /**
    * Abstract intantiation of the Model.
    * FeatureSelectors implement this as a constructor for FeatureSelectorModels
    * @param uid of Model
    * @param selectedFeatures list of indices to select
    * @param featureImportances Map that stores each feature importance
    * @return Fitted model
    */
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): M

  @Since("2.1.1")
  def transformSchema(schema: StructType): StructType = {
    val otherPairs = FeatureSelector.supportedSelectorTypes.filter(_ != $(selectorType))
    otherPairs.foreach { paramName: String =>
      if (isSet(getParam(paramName))) {
        logWarning(s"Param $paramName will take no effect when selector type = ${$(selectorType)}.")
      }
    }
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkNumericType(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
  * Abstraction for a model for selecting features.
  * @param uid of Model
  * @param selectedFeatures list of indices to select
  * @param featureImportances Map that stores each feature importance
  * @tparam M  Specialization of [[FeatureSelectorModel]]. If you subclass this type, use this type
  *            parameter to specify the concrete type for the corresponding model.
  */
@Since("2.1.1")
abstract class FeatureSelectorModel[M <: FeatureSelectorModel[M]] private[ml] (@Since("2.1.1") override val uid: String,
                                                                               @Since("2.1.1") val selectedFeatures: Array[Int],
                                                                               @Since("2.1.1") val featureImportances: Map[String, Double])
  extends Model[M] with FeatureSelectorParams with MLWritable {
  /** @group setParam */
  @Since("2.1.1")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  @Since("2.1.1")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @Since("2.1.1")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    // Validity checks
    val inputAttr = AttributeGroup.fromStructField(dataset.schema($(featuresCol)))
    inputAttr.numAttributes.foreach { numFeatures =>
      val maxIndex = selectedFeatures.max
      require(maxIndex < numFeatures,
        s"Selected feature index $maxIndex invalid for only $numFeatures input features.")
    }

    // Prepare output attributes
    val selectedAttrs: Option[Array[Attribute]] = inputAttr.attributes.map { attrs =>
      selectedFeatures.map(index => attrs(index))
    }
    val outputAttr = selectedAttrs match {
      case Some(attrs) => new AttributeGroup($(outputCol), attrs)
      case None => new AttributeGroup($(outputCol), selectedFeatures.length)
    }

    // Select features
    val slicer = udf { vec: Vector =>
      vec match {
        case features: DenseVector => Vectors.dense(selectedFeatures.map(features.apply))
        case features: SparseVector => features.slice(selectedFeatures)
      }
    }
    dataset.withColumn($(outputCol), slicer(dataset($(featuresCol))), outputAttr.toMetadata())
  }

  @Since("2.1.1")
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val newField = prepOutputField(schema)
    val outputFields = schema.fields :+ newField
    StructType(outputFields)
  }

  /**
    * Prepare the output column field, including per-feature metadata.
    */
  private def prepOutputField(schema: StructType): StructField = {
    val selector = selectedFeatures.toSet
    val origAttrGroup = AttributeGroup.fromStructField(schema($(featuresCol)))
    val featureAttributes: Array[Attribute] = if (origAttrGroup.attributes.nonEmpty) {
      origAttrGroup.attributes.get.zipWithIndex.filter(x => selector.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](selector.size)(NominalAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }
}


@Since("2.1.1")
protected [selection] object FeatureSelectorModel {

  class FeatureSelectorModelWriter[M <: FeatureSelectorModel[M]](instance: M) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int], featureImportances: Map[String, Double])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq, instance.featureImportances)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  abstract class FeatureSelectorModelReader[M <: FeatureSelectorModel[M]] extends MLReader[M] {

    protected val className: String

    override def load(path: String): M = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val selectedFeatures = data.select("selectedFeatures").head().getAs[Seq[Int]](0).toArray
      val featureImportances = data.select("featureImportances").head().getAs[Map[String, Double]](0)
      val model = make(metadata.uid, selectedFeatures, featureImportances)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }

    /**
      * Abstract intantiation of the Model.
      * FeatureSelectors implement this as a constructor for FeatureSelectorModels
      * @param uid of Model
      * @param selectedFeatures list of indices to select
      * @param featureImportances Map that stores each feature importance
      * @return Fitted model
      */
    protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): M
  }
}

private[selection] object FeatureSelector {
  /**
    * String name for `numTopFeatures` selector type.
    */
  val NumTopFeatures: String = "numTopFeatures"

  /**
    * String name for `percentile` selector type.
    */
  val Percentile: String = "percentile"

  /**
    * String name for `random` selector type.
    */
  val Random: String = "randomCutOff"

  /** Set of selector types that FeatureSelector supports. */
  val supportedSelectorTypes: Array[String] = Array(NumTopFeatures, Percentile, Random)
}