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

package org.apache.spark.ml.feature.selection.filter

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.selection.{FeatureSelector, FeatureSelectorModel}
import org.apache.spark.ml.linalg.{DenseVector, _}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import scala.collection.mutable

/**
  * Feature selection based on Correlation (absolute value).
  */
@Since("2.1.1")
final class CorrelationSelector @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends FeatureSelector[CorrelationSelector, CorrelationSelectorModel] {

  /**
    * The correlation type of the CorrelationSelector.
    * Supported options: "spearman" (default), "pearson".
    *
    * @group param
    */
  @Since("2.1.1")
  val correlationType = new Param[String](this, "correlationType",
    "The correlation type used for correlation calculation. " +
      "Supported options: " + CorrelationSelector.supportedCorrelationTypes.mkString(", "),
    ParamValidators.inArray[String](CorrelationSelector.supportedCorrelationTypes))
  setDefault(correlationType -> CorrelationSelector.Spearman)

  /** @group getParam */
  @Since("2.1.1")
  def getCorrelationType: String = $(correlationType)

  /** @group setParam */
  @Since("2.1.1")
  def setCorrelationType(value: String): this.type = set(correlationType, value)

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("correlationSelector"))

  @Since("2.1.1")
  override protected def train(dataset: Dataset[_]): Array[(Int, Double)] = {
    val assembleFunc = udf { r: Row =>
      CorrelationSelector.assemble(r.toSeq: _*)
    }

    // Merge featuresCol and labelCol into one OldVector like this: [featuresCol, labelCol]
    val tmpDf = dataset.select(assembleFunc(struct(dataset($(featuresCol)), dataset($(labelCol)))).as("input"))
    val input = tmpDf.select(col("input")).rdd.map { case Row(features: Vector) => OldVectors.fromML(features) }

    // Calculate correlation between all columns in input
    // We're only interested in the last row/ column (correlationmatrix is symmetric) of correlation,
    // which stands for the correlation between all feature columns and the label column
    val correlations = Statistics.corr(input, ${correlationType})

    // Extract the information we're interested in
    val columns = correlations.toArray.grouped(correlations.numRows)
    val corrList = columns.map(column => new DenseVector(column)).toList
    val targetCorr = corrList.last.toArray

    // targetCorr.length - 1, because last element is correlation between label column and itself
    // Also take abs(), because important features are also negatively correlated
    val targetCorrs = Array.tabulate(targetCorr.length - 1) { i => (i, targetCorr(i)) }
      .map(elem => (elem._1, math.abs(elem._2)))

    targetCorrs.filter(!_._2.isNaN)
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): CorrelationSelector = defaultCopy(extra)

  @Since("2.1.1")
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): CorrelationSelectorModel = {
    new CorrelationSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

/**
  * Model fitted by [[CorrelationSelector]].
  * @param selectedFeatures list of indices to select (filter)
  */
@Since("2.1.1")
final class CorrelationSelectorModel private[selection] (@Since("2.1.1") override val uid: String,
                                                         @Since("2.1.1") override val selectedFeatures: Array[Int],
                                                         @Since("2.1.1") override val featureImportances: Map[String, Double])
  extends FeatureSelectorModel[CorrelationSelectorModel](uid, selectedFeatures, featureImportances) {
  /**
    * The correlation type of the CorrelationSelector.
    * Supported options: "spearman" (default), "pearson".
    *
    * @group param
    */
  @Since("2.1.1")
  val correlationType = new Param[String](this, "correlationType",
    "The correlation type used for correlation calculation. " +
      "Supported options: " + CorrelationSelector.supportedCorrelationTypes.mkString(", "),
    ParamValidators.inArray[String](CorrelationSelector.supportedCorrelationTypes))
  setDefault(correlationType -> CorrelationSelector.Spearman)

  /** @group getParam */
  @Since("2.1.1")
  def getCorrelationType: String = $(correlationType)

  @Since("2.1.1")
  override def copy(extra: ParamMap): CorrelationSelectorModel = {
    val copied = new CorrelationSelectorModel(uid, selectedFeatures, featureImportances)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.1.1")
  override def write: MLWriter = new FeatureSelectorModel.FeatureSelectorModelWriter[CorrelationSelectorModel](this)
}

@Since("2.1.1")
object CorrelationSelectorModel extends MLReadable[CorrelationSelectorModel] {
  @Since("2.1.1")
  override def read: MLReader[CorrelationSelectorModel] = new CorrelationSelectorModelReader

  @Since("2.1.1")
  override def load(path: String): CorrelationSelectorModel = super.load(path)
}

@Since("2.1.1")
final class CorrelationSelectorModelReader extends FeatureSelectorModel.FeatureSelectorModelReader[CorrelationSelectorModel]{
  @Since("2.1.1")
  override protected val className: String = classOf[CorrelationSelectorModel].getName

  @Since("2.1.1")
  override protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): CorrelationSelectorModel = {
    new CorrelationSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

private[filter] object CorrelationSelector extends DefaultParamsReadable[CorrelationSelector] {
  @Since("2.1.1")
  override def load(path: String): CorrelationSelector = super.load(path)

  /**
    * String name for `pearson` correlation type.
    */
  val Pearson: String = "pearson"

  /**
    * String name for `spearman` correlation type.
    */
  val Spearman: String = "spearman"

  /** Set of correlation types that CorrelationSelector supports. */
  val supportedCorrelationTypes: Array[String] = Array(Pearson, Spearman)

  // From VectorAssembler
  private def assemble(vv: Any*): Vector = {
    val indices = mutable.ArrayBuilder.make[Int]
    val values = mutable.ArrayBuilder.make[Double]
    var cur = 0
    vv.foreach {
      case v: Double =>
        if (v != 0.0) {
          indices += cur
          values += v
        }
        cur += 1
      case vec: Vector =>
        vec.foreachActive { case (i, v) =>
          if (v != 0.0) {
            indices += cur + i
            values += v
          }
        }
        cur += vec.size
      case null =>
        throw new SparkException("Values to assemble cannot be null.")
      case o =>
        throw new SparkException(s"$o of type ${o.getClass.getName} is not supported.")
    }
    Vectors.sparse(cur, indices.result(), values.result()).compressed
  }
}