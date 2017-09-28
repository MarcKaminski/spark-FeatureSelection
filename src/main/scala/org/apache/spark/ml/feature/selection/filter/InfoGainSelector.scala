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

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.feature.selection.{FeatureSelector, FeatureSelectorModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, _}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

/**
  * Feature selection based on Information Gain.
  */
@Since("2.1.1")
final class InfoGainSelector @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends FeatureSelector[InfoGainSelector, InfoGainSelectorModel] {

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("igSelector"))

  @Since("2.1.1")
  override def train(dataset: Dataset[_]): Array[(Int, Double)] = {
    val input: RDD[LabeledPoint] =
      dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          LabeledPoint(label, features)
      }

    // Calculate gains of all features (features that are always zero will be dropped)
    new InfoGainCalculator(input).calculateIG().collect()
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): InfoGainSelector = defaultCopy(extra)

  @Since("2.1.1")
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): InfoGainSelectorModel = {
    new InfoGainSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

object InfoGainSelector extends DefaultParamsReadable[InfoGainSelector] {
  @Since("2.1.1")
  override def load(path: String): InfoGainSelector = super.load(path)
}

/**
  * Model fitted by [[InfoGainSelector]].
  * @param selectedFeatures list of indices to select (filter)
  */
@Since("2.1.1")
final class InfoGainSelectorModel private[filter] (@Since("2.1.1") override val uid: String,
                                                   @Since("2.1.1") override val selectedFeatures: Array[Int],
                                                   @Since("2.1.1") override val featureImportances: Map[String, Double])
  extends FeatureSelectorModel[InfoGainSelectorModel](uid, selectedFeatures, featureImportances) {

  @Since("2.1.1")
  override def copy(extra: ParamMap): InfoGainSelectorModel = {
    val copied = new InfoGainSelectorModel(uid, selectedFeatures, featureImportances)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.1.1")
  override def write: MLWriter = new FeatureSelectorModel.FeatureSelectorModelWriter[InfoGainSelectorModel](this)
}

@Since("2.1.1")
object InfoGainSelectorModel extends MLReadable[InfoGainSelectorModel] {
  @Since("2.1.1")
  override def read: MLReader[InfoGainSelectorModel] = new InfoGainSelectorModelReader

  @Since("2.1.1")
  override def load(path: String): InfoGainSelectorModel = super.load(path)
}

@Since("2.1.1")
final class InfoGainSelectorModelReader extends FeatureSelectorModel.FeatureSelectorModelReader[InfoGainSelectorModel]{
  @Since("2.1.1")
  override protected val className: String = classOf[InfoGainSelectorModel].getName

  @Since("2.1.1")
  override protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): InfoGainSelectorModel = {
    new InfoGainSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

private [filter] class InfoGainCalculator (val data: RDD[LabeledPoint]) {
  def calculateIG(): RDD[(Int, Double)] = {
    val LOG2 = math.log(2)

    /** log base 2 of x
      * @return log base 2 of x */
    val log2 = { x: Double => math.log(x) / LOG2 }
    /** entropy of x
      *  @return entropy of x */
    val entropy = { x: Double => if (x == 0) 0 else -x * log2(x) }

    val labels2Int = data.map(_.label).distinct.collect.zipWithIndex.toMap
    val nLabels = labels2Int.size

    // Basic info. about the dataset
    val classDistrib = data.map(d => labels2Int(d.label)).countByValue().toMap

    // Generate pairs ((featureID, featureVal), (Hot encoded) targetVal)
    val featureValues =
      data.flatMap({
        case LabeledPoint(label, dv: DenseVector) =>
          val c = Array.fill[Long](nLabels)(0L)
          c(labels2Int(label)) = 1L
          for (i <- dv.values.indices) yield ((i, dv(i).toFloat), c)
        case LabeledPoint(label, sv: SparseVector) =>
          val c = Array.fill[Long](nLabels)(0L)
          c(labels2Int(label)) = 1L
          for (i <- sv.indices.indices) yield ((sv.indices(i), sv.values(i).toFloat), c)
      })

    val sortedValues = getSortedDistinctValues(classDistrib, featureValues)

    val numSamples = classDistrib.values.sum

    // Calculate Probabilities
    val classDistribProb = classDistrib.map { case (k, v) => (k, v.toDouble / numSamples) }
    val featureProbs = sortedValues.map { case ((k, v), a) => ((k, v), a.sum.toDouble / numSamples) }
    val jointProbab = sortedValues.map { case ((k, v), a) => ((k, v), a.map(elem => elem.toDouble / numSamples)) }

    val jpTable = jointProbab.groupBy { case ((k, v), a) => k }
    val fpTable = featureProbs.groupBy { case ((k, v), a) => k }

    // Calculate entropies
    val featureEntropies = fpTable.map { case (k, v) => (k, v.map { case ((k1, v1), a) => entropy(a) }.sum) }.sortByKey(ascending = true)
    val jointEntropies = jpTable.map { case (k, v) => (k, v.map { case ((k1, v1), a) => a.map(v2 => entropy(v2)).sum }.sum) }.sortByKey(ascending = true)
    val targetEntropy = classDistribProb.foldLeft(0.0) { case (acc, (k, v)) => acc + entropy(v) }

    // Calculate information gain: targetEntropy + featureEntropy - jointEntropy
    // Format: RDD[(featureID->InformationGain)]

    val spark = featureEntropies.sparkContext

    spark.parallelize(featureEntropies.collect()).zip(spark.parallelize(jointEntropies.collect())).map { case ((k1, v1), (k2, v2)) => k1 -> (targetEntropy + v1 - v2) }

    //    customZip(featureEntropies, jointEntropies).map { case ((k1, v1), (k2, v2)) => k1->(targetEntropy + v1 - v2)}
  }

  /**
    * Group elements by feature and point (get distinct points).
    * Since values like (0, Float.NaN) are not considered unique when calling reduceByKey,
    * use the serialized version of the tuple.
    *
    * @return sorted list of unique feature values
    */
  private def getSortedDistinctValues(classDistrib: Map[Int, Long],
                                      featureValues: RDD[((Int, Float), Array[Long])]): RDD[((Int, Float), Array[Long])] = {

    val nonZeros: RDD[((Int, Float), Array[Long])] =
      featureValues.map(y => (y._1._1 + "," + y._1._2, y._2)).reduceByKey { case (v1, v2) =>
        (v1, v2).zipped.map(_ + _)
      }.map(y => {
        val s = y._1.split(",")
        ((s(0).toInt, s(1).toFloat), y._2)
      })

    val zeros = addZerosIfNeeded(nonZeros, classDistrib)
    val distinctValues = nonZeros.union(zeros)

    // Sort these values to perform the boundary points evaluation
    distinctValues.sortByKey()
  }

  /**
    * Add zeros if dealing with sparse data
    * Features that do not have any non-zero value will not be added
    *
    * @return rdd with 0's filled in
    */
  private def addZerosIfNeeded(nonZeros: RDD[((Int, Float), Array[Long])],
                               classDistrib: Map[Int, Long]): RDD[((Int, Float), Array[Long])] = {
    nonZeros.map { case ((k, p), v) => (k, v) }
      .reduceByKey { case (v1, v2) => (v1, v2).zipped.map(_ + _) }
      .map { case (k, v) =>
        val v2 = for (i <- v.indices) yield classDistrib(i) - v(i)
        ((k, 0.0F), v2.toArray)
      }.filter { case (_, v) => v.sum > 0 }
  }
}