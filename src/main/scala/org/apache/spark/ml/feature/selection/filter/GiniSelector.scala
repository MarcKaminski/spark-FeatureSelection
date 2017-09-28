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
  * Feature selection based on Gini index.
  */
@Since("2.1.1")
final class GiniSelector @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends FeatureSelector[GiniSelector, GiniSelectorModel] {

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("giniSelector"))

  @Since("2.1.1")
  override def train(dataset: Dataset[_]): Array[(Int, Double)] = {
    val input: RDD[LabeledPoint] =
      dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          LabeledPoint(label, features)
      }

    // Calculate gini indices of all features
    new GiniCalculator(input).calculateGini().collect()
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): GiniSelector = defaultCopy(extra)

  @Since("2.1.1")
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): GiniSelectorModel = {
    new GiniSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

object GiniSelector extends DefaultParamsReadable[GiniSelector] {
  @Since("2.1.1")
  override def load(path: String): GiniSelector = super.load(path)
}

/**
  * Model fitted by [[GiniSelector]].
  * @param selectedFeatures list of indices to select (filter)
  */
@Since("2.1.1")
final class GiniSelectorModel private[selection] (@Since("2.1.1") override val uid: String,
                                                  @Since("2.1.1") override val selectedFeatures: Array[Int],
                                                  @Since("2.1.1") override val featureImportances: Map[String, Double])
  extends FeatureSelectorModel[GiniSelectorModel](uid, selectedFeatures, featureImportances) {

  @Since("2.1.1")
  override def copy(extra: ParamMap): GiniSelectorModel = {
    val copied = new GiniSelectorModel(uid, selectedFeatures, featureImportances)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.1.1")
  override def write: MLWriter = new FeatureSelectorModel.FeatureSelectorModelWriter(this)
}

@Since("2.1.1")
object GiniSelectorModel extends MLReadable[GiniSelectorModel] {
  @Since("2.1.1")
  override def read: MLReader[GiniSelectorModel] = new GiniSelectorModelReader

  @Since("2.1.1")
  override def load(path: String): GiniSelectorModel = super.load(path)
}

@Since("2.1.1")
final class GiniSelectorModelReader extends FeatureSelectorModel.FeatureSelectorModelReader[GiniSelectorModel]{
  @Since("2.1.1")
  override protected val className: String = classOf[GiniSelectorModel].getName

  @Since("2.1.1")
  override protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): GiniSelectorModel = {
    new GiniSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

private [filter] class GiniCalculator (val data: RDD[LabeledPoint]) {
  def calculateGini(): RDD[(Int, Double)] = {
    val labels2Int = data.map(_.label).distinct.collect.zipWithIndex.toMap
    val nLabels = labels2Int.size

    // Basic info. about the dataset
    val classDistrib = data.map(d => labels2Int(d.label)).countByValue().toMap
    val count = data.count

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

    // Calculate Gini Indices for all features
    // P(X_j) - prior probability of feature X having value X_j ([[featurePrior]])
    // P(Y_c | X_j) - cond. probability that a sample is of class Y_c, given that feature X has value X_j ([[condProbabClassGivenFeature]])
    // P(Y_c) - prior probability that the label Y has value Y_c ([[classPrior]])
    // Gini(X) = Sum_j{P(X_j) * Sum_c{P(Y_c | X_j)²}} - Sum_c{P(Y_c)}
    // Step 1: (featureNumber, List[featureValue, Array[classCount]])
    val featureClassDistrib = getSortedDistinctValues(classDistrib, featureValues)
      .map { case ((fk, fv), cd) => (fk, (fv, cd)) }
      .groupBy { case (fk, (_, _)) => fk }
      .map { case (fk, arr) =>
        (fk, arr.map { case (_, value) => value })
      }

    // Step 2: Calculate class priors and right sum
    val classPrior = classDistrib.map { case (k, v) => (k, v.toDouble / count) }
    val rightPart = classPrior.foldLeft(0.0) { case (agg, (_, v)) => agg + v * v }

    // Step 3: Calculate left sum
    val condProbabClassGivenFeatureValue = featureClassDistrib
      .map { case (fk, arr) => (fk, arr
        .map { case (fv, cd) => (fv, cd
          .map(cVal => cVal.toDouble / cd.sum))
        })
      }
    val condProbabClassGivenFeatureValueSum = condProbabClassGivenFeatureValue
      .map { case (fk, arr) => (fk, arr
        .map { case (fv, cc) => (fv, cc
          .map(c => c * c).sum)
        })
      }
      .map { case (fk, arr) => (fk, arr.toList
        .sortBy(_._1))
      }
      .sortBy(_._1)

    val featurePrior = featureClassDistrib
      .map { case (fk, arr) => (fk, arr
        .map { case (fv, cc) => (fv, cc.sum.toDouble / count) })
      }
      .map { case (fk, arr) => (fk, arr.toList.sortBy(_._1))
      }
      .sortBy(_._1)

    // Parallelize again, so zip won't fail
    val spark = condProbabClassGivenFeatureValueSum.sparkContext
    val tmp = spark.parallelize(condProbabClassGivenFeatureValueSum.collect())
      .zip(spark.parallelize(featurePrior.collect()))
      .map { case ((fk, arr), (fk2, arr2)) => if (fk == fk2) {
        (fk, arr.zip(arr2))
      } else {
        throw new IllegalStateException("Featurekeys don't match! This should never happen.")
      }
      }

    val leftPart = tmp.map { case (fk, calc) => (fk, calc.foldLeft(0.0) {
      case (agg, (x, y)) => if (x._1 == y._1) {
        agg + x._2 * y._2
      } else {
        throw new IllegalStateException("Featurevalues don't match! This should never happen.")
      }
    })
    }

    // Step 3: Calculate Gini indices and return
    leftPart.map { case (fk, value) => (fk, value - rightPart) }
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
    *
    * @return rdd with 0's filled in
    */
  private def addZerosIfNeeded(nonZeros: RDD[((Int, Float), Array[Long])],
                               classDistrib: Map[Int, Long]): RDD[((Int, Float), Array[Long])] = {
    nonZeros.map { case ((k, _), v) => (k, v) }
      .reduceByKey { case (v1, v2) => (v1, v2).zipped.map(_ + _) }
      .map { case (k, v) =>
        val v2 = for (i <- v.indices) yield classDistrib(i) - v(i)
        ((k, 0.0F), v2.toArray)
      }.filter { case (_, v) => v.sum > 0 }
  }

}