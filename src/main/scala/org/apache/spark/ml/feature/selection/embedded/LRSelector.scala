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


package org.apache.spark.ml.feature.selection.embedded

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.feature.selection.{FeatureSelector, FeatureSelectorModel}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{Param, _}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.{Vectors => MllibVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{JArray, JDouble, JInt, JObject, JValue}

/**
  * Specialized version of `Param[Matrix]` for Java.
  */
@DeveloperApi
private[embedded] class MatrixParam(parent: String, name: String, doc: String, isValid: Matrix => Boolean)
  extends Param[Matrix](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (mat: Matrix) => true)

  def this(parent: Identifiable, name: String, doc: String, isValid: Matrix => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: Matrix): ParamPair[Matrix] = super.w(value)

  override def jsonEncode(value: Matrix): String = {
    compact(render(MatrixParam.jValueEncode(value)))
  }

  override def jsonDecode(json: String): Matrix = {
    MatrixParam.jValueDecode(parse(json))
  }
}

private[embedded] object MatrixParam {
  /** Encodes a param value into JValue. */
  def jValueEncode(value: Matrix): JValue = {
    val rows = JInt(value.numRows)
    val cols = JInt(value.numCols)
    val vals = JArray(for (v <- value.toArray.toList) yield JDouble(v))

    JObject(List(("rows", rows), ("cols", cols), ("vals", vals)))
  }

  /** Decodes a param value from JValue. */
  def jValueDecode(jValue: JValue): Matrix = {
    var rows, cols: Int = 0
    var vals: Array[Double] = Array.empty[Double]

    var rowsSet, colsSet, valsSet = false

    jValue match {
      case obj: JObject => for (kvPair <- obj.values) {
        kvPair._2 match {
          case x: BigInt =>
            if (kvPair._1 == "rows") {
              rows = x.toInt
              rowsSet = true
            }
            else if (kvPair._1 == "cols") {
              cols = x.toInt
              colsSet = true
            }
            else
              throw new IllegalArgumentException(s"Cannot recognize unexpected key ${kvPair._1}. (Value is BigInt)")
          case arr: List[_] =>
            if (arr.forall { case _: Double => true; case _ => false })
              if (kvPair._1 == "vals") {
                vals = arr.asInstanceOf[List[Double]].toArray
                valsSet = true
              }
              else
                throw new IllegalArgumentException(s"Cannot decode unexpected key ${kvPair._1} to Matrix.")
            else
              throw new IllegalArgumentException(s"Cannot decode unexpected key ${kvPair._1} with value: ${kvPair._2}.")
          case _ => throw new IllegalArgumentException(s"Cannot decode unexpected key ${kvPair._1} with value: ${kvPair._2}.")
        }
      }
      case _ =>
        throw new IllegalArgumentException(s"Cannot decode $jValue to Matrix.")
    }

    if (colsSet && rowsSet && valsSet)
      Matrices.dense(rows, cols, vals)
    else
      throw new IllegalArgumentException(s"Cannot decode $jValue. " +
        s"Missing values to create Matrix: colsSet: $colsSet; rowsSet: $rowsSet; valsSet: $valsSet")
  }
}

/**
  * Params for [[LRSelector]] and [[LRSelectorModel]].
  */

private[embedded] trait LRSelectorParams extends Params {
  /**
    * Param for coefficientMatrix.
    *
    * @group param
    */
  final val coefficientMatrix: MatrixParam = new MatrixParam(this, "coefficientMatrix", "coefficientMatrix of LR model")

  /** @group getParam */
  final def getCoefficientMatrix: Matrix = $(coefficientMatrix)

  /**
    * Choose, if coefficients shall be scaled using the maximum value of the corresponding feature.
    * Use, if no StandardScaler was used prior LR training
    * @group param
    */
  @Since("2.1.1")
  final val scaleCoefficients = new BooleanParam(this, "scaleCoefficients",
    "Scale the coefficients using the maximum values of the corresponding feature.")
  setDefault(scaleCoefficients, false)

  /** @group getParam */
  @Since("2.1.1")
  def getScaleCoefficients: Boolean = $(scaleCoefficients)
}

/**
  * Feature selection based on LR weights (absolute value).
  * The selector can scale the coefficients using the corresponding maximum feature value. To activate, set `scaleCoefficients` to true.
  * Default: false
  */
@Since("2.1.1")
final class LRSelector @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends FeatureSelector[LRSelector, LRSelectorModel] with LRSelectorParams {

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("lrSelector"))

  /** @group setParam */
  @Since("2.1.1")
  def setCoefficientMatrix(value: Matrix): this.type = set(coefficientMatrix, value)

  /** @group setParam */
  @Since("2.1.1")
  def setScaleCoefficients(value: Boolean): this.type = set(scaleCoefficients, value)

  @Since("2.1.1")
  override protected def train(dataset: Dataset[_]): Array[(Int, Double)] = {
    val input = dataset.select($(featuresCol)).rdd.map { case Row(features: Vector) => MllibVectors.fromML(features) }
    val inputStats = Statistics.colStats(input)
    // Calculate maxValues = max(abs(min(feature)), abs(max(feature)))
    val absMaxValues = inputStats.max.toArray.map(elem => math.abs(elem))
    val absMinValues = inputStats.min.toArray.map(elem => math.abs(elem))

    val maxValues = if (!$(scaleCoefficients))
      Array.fill(absMinValues.length)(1.0)
    else
      absMaxValues.zip(absMinValues).map { case (max, min) => math.max(max, min) }

    // Calculate normalized and absolute sum of LR weights for each feature
    val coeffVectors = $(coefficientMatrix).toArray.grouped($(coefficientMatrix).numRows).toArray

    val coeffFw = coeffVectors.map(col => col.map(elem => math.abs(elem)).sum)

    val scaled = coeffFw.zip(maxValues).map { case (coeff, max) => coeff * max }

    val coeffSum = scaled.sum

    val toScale = if ($(selectorType) == FeatureSelector.Random) {
      val coeffMean = if (scaled.length == 0) 0.0 else coeffSum / scaled.length
      scaled :+ coeffMean
    } else
      scaled

    val featureImportances = if (coeffSum == 0) toScale else toScale.map(elem => elem / coeffSum)

    featureImportances.zipWithIndex.map { case (value, index) => (index, value) }
  }

  @Since("2.1.1")
  override def transformSchema(schema: StructType): StructType = {
    val otherPairs = FeatureSelector.supportedSelectorTypes.filter(_ != $(selectorType))
    otherPairs.foreach { paramName: String =>
      if (isSet(getParam(paramName))) {
        logWarning(s"Param $paramName will take no effect when selector type = ${$(selectorType)}.")
      }
    }

    require(isDefined(coefficientMatrix))
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): LRSelector = defaultCopy(extra)

  @Since("2.1.1")
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): LRSelectorModel = {
    new LRSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

object LRSelector extends DefaultParamsReadable[LRSelector] {
  @Since("2.1.1")
  override def load(path: String): LRSelector = super.load(path)
}

/**
  * Model fitted by [[LRSelector]].
  * @param uid of Model
  * @param selectedFeatures list of indices to select
  * @param featureImportances Map that stores each feature importance
  */
@Since("2.1.1")
final class LRSelectorModel private[selection] (@Since("2.1.1") override val uid: String,
                                                @Since("2.1.1") override val selectedFeatures: Array[Int],
                                                @Since("2.1.1") override val featureImportances: Map[String, Double])
  extends FeatureSelectorModel[LRSelectorModel](uid, selectedFeatures, featureImportances) with LRSelectorParams {

  @Since("2.1.1")
  override def copy(extra: ParamMap): LRSelectorModel = {
    val copied = new LRSelectorModel(uid, selectedFeatures, featureImportances)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.1.1")
  override def write: MLWriter = new FeatureSelectorModel.FeatureSelectorModelWriter[LRSelectorModel](this)
}

@Since("2.1.1")
object LRSelectorModel extends MLReadable[LRSelectorModel] {
  @Since("2.1.1")
  override def read: MLReader[LRSelectorModel] = new LRSelectorModelReader

  @Since("2.1.1")
  override def load(path: String): LRSelectorModel = super.load(path)
}

@Since("2.1.1")
final class LRSelectorModelReader extends FeatureSelectorModel.FeatureSelectorModelReader[LRSelectorModel] {
  @Since("2.1.1")
  override protected val className: String = classOf[LRSelectorModel].getName

  @Since("2.1.1")
  override protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): LRSelectorModel = {
    new LRSelectorModel(uid, selectedFeatures, featureImportances)
  }
}