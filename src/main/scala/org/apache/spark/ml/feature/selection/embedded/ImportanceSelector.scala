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

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.selection.{FeatureSelector, FeatureSelectorModel}
import org.apache.spark.ml.linalg.{Vector, _}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/**
  * Params for [[ImportanceSelector]] and [[ImportanceSelectorModel]].
  */

private[embedded] trait ImportanceSelectorParams extends Params {
  /**
    * Param for featureImportances.
    *
    * @group param
    */
  final val featureWeights: Param[Vector] = new Param[Vector](this, "featureWeights",
    "featureWeights to rank features and select from a column")

  /** @group getParam */
  final def getFeatureWeights: Vector = $(featureWeights)
}

/**
  * Feature selection based on featureImportances (e.g. from RandomForest).
  */
@Since("2.1.1")
final class ImportanceSelector @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends FeatureSelector[ImportanceSelector, ImportanceSelectorModel] with ImportanceSelectorParams {

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("importanceSelector"))

  /** @group setParam */
  @Since("2.1.1")
  def setFeatureWeights(value: Vector): this.type = set(featureWeights, value)

  @Since("2.1.1")
  override protected def train(dataset: Dataset[_]): Array[(Int, Double)] = {
    val arrBuilder = new mutable.ArrayBuffer[(Int, Double)]()
    $(featureWeights).foreachActive((idx, value) => arrBuilder.append((idx, value)))
    val featureImportancesLocal = arrBuilder.toArray

    if ($(selectorType) == FeatureSelector.Random) {
      val mean = featureImportancesLocal.map(_._2).sum / featureImportancesLocal.length
      featureImportancesLocal :+ (featureImportancesLocal.map(_._1).max + 1, mean)
    } else
      featureImportancesLocal
  }

  @Since("2.1.1")
  override def transformSchema(schema: StructType): StructType = {
    val otherPairs = FeatureSelector.supportedSelectorTypes.filter(_ != $(selectorType))
    otherPairs.foreach { paramName: String =>
      if (isSet(getParam(paramName))) {
        logWarning(s"Param $paramName will take no effect when selector type = ${$(selectorType)}.")
      }
    }

    require(isDefined(featureWeights))
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): ImportanceSelector = defaultCopy(extra)

  @Since("2.1.1")
  protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): ImportanceSelectorModel = {
    new ImportanceSelectorModel(uid, selectedFeatures, featureImportances)
  }
}

object ImportanceSelector extends DefaultParamsReadable[ImportanceSelector] {
  @Since("2.1.1")
  override def load(path: String): ImportanceSelector = super.load(path)
}

/**
  * Model fitted by [[ImportanceSelector]].
  * @param uid of Model
  * @param selectedFeatures list of indices to select
  * @param featureImportances Map that stores each feature importance
  */
@Since("2.1.1")
final class ImportanceSelectorModel private[selection] (@Since("2.1.1") override val uid: String,
                                                        @Since("2.1.1") override val selectedFeatures: Array[Int],
                                                        @Since("2.1.1") override val featureImportances: Map[String, Double])
  extends FeatureSelectorModel[ImportanceSelectorModel](uid, selectedFeatures, featureImportances) with ImportanceSelectorParams {
  @Since("2.1.1")
  override def copy(extra: ParamMap): ImportanceSelectorModel = {
    val copied = new ImportanceSelectorModel(uid, selectedFeatures, featureImportances)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.1.1")
  override def write: MLWriter = new FeatureSelectorModel.FeatureSelectorModelWriter[ImportanceSelectorModel](this)
}

@Since("2.1.1")
object ImportanceSelectorModel extends MLReadable[ImportanceSelectorModel] {
  @Since("2.1.1")
  override def read: MLReader[ImportanceSelectorModel] = new ImportanceSelectorModelReader

  @Since("2.1.1")
  override def load(path: String): ImportanceSelectorModel = super.load(path)
}

@Since("2.1.1")
final class ImportanceSelectorModelReader extends FeatureSelectorModel.FeatureSelectorModelReader[ImportanceSelectorModel] {
  @Since("2.1.1")
  override protected val className: String = classOf[ImportanceSelectorModel].getName

  @Since("2.1.1")
  override protected def make(uid: String, selectedFeatures: Array[Int], featureImportances: Map[String, Double]): ImportanceSelectorModel = {
    new ImportanceSelectorModel(uid, selectedFeatures, featureImportances)
  }
}