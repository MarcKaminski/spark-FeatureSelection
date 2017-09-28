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

import org.apache.spark.annotation.Since
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._

/**
  * Params for [[FeatureSelector]] and [[FeatureSelectorModel]].
  */
private[selection] trait FeatureSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol {

  /**
    * Number of features that selector will select. If the
    * number of features is less than numTopFeatures, then this will select all features.
    * Only applicable when selectorType = "numTopFeatures".
    * The default value of numTopFeatures is 50.
    *
    * @group param
    */
  @Since("2.1.1")
  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select. If the" +
      " number of features is < numTopFeatures, then this will select all features.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures -> 50)

  /** @group getParam */
  @Since("2.1.1")
  def getNumTopFeatures: Int = $(numTopFeatures)

  /**
    * Percentile of features that selector will select.
    * Only applicable when selectorType = "percentile".
    * Default value is 0.5.
    *
    * @group param
    */
  @Since("2.1.1")
  final val percentile = new DoubleParam(this, "percentile",
    "Percentile of features that selector will select.",
    ParamValidators.inRange(0, 1))
  setDefault(percentile -> 0.5)

  /** @group getParam */
  @Since("2.1.1")
  def getPercentile: Double = $(percentile)

  /**
    * Percentile of features that selector will select after the random column threshold (of number of remaining features).
    * Only applicable when selectorType = "randomCutOff".
    * Default value is 0.05.
    *
    * @group param
    */
  @Since("2.1.1")
  final val randomCutOff = new DoubleParam(this, "randomCutOff",
    "Percentile of features that selector will select after the random column threshold (of number of remaining features).",
    ParamValidators.inRange(0, 1))
  setDefault(percentile -> 0.05)

  /** @group getParam */
  @Since("2.1.1")
  def getRandomCutOff: Double = $(randomCutOff)

  /**
    * The selector type of the FeatureSelector.
    * Supported options: "numTopFeatures", "percentile" (default), .
    *
    * @group param
    */
  @Since("2.1.1")
  final val selectorType = new Param[String](this, "selectorType",
    "The selector type of the FeatureSelector. " +
      "Supported options: " + FeatureSelector.supportedSelectorTypes.mkString(", "),
    ParamValidators.inArray[String](FeatureSelector.supportedSelectorTypes))
  setDefault(selectorType -> FeatureSelector.Percentile)

  /** @group getParam */
  @Since("2.1.1")
  def getSelectorType: String = $(selectorType)
}
