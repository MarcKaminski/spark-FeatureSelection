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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.selection.{FeatureSelectionTestBase, FeatureSelectorTestBase}

/*  To verify the results with R, run:
  library(RWeka)
  data(iris)
  weights <- InfoGainAttributeEval(Species ~ ., data=iris,)
  print(weights)
 */
class InfoGainSelectorSuite extends FeatureSelectionTestBase {
  test("Test InfoGainSelector: numTopFeatures") {
    val selector = new InfoGainSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(2)

    val importantColNames = Array("pLength", "pWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[InfoGainSelector, InfoGainSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test InfoGainSelector: percentile") {
    val selector = new InfoGainSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51)

    val importantColNames = Array("pLength", "pWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[InfoGainSelector, InfoGainSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test InfoGainSelector: randomCutOff") {
    val selector = new InfoGainSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0)

    val importantColNames = Array("pLength", "pWidth", "sLength", "sWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[InfoGainSelector, InfoGainSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("InfoGainSelector read/write") {
    val nb = new InfoGainSelector
    testEstimatorAndModelReadWrite[InfoGainSelector, InfoGainSelectorModel](nb, dataset, FeatureSelectorTestBase.allParamSettings, FeatureSelectorTestBase.checkModelData)
  }
}

