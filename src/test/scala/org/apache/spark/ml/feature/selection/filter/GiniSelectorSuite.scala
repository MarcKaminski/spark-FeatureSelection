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
  library(CORElearn)
  data(iris)
  weights <-   attrEval(Species ~ ., data=iris,  estimator = "Gini")
  print(weights)
 */
class GiniSelectorSuite extends FeatureSelectionTestBase {
  test("Test GiniSelector: numTopFeatures") {
    val selector = new GiniSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(2)

    val importantColNames = Array("pLength", "pWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[GiniSelector, GiniSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test GiniSelector: percentile") {
    val selector = new GiniSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51)

    val importantColNames = Array("pLength", "pWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[GiniSelector, GiniSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test GiniSelector: randomCutOff") {
    val selector = new GiniSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0)

    val importantColNames = Array("pLength", "pWidth", "sLength", "sWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[GiniSelector, GiniSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("GiniSelector read/write") {
    val nb = new GiniSelector
    testEstimatorAndModelReadWrite[GiniSelector, GiniSelectorModel](nb, dataset, FeatureSelectorTestBase.allParamSettings, FeatureSelectorTestBase.checkModelData)
  }
}

