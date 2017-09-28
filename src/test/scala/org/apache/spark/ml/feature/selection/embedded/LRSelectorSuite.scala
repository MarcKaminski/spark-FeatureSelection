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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.selection.{FeatureSelectionTestBase, FeatureSelectorTestBase}
import org.apache.spark.ml.linalg.Matrices

class LRSelectorSuite extends FeatureSelectionTestBase {
  // Order of feature importances must be: f4 > f3 > f2 > f1
  private val lrWeights = Matrices.dense(3, 4, Array(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, -0.8, -0.8, -0.8, 0.9, 0.9, 0.9))

  test("Test LRSelector: numTopFeatures") {
    val selector = new LRSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCoefficientMatrix(lrWeights)
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(2)

    val importantColNames = Array("pWidth", "pLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[LRSelector, LRSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test LRSelector: percentile") {
    val selector = new LRSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51).setCoefficientMatrix(lrWeights)

    val importantColNames = Array("pWidth", "pLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[LRSelector, LRSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test LRSelector: randomCutOff") {
    val selector = new LRSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0).setCoefficientMatrix(lrWeights)

    val importantColNames = Array("pWidth", "pLength", "sWidth", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[LRSelector, LRSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("LRSelector read/write") {
    val nb = new LRSelector
    testEstimatorAndModelReadWrite[LRSelector, LRSelectorModel](nb, dataset,
      FeatureSelectorTestBase.allParamSettings.+("coefficientMatrix" -> lrWeights), FeatureSelectorTestBase.checkModelData)
  }
}