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
import org.apache.spark.ml.linalg.Vectors

class ImportanceSelectorSuite extends FeatureSelectionTestBase {
  // Order of feature importances must be: f4 > f3 > f2 > f1
  private val featureWeights = Vectors.dense(Array(0.3, 0.5, 0.7, 0.8))

  test("Test ImportanceSelector: numTopFeatures") {
    val selector = new ImportanceSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setFeatureWeights(featureWeights)
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(2)

    val importantColNames = Array("pWidth", "pLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[ImportanceSelector, ImportanceSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test ImportanceSelector: percentile") {
    val selector = new ImportanceSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51).setFeatureWeights(featureWeights)

    val importantColNames = Array("pWidth", "pLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[ImportanceSelector, ImportanceSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test ImportanceSelector: randomCutOff") {
    val selector = new ImportanceSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName)
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0).setFeatureWeights(featureWeights)

    val importantColNames = Array("pWidth", "pLength", "sWidth", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[ImportanceSelector, ImportanceSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("ImportanceSelector read/write") {
    val nb = new ImportanceSelector
    testEstimatorAndModelReadWrite[ImportanceSelector, ImportanceSelectorModel](nb, dataset,
      FeatureSelectorTestBase.allParamSettings.+("featureWeights" -> featureWeights), FeatureSelectorTestBase.checkModelData)
  }
}