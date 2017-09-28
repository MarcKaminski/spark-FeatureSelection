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
  library(dplyr)
  data(iris)
  df <- iris %>%
    dplyr::mutate(label = ifelse(Species == "setosa", 0.0, ifelse(Species == "versicolor", 1.0, 2.0))) %>%
    dplyr::select("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "label")
  print(cor(df, method = "pearson"))
  print(cor(df, method = "spearman"))
 */

class CorrelationSelectorSuite extends FeatureSelectionTestBase {
  test("Test CorrelationSelector - pearson: numTopFeatures") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("pearson")
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(3)

    val importantColNames = Array("pWidth", "pLength", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test CorrelationSelector - pearson: percentile") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("pearson")
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51)

    val importantColNames = Array("pWidth", "pLength", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test CorrelationSelector - pearson: randomCutOff") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("pearson")
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0)

    val importantColNames = Array("pWidth", "pLength", "sLength", "sWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test CorrelationSelector - spearman: numTopFeatures") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("spearman")
      .setOutputCol("filtered").setSelectorType("numTopFeatures").setNumTopFeatures(3)

    val importantColNames = Array("pWidth", "pLength", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test CorrelationSelector - spearman: percentile") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("spearman")
      .setOutputCol("filtered").setSelectorType("percentile").setPercentile(0.51)

    val importantColNames = Array("pWidth", "pLength", "sLength")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("Test CorrelationSelector - spearman: randomCutOff") {
    val selector = new CorrelationSelector().setFeaturesCol(featuresColName).setLabelCol(labelColName).setCorrelationType("spearman")
      .setOutputCol("filtered").setSelectorType("randomCutOff").setRandomCutOff(1.0)

    val importantColNames = Array("pWidth", "pLength", "sLength", "sWidth")
    val df = new VectorAssembler().setInputCols(importantColNames).setOutputCol("ImportantFeatures").transform(dataset)

    FeatureSelectorTestBase.testSelector[CorrelationSelector, CorrelationSelectorModel](selector, df, importantColNames, "ImportantFeatures")
  }

  test("CorrelationSelector read/write") {
    val nb = new CorrelationSelector
    testEstimatorAndModelReadWrite[CorrelationSelector, CorrelationSelectorModel](nb, dataset,
      FeatureSelectorTestBase.allParamSettings.+("correlationType" -> "pearson"),
      FeatureSelectorTestBase.checkModelData)
  }
}

