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

package org.apache.spark.ml.feature.selection.util

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.selection.{FeatureSelectionTestBase, FeatureSelectorTestBase}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

class VectorMergerSuite extends FeatureSelectionTestBase {

  test("VectorMerger: merges two VectorColumns with different names") {
    val dfTmp = new VectorAssembler().setInputCols(Array("pWidth", "pLength")).setOutputCol("vector1").transform(dataset)
    val dfTmp2 = new VectorAssembler().setInputCols(Array("sWidth", "sLength")).setOutputCol("vector2").transform(dfTmp)
    val df = new VectorAssembler().setInputCols(Array("pWidth", "pLength", "sWidth", "sLength")).setOutputCol("expected").transform(dfTmp2)

    val dfT = new VectorMerger().setInputCols(Array("vector1", "vector2")).setOutputCol("merged").transform(df)

    val outCols = AttributeGroup.fromStructField(dfT.schema("merged")).attributes.get.map(attr => attr.name.get)

    assert(outCols.length == 4, "Length of merged column is not equal to 4!")

    assert(outCols.sorted sameElements Array("pWidth", "pLength", "sWidth", "sLength").sorted,
      "Input and output column names do not match!")

    dfT.select("merged", "expected").collect()
      .foreach { case Row(vec1: Vector, vec2: Vector) =>
        assert(FeatureSelectorTestBase.tolerantVectorEquality(vec1, vec2), "column in merged and expected do not match!")
      }
  }

  test("VectorMerger: merges two VectorColumns with duplicate names") {
    val dfTmp = new VectorAssembler().setInputCols(Array("pWidth", "pLength")).setOutputCol("vector1").transform(dataset)
    val dfTmp2 = new VectorAssembler().setInputCols(Array("pLength", "sLength")).setOutputCol("vector2").transform(dfTmp)
    val df = new VectorAssembler().setInputCols(Array("pWidth", "pLength", "sLength")).setOutputCol("expected").transform(dfTmp2)

    val dfT = new VectorMerger().setInputCols(Array("vector1", "vector2")).setOutputCol("merged").transform(df)

    val outCols = AttributeGroup.fromStructField(dfT.schema("merged")).attributes.get.map(attr => attr.name.get)

    assert(outCols.length == 3, "Length of merged column is not equal to 3!")

    assert(outCols.sorted sameElements Array("pWidth", "pLength", "sLength").sorted,
      "Input and output column names do not match!")

    dfT.select("merged", "expected").collect()
      .foreach { case Row(vec1: Vector, vec2: Vector) =>
        assert(FeatureSelectorTestBase.tolerantVectorEquality(vec1, vec2), "column in merged and expected do not match!")
      }
  }

  test("VectorMerger - useFeatureCol: merges two VectorColumns with different names") {
    val dfTmp = new VectorAssembler().setInputCols(Array("pWidth", "pLength")).setOutputCol("vector1").transform(dataset)
    val dfTmp2 = new VectorAssembler().setInputCols(Array("sWidth", "sLength")).setOutputCol("vector2").transform(dfTmp)
    // The features column has a different ordering to test if the correct values are taken for each column
    val dfTp3 = new VectorAssembler().setInputCols(Array("pWidth", "sLength", "sWidth", "pLength")).setOutputCol("formerging").transform(dfTmp2)
    val df = new VectorAssembler().setInputCols(Array("pWidth", "pLength", "sWidth", "sLength")).setOutputCol("expected").transform(dfTp3)

    val dfT = new VectorMerger()
      .setInputCols(Array("vector1", "vector2"))
      .setFeatureCol("formerging")
      .setUseFeaturesCol(true)
      .setOutputCol("merged").transform(df)

    val outCols = AttributeGroup.fromStructField(dfT.schema("merged")).attributes.get.map(attr => attr.name.get)

    assert(outCols.length == 4, "Length of merged column is not equal to 4!")

    assert(outCols.sorted sameElements Array("pWidth", "pLength", "sWidth", "sLength").sorted,
      "Input and output column names do not match!")

    dfT.select("merged", "expected").collect()
      .foreach { case Row(vec1: Vector, vec2: Vector) =>
        assert(FeatureSelectorTestBase.tolerantVectorEquality(vec1, vec2), "column in merged and expected do not match!")
      }
  }

  test("VectorMerger - useFeatureCol: merges two VectorColumns with duplicate names") {
    val dfTmp = new VectorAssembler().setInputCols(Array("pWidth", "pLength")).setOutputCol("vector1").transform(dataset)
    val dfTmp2 = new VectorAssembler().setInputCols(Array("pLength", "sLength")).setOutputCol("vector2").transform(dfTmp)
    // The features column has a different ordering to test if the correct values are taken for each column
    val dfTp3 = new VectorAssembler().setInputCols(Array("pWidth", "sLength", "sWidth", "pLength")).setOutputCol("formerging").transform(dfTmp2)
    val df = new VectorAssembler().setInputCols(Array("pWidth", "pLength", "sLength")).setOutputCol("expected").transform(dfTp3)

    val dfT = new VectorMerger()
      .setInputCols(Array("vector1", "vector2"))
      .setFeatureCol("formerging")
      .setUseFeaturesCol(true)
      .setOutputCol("merged").transform(df)

    val outCols = AttributeGroup.fromStructField(dfT.schema("merged")).attributes.get.map(attr => attr.name.get)

    assert(outCols.length == 3, "Length of merged column is not equal to 3!")

    assert(outCols.sorted sameElements Array("pWidth", "pLength", "sLength").sorted,
      "Input and output column names do not match!")

    dfT.select("merged", "expected").collect()
      .foreach { case Row(vec1: Vector, vec2: Vector) =>
        assert(FeatureSelectorTestBase.tolerantVectorEquality(vec1, vec2), "column in merged and expected do not match!")
      }
  }

  test("VectorMerger read/write") {
    val nb = new VectorMerger
    testDefaultReadWrite[VectorMerger](nb, testParams = true)
  }
}
