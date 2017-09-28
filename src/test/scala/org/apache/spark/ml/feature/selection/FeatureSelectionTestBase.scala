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

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.selection.test_util._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalactic.TolerantNumerics
import org.scalatest._

abstract class FeatureSelectionTestBase extends FunSuite with SharedSparkSession with BeforeAndAfter with DefaultReadWriteTest

trait SharedSparkSession extends BeforeAndAfterAll with BeforeAndAfterEach {
  self: Suite =>

  @transient private var _sc: SparkSession = _
  @transient var dataset: Dataset[_] = _

  private val testPath = getClass.getResource("/iris.data").getPath
  protected val featuresColName = "features"
  protected val labelColName = "Species"

  def sc: SparkSession = _sc

  override def beforeAll() {
    super.beforeAll()
    _sc = SparkSession
      .builder()
      .master("local[*]")
      .appName("spark test base")
      .getOrCreate()
    _sc.sparkContext.setLogLevel("ERROR")

    val df = sc.read.option("inferSchema", true).option("header", true).csv(testPath)
    dataset = new VectorAssembler()
      .setInputCols(Array("sLength", "sWidth", "pLength", "pWidth"))
      .setOutputCol(featuresColName)
      .transform(df)
  }

  override def afterAll() {
    try {
      if (_sc != null) {
        _sc.stop()
      }
      _sc = null
    } finally {
      super.afterAll()
    }
  }
}

object FeatureSelectorTestBase {
  private val epsilon = 1e-4f
  private implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)

  def tolerantVectorEquality(a: Vector, b: Vector): Boolean = {
    a.toArray.zip(b.toArray).map {
      case (a: Double, b: Double) => doubleEq.areEqual(a, b)
      case (a: Any, b: Any) => a == b
    }.forall(value => value)
  }

  def testSelector[
  Learner <: FeatureSelector[Learner, M],
  M <: FeatureSelectorModel[M]](selector: FeatureSelector[Learner, M],
                                dataset: Dataset[_],
                                importantColNames: Array[String],
                                groundTruthColname: String): Unit = {
    val selectorModel = selector.fit(dataset)
    val transformed = selectorModel.transform(dataset)

    val inputCols = AttributeGroup.fromStructField(transformed.schema(selector.getFeaturesCol))
      .attributes.get.map(attr => attr.name.get)

    assert(selectorModel.featureImportances.size == inputCols.length,
      "Length of featureImportances array is not equal to number of input columns!")

    val selectedColNames = AttributeGroup.fromStructField(transformed.schema(selector.getOutputCol))
      .attributes.get.map(attr => attr.name.get)

    val importantColsSelected = importantColNames.sorted.zip(selectedColNames.sorted).map(elem => elem._1 == elem._2).forall(elem => elem)

    assert(importantColsSelected, "Selected and important column names do not match!")

    transformed.select(selectorModel.getOutputCol, groundTruthColname).collect()
      .foreach { case Row(vec1: Vector, vec2: Vector) =>
        assert(tolerantVectorEquality(vec1, vec2))
      }
  }

  def checkModelData[M <: FeatureSelectorModel[M]](model1: FeatureSelectorModel[M], model2: FeatureSelectorModel[M]): Unit = {

    assert(model1.selectedFeatures sameElements model2.selectedFeatures
      , "Persisted model has different selectedFeatures.")
    assert(model1.featureImportances.toArray.sortBy(elem => elem._1) sameElements model2.featureImportances.toArray.sortBy(elem => elem._1),
      "Persisted model has different featureImportances.")
  }

  /**
    * Mapping from all Params to valid settings which differ from the defaults.
    * This is useful for tests which need to exercise all Params, such as save/load.
    * This excludes input columns to simplify some tests.
    */
  val allParamSettings: Map[String, Any] = Map(
    "selectorType" -> "percentile",
    "numTopFeatures" -> 1,
    "percentile" -> 0.12,
    "randomCutOff" -> 0.1,
    "featuresCol" -> "features",
    "labelCol" -> "Species",
    "outputCol" -> "myOutput"
  )
}
