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

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable

/**
  * A feature transformer that merges multiple columns into a vector column, without keeping duplicates.
  * The Transformer has two modes, triggered by useFeaturesCol:
  * 1) useFeaturesCol true and featuresCol set: the output column will contain columns from featuresCol that have
  *    names appearing in one of the inputCols (type vector)
  * 2) useFeaturesCol false: the output column will contain the columns from the inputColumns, but dropping duplicates
  */
@Since("2.1.1")
class VectorMerger @Since("2.1.1") (@Since("2.1.1") override val uid: String)
  extends Transformer with HasFeaturesCol with HasInputCols with HasOutputCol with DefaultParamsWritable {

  @Since("2.1.1")
  def this() = this(Identifiable.randomUID("vecAssemblerMerger"))

  /** @group setParam */
  @Since("2.1.1")
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  @Since("2.1.1")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("2.1.1")
  def setFeatureCol(value: String): this.type = set(featuresCol, value)

  /** @group param */
  @Since("2.1.1")
  final val useFeaturesCol = new BooleanParam(this, "useFeaturesCol",
    "The output column will contain columns from featuresCol that have names appearing in one of the inputCols (type vector)")
  setDefault(useFeaturesCol, true)

  /** @group getParam */
  @Since("2.1.1")
  def getUseFeaturesCol: Boolean = $(useFeaturesCol)

  /** @group setParam */
  @Since("2.1.1")
  def setUseFeaturesCol(value: Boolean): this.type = set(useFeaturesCol, value)

  @Since("2.1.1")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    if ($(useFeaturesCol))
      transformUsingFeaturesColumn(dataset)
    else
      transformUsingInputColumns(dataset)
  }

  @Since("2.1.1")
  private def transformUsingInputColumns(dataset: Dataset[_]): DataFrame = {
    // Schema transformation.
    val schema = dataset.schema
    lazy val first = dataset.toDF.first()

    val uniqueNames = mutable.ArrayBuffer[String]()

    val indicesToKeep = mutable.ArrayBuilder.make[Int]
    var cur = 0

    def doNotAdd(): Option[Nothing] = {
      cur += 1
      None
    }

    def addAndIncrementIndex() {
      indicesToKeep += cur
      cur += 1
    }

    val attrs: Array[Attribute] = $(inputCols).flatMap { c =>
      val field = schema(c)
      val index = schema.fieldIndex(c)
      field.dataType match {
        case _: VectorUDT =>
          val group = AttributeGroup.fromStructField(field)
          if (group.attributes.isDefined) {
            // If attributes are defined, copy them, checking for duplicates and preserving name.
            group.attributes.get.zipWithIndex.flatMap { case (attr, i) =>
              if (attr.name.isDefined) {
                val name = attr.name.get
                if (!uniqueNames.contains(name)) {
                  addAndIncrementIndex()
                  uniqueNames.append(name)
                  Some(attr)
                } else
                  doNotAdd()
              } else {
                addAndIncrementIndex()
                Some(attr.withName(c + "_" + i))
              }
            }.toList
          } else {
            // Otherwise, treat all attributes as numeric. If we cannot get the number of attributes
            // from metadata, check the first row.
            val numAttrs = group.numAttributes.getOrElse(first.getAs[Vector](index).size)
            Array.tabulate(numAttrs)(i => {
              addAndIncrementIndex()
              NumericAttribute.defaultAttr.withName(c + "_" + i)
            })
          }
        case otherType =>
          throw new SparkException(s"VectorMerger does not support the $otherType type")
      }
    }
    val metadata = new AttributeGroup($(outputCol), attrs).toMetadata()

    // Data transformation.
    val assembleFunc = udf { r: Row =>
      VectorMerger.assemble(indicesToKeep.result(), r.toSeq: _*)
    }

    val args = $(inputCols).map { c =>
      schema(c).dataType match {
        case DoubleType => dataset(c)
        case _: VectorUDT => dataset(c)
        case _: NumericType | BooleanType => dataset(c).cast(DoubleType).as(s"${c}_double_$uid")
      }
    }

    dataset.select(col("*"), assembleFunc(struct(args: _*)).as($(outputCol), metadata))
  }

  @Since("2.1.1")
  private def transformUsingFeaturesColumn(dataset: Dataset[_]): DataFrame = {
    // Schema transformation.
    val schema = dataset.schema

    val notUniqueNames = mutable.ArrayBuffer[String]()
    val featuresColName = $(featuresCol)
    val featureColAttrs = AttributeGroup.fromStructField(schema(featuresColName)).attributes.get.zipWithIndex
    val featuresColNames = featureColAttrs.flatMap { case (attr, _) => attr.name }

    $(inputCols).foreach { c =>
      val field = schema(c)
      field.dataType match {
        case _: VectorUDT =>
          val group = AttributeGroup.fromStructField(field)
          if (group.attributes.isDefined) {
            // If attributes are defined, remember name to get column from $featureCol.
            group.attributes.get.zipWithIndex.foreach { case (attr, _) =>
              if (attr.name.isDefined) {
                val name = attr.name.get
                if (featuresColNames.contains(name))
                  notUniqueNames.append(name)
                else
                  throw new IllegalArgumentException(s"Features column $featuresColName does not contain column with name $name!")
              } else
                throw new IllegalArgumentException(s"Input column $c contains column without name attribute!")
            }
          } else {
            // Otherwise, merging not possible
            throw new IllegalArgumentException(s"Input column $c does not contain attributes!")
          }
      }
    }

    val uniqueNames = notUniqueNames.toSet

    new VectorSlicer()
      .setInputCol(featuresColName)
      .setNames(uniqueNames.toArray)
      .setOutputCol($(outputCol))
      .transform(dataset)
  }

  @Since("2.1.1")
  override def transformSchema(schema: StructType): StructType = {
    if ($(useFeaturesCol)) {
      val featuresColName = $(featuresCol)

      if (!schema(featuresColName).dataType.isInstanceOf[VectorUDT])
        throw new IllegalArgumentException(s"Features column $featuresColName is not of type VectorUDT!")
    }

    val inputColNames = $(inputCols)
    val outputColName = $(outputCol)

    inputColNames.foreach(name => if (!schema(name).dataType.isInstanceOf[VectorUDT])
      throw new IllegalArgumentException(s"Input column $name is not of type VectorUDT!")
    )

    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }

    StructType(schema.fields :+ StructField(outputColName, new VectorUDT, nullable = true))
  }

  @Since("2.1.1")
  override def copy(extra: ParamMap): VectorMerger = defaultCopy(extra)
}

@Since("2.1.1")
object VectorMerger extends DefaultParamsReadable[VectorMerger] {

  @Since("2.1.1")
  override def load(path: String): VectorMerger = super.load(path)

  private[feature] def assemble(indicesToKeep: Array[Int], vv: Any*): Vector = {
    val indices = mutable.ArrayBuilder.make[Int]
    val values = mutable.ArrayBuilder.make[Double]

    var returnCur = 0
    var globalCur = 0
    vv.foreach {
      case v: Double =>
        if (indicesToKeep.contains(globalCur)) {
          if (v != 0.0) {
            indices += returnCur
            values += v
          }
          returnCur += 1
        }
        globalCur += 1
      case vec: Vector =>
        vec.toDense.foreachActive { case (_, v) =>
          if (indicesToKeep.contains(globalCur)) {
            if (v != 0.0) {
              indices += returnCur
              values += v
            }
            returnCur += 1
          }
          globalCur += 1
        }
      case null =>
        throw new SparkException("Values to assemble cannot be null.")
      case o =>
        throw new SparkException(s"$o of type ${o.getClass.getName} is not supported.")
    }
    Vectors.sparse(returnCur, indices.result(), values.result()).compressed
  }
}
