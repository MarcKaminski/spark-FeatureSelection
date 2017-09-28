name := "spark-FeatureSelection"

organization := "MarcKaminski"

version := "1.0.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"


// spark version to be used
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion  % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion  % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion  % "provided"
)

// For tests
parallelExecution in Test := false
fork in Test := false // true -> Spark during tests; false -> debug during tests  (for debug run sbt with: sbt -jvm-debug 5005)
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.1" % "test"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1" % "test"


/********************
  * Release settings *
  ********************/

publishMavenStyle := true

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

pomExtra :=
  <url>https://github.com/MarcKaminski/spark-FeatureSelection</url>
    <scm>
      <url>git@github.com:MarcKaminski/spark-FeatureSelection.git</url>
      <connection>scm:git:git@github.com:MarcKaminski/spark-FeatureSelection.git</connection>
    </scm>
    <developers>
      <developer>
        <id>MarcKaminski</id>
        <name>Marc Kaminski</name>
        <url>https://github.com/MarcKaminski</url>
      </developer>
    </developers>