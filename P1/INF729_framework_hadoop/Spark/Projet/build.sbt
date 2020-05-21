name := "spark_project_kickstarter_2019_2020"

version := "1.0"

organization := "paristech"

scalaVersion := "2.11.11"

val sparkVersion = "2.3.4"

libraryDependencies ++= Seq(
  // Spark dependencies. Marked as provided because they must not be included in the uber jar
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",

  // Third-party libraries
  "org.apache.hadoop" % "hadoop-aws" % "2.6.0" % "provided",
  "com.amazonaws" % "aws-java-sdk" % "1.7.4" % "provided"
  //"com.github.scopt" %% "scopt" % "3.4.0"        // to parse options given to the jar in the spark-submit
)

// A special option to exclude Scala itself form our assembly JAR, since Spark already bundles Scala.
assembly / assemblyOption := (assembly / assemblyOption).value.copy(includeScala = false)

// Disable parallel execution because of spark-testing-base
Test / parallelExecution := false

// Configure the build to publish the assembly JAR
(Compile / assembly / artifact) := {
  val art = (Compile / assembly / artifact).value
  art.withClassifier(Some("assembly"))
}

addArtifact(Compile / assembly / artifact, assembly)
