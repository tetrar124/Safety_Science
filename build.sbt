//name := "Simple Project"
name := "netCDF2CSV"
version := "1.0"
scalaVersion := "2.11.8"
val sparkVersion = "2.3.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % sparkVersion
libraryDependencies += "org.apache.spark" % "spark-hive_2.11" % sparkVersion

//libraryDependencies += "edu.ucar" % "netcdf4" % "4.5.5"
//libraryDependencies += "default" % "netcdfAll" % "4.6"
libraryDependencies += "com.github.pathikrit" %% "better-files" % "3.4.0"
libraryDependencies += "org.vegas-viz" %% "vegas" % "0.3.9"
libraryDependencies += "org.vegas-viz" %% "vegas-spark" % "0.3.9"

unmanagedBase := baseDirectory.value / "custom_lib"