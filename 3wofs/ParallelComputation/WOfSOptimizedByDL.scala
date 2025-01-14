package whu.edu.cn.core.cube.raster

import java.io.{File, FileOutputStream, PrintWriter}
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter

import com.fasterxml.jackson.databind.ObjectMapper
import geotrellis.spark._
import geotrellis.layer.{SpaceTimeKey, SpatialKey, TileLayerMetadata}
import geotrellis.proj4.CRS
import geotrellis.raster.io.geotiff.GeoTiff
import geotrellis.raster.{FloatArrayTile, FloatCellType, Raster, Tile, isNoData}
import geotrellis.raster.render.{ColorMap, GreaterThanOrEqualTo}
import geotrellis.spark.{ContextRDD, TileLayerRDD}
import geotrellis.store.index.{KeyIndex, ZCurveKeyIndexMethod}
import org.apache.spark.{Partitioner, SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.scheduler.{SparkListener, SparkListenerStageSubmitted, SparkListenerTaskEnd}
import whu.edu.cn.application.spetralindices.NDWI
import whu.edu.cn.core.cube.raster.CompuIntensityPartitioner
import whu.edu.cn.core.entity.{QueryParams, RasterTileLayerMetadata, SpaceTimeBandKey}
import whu.edu.cn.core.raster.query.DistributedQueryRasterTiles
import whu.edu.cn.util.{GcConstant, RuntimeData}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Source
import org.apache.spark.scheduler._
import geotrellis.raster.isData
import scala.collection.mutable.{ListBuffer, Set}
import scala.util.Random

object WOfSOptimizedByDL {
  /***
   * wofs optimized deeplearning method,such as gcn,cn,gat
   * @param args
   */
  def main(args:Array[String]):Unit = {
    val conf = new SparkConf()
      .setAppName("WOfS analysis")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "geotrellis.spark.store.kryo.KryoRegistrator")
      .set("spark.kryoserializer.buffer.max", "1024m")
    val sc = new SparkContext(conf)

    //query and access
    val queryBegin = System.currentTimeMillis()
    val queryParams = new QueryParams
    queryParams.setCubeId(args(4))
    queryParams.setRasterProductNames(Array("LC08_L1TP_ARD_EO"))
    val extent = args(0).split(",").map(_.toDouble)
    queryParams.setExtent(extent(0), extent(1), extent(2), extent(3))

    // get output path
    val outputPath = args(1)
    val startTime = args(2)
    val endTime = args(3)

    println("extent: " + (extent(0), extent(1), extent(2), extent(3)))
    println("time: " + (startTime, endTime))

    queryParams.setTime(startTime, endTime)
    queryParams.setMeasurements(Array("Green", "Near-Infrared"))
    queryParams.setLevel(args(5))
    val tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey, Tile)],RasterTileLayerMetadata[SpaceTimeKey]) = DistributedQueryRasterTiles.getRasterTileRDD(sc, queryParams)
    val queryEnd = System.currentTimeMillis()

    val file = Source.fromFile(args(6))
    val gridZOrderCodeWithCI: Array[(BigInt, Double)] = file.getLines().map{ line =>
      val arr = line.split(" ")
      val pattern = """SpatialKey\((\d+),\s*(\d+)\)""".r
      val spatialKey = arr(0) match {
        case pattern(col, row) => SpatialKey(col.toInt, row.toInt)
        case _ => throw new IllegalArgumentException("Invalid SpatialKey format")
      }
      val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)

      val zorderCode = BigInt(keyIndex.toIndex(spatialKey).toInt)
      val compuIntesity = arr(1).toDouble
      (zorderCode, compuIntesity)
    }.toArray
    file.close
    gridZOrderCodeWithCI.foreach { case (zorderCode, compuIntensity) =>
      println(s"CI ZOrder Code: $zorderCode, Computed Intensity: $compuIntensity")
    }
    //wofs
    val numPartations=args(7).toInt
    val analysisBegin = System.currentTimeMillis()
    wofs(tileLayerRddWithMeta,outputPath,numPartations)
    val analysisEnd = System.currentTimeMillis()



    val computeCIRBegin = System.currentTimeMillis()
    val gridZOrderCodeWithCIR=computeCIR(tileLayerRddWithMeta)
    gridZOrderCodeWithCIR.foreach { case (zorderCode, compuIntensity) =>
      println(s"CIR ZOrder Code: $zorderCode, Computed Intensity: $compuIntensity")
    }
    val computeCIREnd = System.currentTimeMillis()
    val analysisOptimizedByCIRBegin = System.currentTimeMillis()
    wofsOptimizedByCIR(tileLayerRddWithMeta,gridZOrderCodeWithCIR,outputPath,numPartations,sc)
    val analysisOptimizedByCIREnd = System.currentTimeMillis()


    val computeEABegin = System.currentTimeMillis()
    val gridZOrderCodeWithEA=computeEA(tileLayerRddWithMeta)
    gridZOrderCodeWithEA.foreach { case (zorderCode, compuIntensity) =>
      println(s"EA ZOrder Code: $zorderCode, Computed Intensity: $compuIntensity")
    }
    val computeEAEnd = System.currentTimeMillis()
    val analysisOptimizedByEqualAreaBegin = System.currentTimeMillis()
    wofsOptimizedByEqualArea(tileLayerRddWithMeta,gridZOrderCodeWithEA,outputPath,numPartations,sc)
    val analysisOptimizedByEqualAreaEnd = System.currentTimeMillis()

    val analysisOptimizedByDLBegin = System.currentTimeMillis()
    wofsOptimizedByDL(tileLayerRddWithMeta,gridZOrderCodeWithCI,outputPath,numPartations,sc)
    val analysisOptimizedByDLEnd = System.currentTimeMillis()

    val analysisNoOptimizedBegin = System.currentTimeMillis()
    wofsNoOptimized(tileLayerRddWithMeta,outputPath,numPartations,sc)
    val analysisNoOptimizedEnd = System.currentTimeMillis()


    val computeMLBegin = System.currentTimeMillis()
    val outputFeaturePath=outputPath+"ml_feature.txt"
    computeMLFeature(tileLayerRddWithMeta,outputFeaturePath)
    val outputpredictMLPath=outputPath+"ml_predict.txt"
    val file_ml = Source.fromFile(outputpredictMLPath)
    val gridZOrderCodeWithML: Array[(BigInt, Double)] = file_ml.getLines().map { line =>
      val arr = line.split(",")

      val zorderCode = BigInt(arr(0))
      val compuIntensity = arr(1).toDouble

      (zorderCode, compuIntensity)
    }.toArray

    file_ml.close
    val aggregatedResult = gridZOrderCodeWithML
      .groupBy(_._1)
      .map { case (zorderCode, values) =>
        val totalCompuIntensity = values.map(_._2).sum
        (zorderCode, totalCompuIntensity)
      }
      .toArray
//    val gridZOrderCodeWithML=computeML(tileLayerRddWithMeta)
    aggregatedResult.foreach { case (zorderCode, compuIntensity) =>
      println(s"ML ZOrder Code: $zorderCode, Computed Intensity: $compuIntensity")
    }
    val computeMLEnd = System.currentTimeMillis()
    val analysisOptimizedByMLBegin = System.currentTimeMillis()
    wofsOptimizedByML(tileLayerRddWithMeta,aggregatedResult,outputPath,numPartations,sc)
    val analysisOptimizedByMLEnd = System.currentTimeMillis()



    println("Query time: " + (queryEnd - queryBegin))
    println("WOfS Analysis time : " + (analysisEnd - analysisBegin))

    println("WOfS OptimizedByDL Analysis time : " + (analysisOptimizedByDLEnd - analysisOptimizedByDLBegin))

    println("computeCIR  time : " + (computeCIREnd - computeCIRBegin))
    println("WOfS OptimizedByCIR Analysis time : " + (analysisOptimizedByCIREnd - analysisOptimizedByCIRBegin))

    println("computeEA  time : " + (computeEAEnd - computeEABegin))
    println("WOfS OptimizedByEqualArea Analysis time : " + (analysisOptimizedByEqualAreaEnd - analysisOptimizedByEqualAreaBegin))

    println("WOfS NoOptimized Analysis time : " + (analysisNoOptimizedEnd - analysisNoOptimizedBegin))

    println("computeML  time : " + (computeMLEnd - computeMLBegin))
    println("WOfS OptimizedByML Analysis time : " + (analysisOptimizedByMLEnd - analysisOptimizedByMLBegin))
  }
  def wofsOptimizedByDL(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),gridZOrderCodeWithCI: Array[(BigInt, Double)], outputDir: String,numPartations:Int,sc:SparkContext): Unit ={
    /***repartition using sort***/
    val partitionContainers: Array[ArrayBuffer[BigInt]] = CompuIntensityPartitioner.sortPut(gridZOrderCodeWithCI, numPartations)


    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, (Tile,Double))] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))

        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)

        val ndwiEndTime = System.currentTimeMillis()
        val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0 // Convert to seconds

        (spaceTimeKey, (ndwi, ndwiTimeCost))
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val balancedSpatialGroupRdd=spatialGroupRdd.partitionBy(new CompuIntensityPartitioner(numPartations, partitionContainers))
    val wofsRdd: RDD[(SpatialKey, (Tile,Double))] = balancedSpatialGroupRdd
      .map { x =>
//      partition.map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 32, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val tile = k._2._1
                  val value = tile.get(i, j)
                  if(isNoData(value)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength - nanCount))
              }
              flag(threadId) = 1
            }
          }.start()
        }
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)
          }catch {
            case ex: InterruptedException  => {
              ex.printStackTrace()
              System.err.println("exception===>: ...")
            }
          }
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0 // Convert to seconds

        // return wofsTile
        (spatialKey,(wofsTile,wofsTimeCost))
      }

    println(wofsRdd.toDebugString)
    val targetRDD = wofsRdd
    val rddId = targetRDD.id
    val listener = new SpecificRDDListener(rddId)
    sc.addSparkListener(listener)

    println("Starting collect on RDD")
    wofsRdd.collect()
    println("Completed collect on RDD")

    println(s"wofsRdd ID: $rddId")

    println(s"Partition execution times collected: ${listener.partitionTimes.size}")
    if (listener.partitionTimes.nonEmpty) {
      println("Partition execution times (seconds):")
      println(s"Listener partition times: ${listener.partitionTimes.toSeq}")
      listener.partitionTimes.toSeq.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq.sortBy(_._1).foreach {
        case (partitionId, totalTime) =>
          println(s"DL Partition $partitionId,$totalTime")
      }
    } else {
      println("No partition times were collected.")
    }

    sc.removeSparkListener(listener)
    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }
  def wofs(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]), outputDir: String,numPartations:Int): Unit ={

    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.repartition(1).map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1.repartition(1)


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    val NDWIRdd: RDD[(SpaceTimeKey, (Tile, Double))] = spatialTemporalBandRdd.repartition(1)
      .groupByKey()
      .map { x =>
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:" + spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTiles: Iterable[(String, Tile)] = x._2
        val greenBandTileOpt = bandTiles.find(_._1 == "Green")
        val nirBandTileOpt = bandTiles.find(_._1 == "Near-Infrared")

        (greenBandTileOpt, nirBandTileOpt) match {
          case (Some(greenBandTile), Some(nirBandTile)) =>
            val ndwi: Tile = NDWI.ndwiShortTile(greenBandTile._2, nirBandTile._2, 0.01)
            val ndwiEndTime = System.currentTimeMillis()
            val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0
            (spaceTimeKey, (ndwi, ndwiTimeCost))

          case _ =>
            throw new RuntimeException("There is no Green band or Nir band")
        }
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency

    val wofsRdd: RDD[(SpatialKey, (Tile, Double))] = spatialGroupRdd.repartition(1)
      .map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)

        for (i <- 0 until cols; j <- 0 until rows) {
          var waterCount: Double = 0.0
          var nanCount: Int = 0
          for (k <- list) {
            val tile = k._2._1
            val value = tile.get(i, j)
            if (isNoData(value)) nanCount += 1
            else waterCount += value / 255.0
          }
          wofsTile.setDouble(i, j, waterCount / (timeLength - nanCount))
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0

        (spatialKey, (wofsTile, wofsTimeCost))
      }
    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.repartition(1).map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }
  def wofsOptimizedByCIR(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),gridZOrderCodeWithCI: Array[(BigInt, Double)], outputDir: String,numPartations:Int,sc:SparkContext): Unit ={
    /***repartition using us CIR***/
    val partitionContainers: Array[ArrayBuffer[BigInt]] = CompuIntensityPartitioner.sortPut(gridZOrderCodeWithCI, numPartations)


    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, (Tile,Double))] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))

        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)

        val ndwiEndTime = System.currentTimeMillis()
        val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0 // Convert to seconds

        (spaceTimeKey, (ndwi, ndwiTimeCost))
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val balancedSpatialGroupRdd=spatialGroupRdd.partitionBy(new CompuIntensityPartitioner(numPartations, partitionContainers))
    val wofsRdd: RDD[(SpatialKey, (Tile,Double))] = balancedSpatialGroupRdd
      .map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 32, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val tile = k._2._1
                  val value = tile.get(i, j)
                  if(isNoData(value)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength - nanCount))
              }
              flag(threadId) = 1
            }
          }.start()
        }
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)
          }catch {
            case ex: InterruptedException  => {
              ex.printStackTrace()
              System.err.println("exception===>: ...")
            }
          }
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0 // Convert to seconds

        // return wofsTile
        (spatialKey,(wofsTile,wofsTimeCost))
      }

    println(wofsRdd.toDebugString)
    val targetRDD = wofsRdd
    val rddId = targetRDD.id
    val listener = new SpecificRDDListener(rddId)
    sc.addSparkListener(listener)
    println("Starting collect on RDD")
    wofsRdd.collect()
    println("Completed collect on RDD")

    println(s"wofsRdd ID: $rddId")

    println(s"Partition execution times collected: ${listener.partitionTimes.size}")
    if (listener.partitionTimes.nonEmpty) {
      println("Partition execution times (seconds):")
      println(s"Listener partition times: ${listener.partitionTimes.toSeq}")
      listener.partitionTimes.toSeq.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq.sortBy(_._1).foreach {
        case (partitionId, totalTime) =>
          println(s"CIR Partition $partitionId,$totalTime")
      }
    } else {
      println("No partition times were collected.")
    }

    sc.removeSparkListener(listener)


    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }
  def computeCIR(tileLayerRddWithMeta: (RDD[(SpaceTimeBandKey, Tile)], RasterTileLayerMetadata[SpaceTimeKey])): Array[(BigInt, Double)] = {
    val (tileLayerRdd, metadata) = tileLayerRddWithMeta

    val zorderCompuIntensity: RDD[(SpatialKey, Double)] = tileLayerRdd.flatMap { case (key, tile) =>
      val spatialKey = key._spaceTimeKey.spatialKey

      val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)

      val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)

      var nullCount = 0
      var nonNullCount = 0

      tile.foreach { cellValue =>
        if (cellValue == null) {
          nullCount += 1
        } else {
          nonNullCount += 1
        }
      }

      val compuIntensity = nullCount * 0.067 + nonNullCount * 1.0

      Seq((spatialKey, compuIntensity))
    }

    val aggregatedResults: RDD[(BigInt, Double)] = zorderCompuIntensity
      .reduceByKey(_ + _)
      .map { case (spatialKey, totalIntensity) =>
        val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)
        val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)
        (zorderCode, totalIntensity)
      }

    aggregatedResults.collect()
  }


  def computeEA(tileLayerRddWithMeta: (RDD[(SpaceTimeBandKey, Tile)], RasterTileLayerMetadata[SpaceTimeKey])): Array[(BigInt, Double)] = {
    val (tileLayerRdd, metadata) = tileLayerRddWithMeta

    // Use map to compute the zorderCode for each tile and set the computation intensity to 1
    val zorderCompuIntensity: RDD[(BigInt, Double)] = tileLayerRdd.map { case (key, tile) =>
      // Extract the SpatialKey
      val spatialKey = key._spaceTimeKey.spatialKey

      // Create a KeyIndex object using ZCurve indexing
      val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)

      // Compute the zorderCode
      val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)

      // Set the computation intensity to a fixed value of 1
      (zorderCode, 1.0) // Return zorderCode and intensity
    }

    // Use distinct to ensure each zorderCode appears only once
    val distinctZorderCompuIntensity = zorderCompuIntensity.distinct()

    // Collect the results and return them
    distinctZorderCompuIntensity.collect() // Return results as Array[(BigInt, Double)]
  }


  def computeML(tileLayerRddWithMeta: (RDD[(SpaceTimeBandKey, Tile)], RasterTileLayerMetadata[SpaceTimeKey])): Array[(BigInt, Double)] = {
    val (tileLayerRdd, metadata) = tileLayerRddWithMeta

    // Use map to calculate the intensity for each tile and return (zorderCode, compuIntensity)
    val zorderCompuIntensity: RDD[(SpatialKey, Double)] = tileLayerRdd.flatMap { case (key, tile) =>
      // Extract the SpatialKey
      val spatialKey = key._spaceTimeKey.spatialKey // Assuming SpaceTimeBandKey contains spatialKey

      // Create a KeyIndex object using ZCurve indexing
      val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)

      // Compute the zorderCode
      val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)

      // Initialize counters
      var nullCount = 0
      var nonNullCount = 0

      // Iterate over each pixel in the tile
      tile.foreach { cellValue =>
        if (cellValue == null) {
          nullCount += 1
        } else {
          nonNullCount += 1
        }
      }

      // Calculate compuIntensity
      val compuIntensity = nullCount * 0.067 + nonNullCount * 1.0

      // Return a tuple (spatialKey, compuIntensity)
      Seq((spatialKey, compuIntensity)) // Use Seq to allow flatMap to unfold the results
    }

    // Aggregate results by SpatialKey, summing up the total compuIntensity for each spatialKey
    val aggregatedResults: RDD[(BigInt, Double)] = zorderCompuIntensity
      .reduceByKey(_ + _) // Combine compuIntensity for the same spatialKey
      .map { case (spatialKey, totalIntensity) =>
        // Compute the zorderCode for each SpatialKey
        val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)
        val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)
        (zorderCode, totalIntensity) // Return (zorderCode, totalIntensity)
      }

    // Collect the results into an array
    aggregatedResults.collect() // Return the result as Array[(BigInt, Double)]
  }



  def computeMLFeature(tileLayerRddWithMeta: (RDD[(SpaceTimeBandKey, Tile)], RasterTileLayerMetadata[SpaceTimeKey]),
                       outputPath: String): Array[(BigInt, Double)] = {
    val (tileLayerRdd, metadata) = tileLayerRddWithMeta

    // Calculate the nullCount and nonNullCount for each spatialKey, and return an RDD[(SpatialKey, nullCount, nonNullCount)]
    val zorderCompuIntensity: RDD[(BigInt, Int, Int)] = tileLayerRdd.flatMap { case (key, tile) =>
      // Extract the SpatialKey
      val spatialKey = key._spaceTimeKey.spatialKey // Assuming SpaceTimeBandKey contains spatialKey

      // Create a KeyIndex object using ZCurve index
      val keyIndex: KeyIndex[SpatialKey] = ZCurveKeyIndexMethod.createIndex(null)

      // Compute the zorderCode
      val zorderCode: BigInt = BigInt(keyIndex.toIndex(spatialKey).toInt)

      // Initialize counters
      var nullCount = 0
      var nonNullCount = 0

      // Iterate over each pixel in the tile
      tile.foreach { cellValue =>
        if (isData(cellValue)) {
          nonNullCount += 1
        } else {
          nullCount += 1
        }
      }

      // Return a tuple (zorderCode, nullCount, nonNullCount)
      Seq((zorderCode, nullCount, nonNullCount))
    }

    // Collect the computed results
    val result = zorderCompuIntensity.collect()

    // Write the results to a file
    val writer = new PrintWriter(outputPath, "UTF-8")  // Create a PrintWriter to write to a file
    try {
      // Write each zorderCode, nullCount, and nonNullCount to the file
      result.foreach { case (zorderCode, nullCount, nonNullCount) =>
        writer.println(s"${zorderCode} ${nullCount} ${nonNullCount}")
      }
    } finally {
      // Ensure the PrintWriter is properly closed
      writer.close()
    }

    // Calculate compuIntensity and return the final result
    val finalResult = zorderCompuIntensity.map { case (zorderCode, nullCount, nonNullCount) =>
      val compuIntensity = nullCount * 0.067 + nonNullCount * 1.0
      (zorderCode, compuIntensity)
    }.collect()

    // Return the final computed result
    finalResult
  }


  def wofsOptimizedByML(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),gridZOrderCodeWithCI: Array[(BigInt, Double)], outputDir: String,numPartations:Int,sc:SparkContext): Unit ={
    /***repartition using ML***/
    val partitionContainers: Array[ArrayBuffer[BigInt]] = CompuIntensityPartitioner.sortPut(gridZOrderCodeWithCI, numPartations)


    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, (Tile,Double))] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))

        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)

        val ndwiEndTime = System.currentTimeMillis()
        val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0 // Convert to seconds

        (spaceTimeKey, (ndwi, ndwiTimeCost))
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val balancedSpatialGroupRdd=spatialGroupRdd.partitionBy(new CompuIntensityPartitioner(numPartations, partitionContainers))
    val wofsRdd: RDD[(SpatialKey, (Tile,Double))] = balancedSpatialGroupRdd
      .map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 32, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val tile = k._2._1
                  val value = tile.get(i, j)
                  if(isNoData(value)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength - nanCount))
              }
              flag(threadId) = 1
            }
          }.start()
        }
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)
          }catch {
            case ex: InterruptedException  => {
              ex.printStackTrace()
              System.err.println("exception===>: ...")
            }
          }
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0 // Convert to seconds

        // return wofsTile
        (spatialKey,(wofsTile,wofsTimeCost))
      }
    println(wofsRdd.toDebugString)
    val targetRDD = wofsRdd
    val rddId = targetRDD.id

    val listener = new SpecificRDDListener(rddId)
    sc.addSparkListener(listener)

    println("Starting collect on RDD")
    wofsRdd.collect()
    println("Completed collect on RDD")

    println(s"wofsRdd ID: $rddId")

    println(s"Partition execution times collected: ${listener.partitionTimes.size}")
    if (listener.partitionTimes.nonEmpty) {
      println("Partition execution times (seconds):")
      println(s"Listener partition times: ${listener.partitionTimes.toSeq}")
      listener.partitionTimes.toSeq.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq.sortBy(_._1).foreach {
        case (partitionId, totalTime) =>
          println(s"ML Partition $partitionId,$totalTime")
      }
    } else {
      println("No partition times were collected.")
    }

    sc.removeSparkListener(listener)
    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }


  def wofsOptimizedByEqualArea(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),gridZOrderCodeWithCI: Array[(BigInt, Double)], outputDir: String,numPartations:Int,sc:SparkContext): Unit ={
    /***repartition using equal area***/
    val partitionContainers: Array[ArrayBuffer[BigInt]] = CompuIntensityPartitioner.sortPut(gridZOrderCodeWithCI, numPartations)


    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, (Tile,Double))] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))

        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)

        val ndwiEndTime = System.currentTimeMillis()
        val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0 // Convert to seconds

        (spaceTimeKey, (ndwi, ndwiTimeCost))
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val balancedSpatialGroupRdd=spatialGroupRdd.partitionBy(new CompuIntensityPartitioner(numPartations, partitionContainers))
    val wofsRdd: RDD[(SpatialKey, (Tile,Double))] = balancedSpatialGroupRdd
      .map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 32, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val tile = k._2._1
                  val value = tile.get(i, j)
                  if(isNoData(value)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength - nanCount))
              }
              flag(threadId) = 1
            }
          }.start()
        }
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)
          }catch {
            case ex: InterruptedException  => {
              ex.printStackTrace()
              System.err.println("exception===>: ...")
            }
          }
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0 // Convert to seconds

        // return wofsTile
        (spatialKey,(wofsTile,wofsTimeCost))
      }
    println(wofsRdd.toDebugString)
    val targetRDD = wofsRdd
    val rddId = targetRDD.id

    val listener = new SpecificRDDListener(rddId)
    sc.addSparkListener(listener)

    println("Starting collect on RDD")
    wofsRdd.collect()
    println("Completed collect on RDD")

    println(s"wofsRdd ID: $rddId")

    println(s"Partition execution times collected: ${listener.partitionTimes.size}")
    if (listener.partitionTimes.nonEmpty) {
      println("Partition execution times (seconds):")
      println(s"Listener partition times: ${listener.partitionTimes.toSeq}")
      listener.partitionTimes.toSeq.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq.sortBy(_._1).foreach {
        case (partitionId, totalTime) =>
          println(s"EA Partition $partitionId,$totalTime")
      }
    } else {
      println("No partition times were collected.")
    }

    sc.removeSparkListener(listener)
    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }
  def wofsNoOptimized(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]), outputDir: String,numPartations:Int,sc:SparkContext): Unit ={
    /***Repartition, calculate using random allocation.***/

    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1


    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, (Tile,Double))] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)

        val ndwiStartTime = System.currentTimeMillis()

        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))

        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)

        val ndwiEndTime = System.currentTimeMillis()
        val ndwiTimeCost = (ndwiEndTime - ndwiStartTime) / 1000.0 // Convert to seconds

        (spaceTimeKey, (ndwi, ndwiTimeCost))
      }
    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,(Tile,Double))])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val balancedSpatialGroupRdd=spatialGroupRdd/*.repartition(numPartations)*/.partitionBy(new CustomPartitioner(8))
    val wofsRdd: RDD[(SpatialKey, (Tile,Double))] = balancedSpatialGroupRdd
      .map { x =>
        val spatialKey = x._1

        val startTime = System.currentTimeMillis()

        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 32, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val tile = k._2._1
                  val value = tile.get(i, j)
                  if(isNoData(value)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength - nanCount))
              }
              flag(threadId) = 1
            }
          }.start()
        }
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)
          }catch {
            case ex: InterruptedException  => {
              ex.printStackTrace()
              System.err.println("exception===>: ...")
            }
          }
        }

        val endTime = System.currentTimeMillis()
        val wofsTimeCost = (endTime - startTime) / 1000.0 // Convert to seconds

        // return wofsTile
        (spatialKey,(wofsTile,wofsTimeCost))
      }
    println(wofsRdd.toDebugString)
    val targetRDD = wofsRdd
    val rddId = targetRDD.id
    val listener = new SpecificRDDListener(rddId)
    sc.addSparkListener(listener)
    println("Starting collect on RDD")
    wofsRdd.collect()
    println("Completed collect on RDD")

    println(s"wofsRdd ID: $rddId")

    println(s"Partition execution times collected: ${listener.partitionTimes.size}")
    if (listener.partitionTimes.nonEmpty) {
      println("Partition execution times (seconds):")
      println(s"Listener partition times: ${listener.partitionTimes.toSeq}")
      listener.partitionTimes.toSeq.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq.sortBy(_._1).foreach {
        case (partitionId, totalTime) =>
          println(s"NoOp Partition $partitionId,$totalTime")
      }
    } else {
      println("No partition times were collected.")
    }
    sc.removeSparkListener(listener)
    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _)) => (spatialKey, tile) }
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(tilesRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent

    val map: Map[Double, Int] =
      Map(
        0.00 -> 0xFFFFFFFF,
        0.11 -> 0xF6EDB1FF,
        0.22 -> 0xF7DA22FF,
        0.33 -> 0xECBE1DFF,
        0.44 -> 0xE77124FF,
        0.55 -> 0xD54927FF,
        0.66 -> 0xCF3A27FF,
        0.77 -> 0xA33936FF,
        0.88 -> 0x7F182AFF,
        0.99 -> 0x68101AFF
      )

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = GreaterThanOrEqualTo,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = true
        )
      )

    val executorSessionDir = sessionDir.toString
    val executorSessionFile = new File(executorSessionDir)
    if (!executorSessionFile.exists) executorSessionFile.mkdir
    val executorOutputDir = outputDir
    val executorOutputFile = new File(executorOutputDir)
    if (!executorOutputFile.exists()) executorOutputFile.mkdir()

    val outputPath = executorOutputDir + "WOfS.png"
    stitched.tile.renderPng(colorMap).write(outputPath)

    val outputMetaPath = executorOutputDir + "WOfS.json"
    val objectMapper =new ObjectMapper()
    val node = objectMapper.createObjectNode()
    val localDataRoot = GcConstant.localDataRoot
    val httpDataRoot = GcConstant.httpDataRoot
    node.put("path", outputPath.replace(localDataRoot, httpDataRoot))
    node.put("meta", outputMetaPath.replace(localDataRoot, httpDataRoot))
    node.put("extent", extentRet.xmin + "," + extentRet.ymin + "," + extentRet.xmax + "," + extentRet.ymax)
    objectMapper.writerWithDefaultPrettyPrinter().writeValue(new FileOutputStream(outputMetaPath), node)
  }
}
class CustomPartitioner(partitions: Int) extends Partitioner {
  require(partitions > 0, " mast > 0")

  val selectedPartitions = Set(1, 2)

  override def numPartitions: Int = partitions

  override def getPartition(key: Any): Int = key match {
    case SpatialKey(_, _) =>
      if (selectedPartitions.nonEmpty) {
        val selectedPartition = selectedPartitions.toSeq(Random.nextInt(selectedPartitions.size))
        selectedPartition
      } else {
        0
      }
    case _ =>
      0
  }
}

class SpecificRDDListener(targetRDDId: Int) extends SparkListener {
  private val targetStageIds: Set[Int] = Set()
  val partitionTimes: ListBuffer[(Int, Double)] = ListBuffer()

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = {
    val stageInfo = stageSubmitted.stageInfo
    println(s"Stage ${stageInfo.stageId} contains RDDs: ${stageInfo.rddInfos.map(_.id)}")
    if (stageInfo.rddInfos.exists(_.id == targetRDDId)) {
      targetStageIds += stageInfo.stageId
      println(s"Stage ${stageInfo.stageId} contains target RDD $targetRDDId")
    }
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    println(s"Task ended for stage ${taskEnd.stageId}")
    if (taskEnd.stageId != -1 && targetStageIds.contains(taskEnd.stageId)) {
      if (taskEnd.taskMetrics != null) {
        val taskIndex = taskEnd.taskInfo.index
        val taskRunTimeInSeconds = taskEnd.taskMetrics.executorRunTime / 1000.0
        println(s"Partition $taskIndex ran for $taskRunTimeInSeconds seconds")
        partitionTimes += ((taskIndex, taskRunTimeInSeconds))
      }
    }
  }
}