import java.io.{File, FileOutputStream, PrintWriter}
import java.text.SimpleDateFormat
import java.util.{Date, UUID}

import com.fasterxml.jackson.databind.ObjectMapper
import geotrellis.layer._
import geotrellis.proj4.CRS
import geotrellis.raster._

import geotrellis.raster.io.geotiff.GeoTiff
import geotrellis.raster.render.{ColorMap, ColorRamp, Exact, GreaterThanOrEqualTo}
import geotrellis.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import whu.edu.cn.application.spetralindices._
import whu.edu.cn.core.entity.{QueryParams, RasterTileLayerMetadata, SpaceTimeBandKey}
import whu.edu.cn.core.raster.query.DistributedQueryRasterTiles
import whu.edu.cn.util.{GcConstant, RuntimeData, TileUtil}
import whu.edu.cn.view.Info

import scala.collection.mutable.ArrayBuffer
import scala.sys.process._
import scala.util.control.Breaks._

import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import scala.math.Ordering.Implicits._



/**
 * Generate Water Observations from Space (WOfS) product,
 * which is acquired by detecting the presence of water
 * during a time period, and summarizing the results as
 * the number of times that water is detected at each pixel.
 */
object WOfSSampleGeneration {
  /**
   * Used in Jupyter Notebook.
   *
   * Using RDD[(SpaceTimeBandKey,Tile)] as input.
   *
   * @param tileLayerRddWithMeta a rdd of raster tiles
   *
   * @return result info
   */
  def wofs(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey])): Array[Info] = {
    println("Task is running ...")
    val tranTileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerRddWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerRddWithMeta._2)

    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tranTileLayerRddWithMeta._1
    val srcMetadata = tranTileLayerRddWithMeta._2.tileLayerMetadata
    val results = new ArrayBuffer[Info]()

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, Tile)] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))
        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiTile(greenBandTile.get, nirBandTile.get, 0.01)
        (spaceTimeKey, ndwi)
      }

    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,Tile)])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey).cache()

    //calculate water presence frequency
    val wofsRdd: RDD[(SpatialKey, Tile)] = spatialGroupRdd
      .map { x =>
        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = ArrayTile(Array.fill(cols * rows)(Double.NaN), cols, rows)
        for(i <- 0 until cols; j <- 0 until rows){
          var waterCount:Double = 0.0
          var nanCount: Int = 0
          for(k <- list){
            val value = k._2.getDouble(i, j)
            if(value.equals(Double.NaN)) nanCount += 1
            else waterCount += value / 255.0
          }
          wofsTile.setDouble(i, j, waterCount/(timeLength-nanCount))
        }
        (x._1, wofsTile)
      }

    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      DoubleCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val wofsTileLayerRdd: TileLayerRDD[SpatialKey] = ContextRDD(wofsRdd, changedSpatialMetadata)
    val stitched = wofsTileLayerRdd.stitch()

    val colorRamp = ColorRamp(
      0xF7DA22AA,
      0xECBE1DAA,
      0xE77124AA,
      0xD54927AA,
      0xCF3A27AA,
      0xA33936AA,
      0x7F182AAA,
      0x68101AAA
    )
    val uuid = UUID.randomUUID
    stitched.tile.renderPng(colorRamp).write(GcConstant.localHtmlRoot + uuid + "_wofs.png")

    val outputTiffPath = GcConstant.localHtmlRoot + uuid + "_wofs.TIF"
    GeoTiff(stitched, srcMetadata.crs).write(outputTiffPath)
    val outputThematicPngPath = GcConstant.localHtmlRoot + uuid + "_wofs_thematic.png"
    val stdout = new StringBuilder
    val stderr = new StringBuilder
    Seq("/home/geocube/qgis/run.sh", "-t", "Wofs", "-r", s"$outputTiffPath", "-o", s"$outputThematicPngPath") ! ProcessLogger(stdout append _, stderr append _)
    results += Info(outputThematicPngPath, 0, "Water Observations from Space")
    results.toArray
  }

  /**
   * Used in Jupyter Notebook.
   *
   * Using Array[(SpaceTimeBandKey,Tile)] as input.
   *
   * @param sc a SparkContext
   * @param tileLayerArrayWithMeta an array of raster tiles
   *
   * @return result info
   */
  def wofs(implicit sc:SparkContext, tileLayerArrayWithMeta:(Array[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey])): Array[Info] = {
    println(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date) + " --- WOfS task is submitted")
    println(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date) + " --- WOfS task is running ...")
    val analysisBegin = System.currentTimeMillis()

    val tileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (sc.parallelize(tileLayerArrayWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2)))), tileLayerArrayWithMeta._2)

    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tileLayerRddWithMeta._1
    val srcMetadata = tileLayerRddWithMeta._2.tileLayerMetadata
    val results = new ArrayBuffer[Info]()

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, Tile)] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))
        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiTile(greenBandTile.get, nirBandTile.get, 0.01)
        (spaceTimeKey, ndwi)
      }

    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,Tile)])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey).cache()

    //calculate water presence frequency
    val wofsRdd: RDD[(SpatialKey, Tile)] = spatialGroupRdd //RDD[(SpatialKey, Iterable((SpaceTimeKey, Tile)))]
      .map { x =>
        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = ArrayTile(Array.fill(cols * rows)(Double.NaN), cols, rows)
        val rd = new RuntimeData(0, 16, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val value = k._2.getDouble(i, j)
                  if(value.equals(Double.NaN)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength-nanCount))
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
        (x._1, wofsTile)
      }

    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      DoubleCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val wofsTileLayerRdd: TileLayerRDD[SpatialKey] = ContextRDD(wofsRdd, changedSpatialMetadata)
    val stitched = wofsTileLayerRdd.stitch()

    val colorRamp = ColorRamp(
      0xF7DA22AA,
      0xECBE1DAA,
      0xE77124AA,
      0xD54927AA,
      0xCF3A27AA,
      0xA33936AA,
      0x7F182AAA,
      0x68101AAA
    )
    val uuid = UUID.randomUUID
    stitched.tile.renderPng(colorRamp).write(GcConstant.localHtmlRoot + uuid + "_wofs.png")

    val outputTiffPath = GcConstant.localHtmlRoot + uuid + "_wofs.TIF"
    GeoTiff(stitched, srcMetadata.crs).write(outputTiffPath)
    val outputThematicPngPath = GcConstant.localHtmlRoot + uuid + "_wofs_thematic.png"
    val stdout = new StringBuilder
    val stderr = new StringBuilder
    Seq("/home/geocube/qgis/run.sh", "-t", "Wofs", "-r", s"$outputTiffPath", "-o", s"$outputThematicPngPath") ! ProcessLogger(stdout append _, stderr append _)

    results += Info(outputThematicPngPath, 0, "Water Observations from Space")
    val ret = results.toArray
    val analysisEnd = System.currentTimeMillis()
    println(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date) + " --- Time cost: " + (analysisEnd - analysisBegin) + " ms")
    ret
  }

  /**
   * Used in web service and web platform.
   *
   * Using RDD[(SpaceTimeBandKey,Tile)] as input.
   *
   * @param tileLayerRddWithMeta a rdd of tiles
   * @param outputDir
   *
   * @return
   */
  def wofs(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),
           outputDir: String): Unit = {
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
    val NDWIRdd: RDD[(SpaceTimeKey, Tile)] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        System.out.println("spaceTimeKey.time:"+spaceTimeKey.time)
        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))
        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)
        (spaceTimeKey, ndwi)
      }

    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,Tile)])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency
    val wofsRdd: RDD[(SpatialKey, Tile)] = spatialGroupRdd
      .map { x =>
        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 16, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val value = k._2.get(i, j)
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
        (x._1, wofsTile)
      }

    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(wofsRdd, changedSpatialMetadata)
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


  /**
   * Used in web service and web platform.
   *
   * Using RDD[(SpaceTimeBandKey,Tile)] as input.
   *
   * generate the sample.
   *
   * @param tileLayerRddWithMeta
   * @param outputDir
   */
  def wofsSampleGeneration(tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),
           outputDir: String): Unit = {
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
    // Creating an RDD to process data and compute the Water Observed Fraction (WOFS)
    val wofsRdd: RDD[(SpatialKey, (Tile, Int, Int, Double))] = spatialGroupRdd
      .map { x =>
        val spatialKey = x._1  // Extract the spatial key (e.g., location)

        val startTime = System.currentTimeMillis()  // Record start time for performance measurement

        val list = x._2.toList  // Convert the data associated with this key to a list
        val timeLength = list.length  // Determine the number of time points (length of the list)
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)  // Get number of columns and rows in the tiles
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)  // Initialize an empty tile for the result
        val rd = new RuntimeData(0, 16, 0)  // Create an instance of runtime data with a predefined configuration
        val assignedCellNum = cols / rd.defThreadCount  // Number of cells each thread will process
        val flag = Array.fill(rd.defThreadCount)(0)  // Flag to monitor if each thread has finished

        // Initialize counters for valid and invalid pixels globally
        val globalValidPixelCount = Array.fill(rd.defThreadCount)(0)
        val globalInvalidPixelCount = Array.fill(rd.defThreadCount)(0)

        // Loop over threads for parallel processing
        for (threadId <- 0 until rd.defThreadCount) {
          new Thread() {
            override def run(): Unit = {
              var localValidPixelCount = 0  // Count valid pixels for this thread
              var localInvalidPixelCount = 0  // Count invalid pixels for this thread
              for (i <- (threadId * assignedCellNum) until (threadId + 1) * assignedCellNum; j <- 0 until rows) {
                var waterCount: Double = 0.0
                var nanCount: Int = 0
                // Loop through each time step (tile) and process the data
                for (k <- list) {
                  val tile = k._2._1  // Access the Tile object
                  val value = tile.get(i, j)  // Get the value of the tile at the specific cell (i, j)
                  if (isNoData(value)) {
                    nanCount += 1  // Increment if the value is NoData
                    localInvalidPixelCount += 1  // Count as invalid pixel
                  } else {
                    waterCount += value / 255.0  // Normalize the value and accumulate the water count
                    localValidPixelCount += 1  // Count as valid pixel
                  }
                }
                // Set the computed water fraction for this cell
                wofsTile.setDouble(i, j, waterCount / (timeLength - nanCount))
              }
              // Update the global pixel counters for this thread
              globalValidPixelCount(threadId) = localValidPixelCount
              globalInvalidPixelCount(threadId) = localInvalidPixelCount
              flag(threadId) = 1  // Mark this thread as finished
            }
          }.start()
        }

        // Wait for all threads to complete
        while (flag.contains(0)) {
          try {
            Thread.sleep(1)  // Sleep for 1ms to avoid busy-waiting
          } catch {
            case ex: InterruptedException =>
              ex.printStackTrace()  // Handle interrupted exception
              System.err.println("exception===>: ...")
          }
        }

        val endTime = System.currentTimeMillis()  // Record end time
        val wofsTimeCost = (endTime - startTime) / 1000.0  // Calculate time cost in seconds

        // Sum up the total number of valid and invalid pixels across all threads
        val totalValidPixelCount = globalValidPixelCount.sum
        val totalInvalidPixelCount = globalInvalidPixelCount.sum

        // Return a tuple with the processed WOFS tile, time cost, and pixel counts
        (spatialKey, (wofsTile, totalValidPixelCount, totalInvalidPixelCount, wofsTimeCost))
      }

    implicit val zonedDateTimeOrdering: Ordering[ZonedDateTime] = Ordering.by(_.toInstant)
    // Extract start and end times from the RDD keys
    val times = NDWIRdd.keys.map(_.temporalKey.time).collect()
    val startTime = times.min  // Get the earliest timestamp
    val endTime = times.max    // Get the latest timestamp

    // Define a date formatter for time representation
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")

    // Format the start and end times into strings
    val formattedStartTime = startTime.format(formatter)
    val formattedEndTime = endTime.format(formatter)

    // Dynamically generate file names based on the time range
    val ndwiResultsPath = s"${outputDir}sample/ndwi_results_${formattedStartTime}_to_${formattedEndTime}.txt"
    val wofsResultsPath = s"${outputDir}sample/wofs_results_${formattedStartTime}_to_${formattedEndTime}.txt"

    // Save NDWI results and times to a file
    val ndwiResultsWriter = new PrintWriter(new File(ndwiResultsPath))

    // Iterate over the NDWI RDD and write formatted data to the file
    NDWIRdd.collect().foreach { case (spaceTimeKey, (tile, timeCost)) =>
      val (col, row, timestamp) = (spaceTimeKey.col, spaceTimeKey.row, spaceTimeKey.temporalKey.time)

      // Replace ":" in timestamp with "_" to make it suitable for file names
      val sanitizedTime = timestamp.toString.replace(":", "_")

      // Write SpaceTimeKey (column, row, time) and processing time to the file
      ndwiResultsWriter.println(s"SpaceTimeKey($col,$row,$sanitizedTime) $timeCost")
    }

    ndwiResultsWriter.close()

    // Save WOFS results and times
    val wofsResultsWriter = new PrintWriter(new File(wofsResultsPath))
    wofsRdd.collect().foreach {
      case (spatialKey, (tile,  validPixelCount, invalidPixelCount,timeCost)) =>
        wofsResultsWriter.println(s"${spatialKey} ${validPixelCount} ${invalidPixelCount} ${timeCost}")
    }
    wofsResultsWriter.close()

    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      FloatCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val tilesRdd: RDD[(SpatialKey, Tile)] = wofsRdd.map { case (spatialKey, (tile, _, _, _)) => (spatialKey, tile) }
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

  /**
   * Used in web service and web platform.
   *
   * Using Array[(SpaceTimeBandKey,Tile)] as input.
   *
   * @param sc a SparkContext
   * @param tileLayerArrayWithMeta Query tiles
   * @param outputDir
   *
   * @return
   */
  def wofs(implicit sc: SparkContext,
           tileLayerArrayWithMeta:(Array[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey]),
           outputDir: String): Unit = {
    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for(i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")

    val tileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (sc.parallelize(tileLayerArrayWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2)))), tileLayerArrayWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tileLayerRddWithMeta._1
    val srcMetadata = tileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, Tile)] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))
        if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiTile(greenBandTile.get, nirBandTile.get, 0.01)
        (spaceTimeKey, ndwi)
      }

    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,Tile)])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey).cache()

    val timeDimension = spatialGroupRdd.map( ele => ele._2.toList.length).reduce((x, y) => Math.max(x, y))

    //calculate water presence frequency
    val wofsRdd: RDD[(SpatialKey, Tile)] = spatialGroupRdd
      .map { x =>
        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = ArrayTile(Array.fill(cols * rows)(Double.NaN), cols, rows)
        val rd = new RuntimeData(0, 16, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val value = k._2.getDouble(i, j)
                  if(value.equals(Double.NaN)) nanCount += 1
                  else waterCount += value / 255.0
                }
                wofsTile.setDouble(i, j, waterCount/(timeLength-nanCount))
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
        (x._1, wofsTile)
      }

    //metadata
    val spatialKeyBounds = srcMetadata.bounds.get.toSpatial
    val changedSpatialMetadata = TileLayerMetadata(
      DoubleCellType,
      srcMetadata.layout,
      srcMetadata.extent,
      srcMetadata.crs,
      spatialKeyBounds)

    //stitch and generate thematic product
    val resultsRdd: TileLayerRDD[SpatialKey] = ContextRDD(wofsRdd, changedSpatialMetadata)
    val stitched = resultsRdd.stitch()
    val extentRet = stitched.extent
    println("<--------wofs: " + extentRet + "-------->")

    val colorArray = Array(
      0x68101AFF,
      0x7F182AFF,
      0xA33936FF,
      0xCF3A27FF,
      0xD54927FF,
      0xE77124FF,
      0xECBE1DFF,
      0xF7DA22FF,
      0xF6EDB1FF,
      0xFFFFFFFF)

    val map: Map[Double, Int] =
      if (timeDimension < 10) {
        val pixelValue = Array.fill(timeDimension + 1)(0.0)
        (0 until pixelValue.length).map { i =>
          pixelValue(i) = (pixelValue(i) + i) / timeDimension.toDouble
          (pixelValue(i), colorArray(pixelValue.length - i - 1))
        }.toMap
      }else{
        Map(
          0.0 -> 0xFFFFFFFF,
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
      }

    val colorMap =
      ColorMap(
        map,
        ColorMap.Options(
          classBoundaryType = Exact,
          noDataColor = 0x00000000, // transparent
          fallbackColor = 0x00000000, // transparent
          strict = false
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

  /**
   * Generate WOfS product with multiple spatial batches.
   *
   * Used in web service and web platform.
   *
   *
   * @param batches num of batches
   * @param queryParams query parameters
   * @param outputDir
   */
  def wofsSpatialBatch(batches: Int,
                       queryParams: QueryParams,
                       outputDir: String): Unit = {
    var queryTimeCost: Long = 0
    var analysisTimeCost: Long = 0

    val outputDirArray = outputDir.split("/")
    val sessionDir = new StringBuffer()
    for (i <- 0 until outputDirArray.length - 1)
      sessionDir.append(outputDirArray(i) + "/")

    val gridCodes: Array[String] = queryParams.getGridCodes.toArray
    println(gridCodes.length)
    val batchInterval = gridCodes.length / batches

    var queryTilesCount: Long = 0
    val spatialTiles = ArrayBuffer[(SpatialKey, Tile)]()
    if(batchInterval == gridCodes.length) {
      throw new RuntimeException("Grid codes length is " + gridCodes.length + ", batches is " + batches + ", batch interval is " + batchInterval + ", which is too large, please increase the batches!")
    } else if (batchInterval <= 1){
      throw new RuntimeException("Grid codes length is " + gridCodes.length + ", batches is " + batches + ", batch interval is " + batchInterval + ", which is too small, please reduce the batches!")
    } else{
      (0 until batches).foreach { i =>
        breakable{
          val conf = new SparkConf()
            .setAppName("WOfS analysis")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryo.registrator", "geotrellis.spark.store.kryo.KryoRegistrator")
          val sc = new SparkContext(conf)

          val batchGridCodes: ArrayBuffer[String] = new ArrayBuffer[String]()
          ((i * batchInterval) until (i + 1) * batchInterval).foreach(j => batchGridCodes.append(gridCodes(j)))
          if(i == (batches - 1) && gridCodes.length % batches != 0){
            ((i + 1) * batchInterval until gridCodes.length).foreach(j => batchGridCodes.append(gridCodes(j)))
          }
          queryParams.setGridCodes(batchGridCodes)

          //query batch tiles
          val batchQueryBegin = System.currentTimeMillis()
          val tileLayerRddWithMeta:(RDD[(SpaceTimeBandKey, Tile)],RasterTileLayerMetadata[SpaceTimeKey]) =
            DistributedQueryRasterTiles.getRasterTileRDD(sc, queryParams)
          if(tileLayerRddWithMeta == null) {
            sc.stop()
            break()
          }
          queryTilesCount += tileLayerRddWithMeta._1.count()
          val batchQueryEnd = System.currentTimeMillis()
          queryTimeCost += (batchQueryEnd - batchQueryBegin)

          //analyze batch tiles
          val batchAnalysisBegin = System.currentTimeMillis()
          val batchWOfS:Array[(SpatialKey, Tile)] = wofsSpatialBatch(tileLayerRddWithMeta)
          spatialTiles.appendAll(batchWOfS)
          val batchAnalysisEnd = System.currentTimeMillis()
          analysisTimeCost += (batchAnalysisEnd - batchAnalysisBegin)

          print("########### Query grid codes length:" + gridCodes.length + ", grid codes of spatial batch " + i + ":")
          batchGridCodes.foreach(ele => print(ele + " ")); println(" ###########")

          sc.stop()
        }
      }
    }

    val stitchBegin = System.currentTimeMillis()

    val extent = geotrellis.vector.Extent(-180, -90, 180, 90)
    val tileSize: Int = spatialTiles(0)._2.rows
    val tl = TileLayout(360, 180, tileSize, tileSize)
    val ld = LayoutDefinition(extent, tl)
    val stitched = TileUtil.stitch(spatialTiles.toArray, ld)
    val extentRet = stitched.extent
    println("<--------wofs: " + extentRet + "-------->")

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

    val stitchEnd = System.currentTimeMillis()
    analysisTimeCost += (stitchEnd - stitchBegin)

    if(batchInterval == gridCodes.length) {
      throw new RuntimeException("Batch interval is " + batchInterval + ", which is too large, please increase the batches!")
    } else if (batchInterval <= 1){
      throw new RuntimeException("Batch interval is " + batchInterval + ", which is too small, please reduce the batches!")
    } else {
      (0 until batches).foreach { i =>
        val batchGridCodes: ArrayBuffer[String] = new ArrayBuffer[String]()
        ((i * batchInterval) until (i + 1) * batchInterval).foreach(j => batchGridCodes.append(gridCodes(j)))
        if(i == (batches - 1) && gridCodes.length % batches != 0){
          ((i + 1) * batchInterval until gridCodes.length).foreach(j => batchGridCodes.append(gridCodes(j)))
        }
        print("########### Query grid codes length:" + gridCodes.length + ", grid codes of spatial batch " + i + ":")
        batchGridCodes.foreach(ele => print(ele + " ")); println(" ###########")
      }
    }

    println("Query time of " + queryTilesCount + " raster tiles: " + queryTimeCost + "ms")
    println("Analysis time: " + analysisTimeCost + "ms")
  }

  /**
   * Generate WOfS product with multiple spatial batches.
   *
   * Used in web service and web platform.
   *
   * @param tileLayerArrayWithMeta
   * @return
   */
  def wofsSpatialBatch(tileLayerArrayWithMeta:(RDD[(SpaceTimeBandKey,Tile)],RasterTileLayerMetadata[SpaceTimeKey])):Array[(SpatialKey, Tile)] = {
    val tileLayerRddWithMeta:(RDD[(SpaceTimeKey, (String, Tile))], RasterTileLayerMetadata[SpaceTimeKey]) =
      (tileLayerArrayWithMeta._1.map(x=>(x._1.spaceTimeKey, (x._1.measurementName, x._2))), tileLayerArrayWithMeta._2)
    val spatialTemporalBandRdd:RDD[(SpaceTimeKey, (String, Tile))] = tileLayerRddWithMeta._1
    val srcMetadata = tileLayerRddWithMeta._2.tileLayerMetadata

    //group by SpaceTimeKey to get a band-series RDD, i.e., RDD[(SpaceTimeKey, Iterable((bandname, Tile)))],
    //and generate ndwi tile
    val NDWIRdd: RDD[(SpaceTimeKey, Tile)] = spatialTemporalBandRdd
      .groupByKey() //group by SpaceTimeKey to get a band-series RDD, i.e. RDD[(SpaceTimeKey, Iterable((band, Tile)))]
      .map { x => //generate ndwi tile
        val spaceTimeKey = x._1
        val bandTileMap = x._2.toMap
        val (greenBandTile, nirBandTile) = (bandTileMap.get("Green"), bandTileMap.get("Near-Infrared"))
        /*if(greenBandTile == None || nirBandTile == None)
          throw new RuntimeException("There is no Green band or Nir band")
        val ndwi:Tile = NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)*/
        val ndwi: Tile = if(greenBandTile == None || nirBandTile == None) null else NDWI.ndwiShortTile(greenBandTile.get, nirBandTile.get, 0.01)
        (spaceTimeKey, ndwi)
      }.filter(x => x._2 != null)

    //group by SpatialKey to get a time-series RDD, i.e. RDD[(SpatialKey, Iterable[(SpaceTimeKey,Tile)])]
    val spatialGroupRdd = NDWIRdd.groupBy(_._1.spatialKey)

    //calculate water presence frequency using hybrid parallel technology
    val wofsRdd: RDD[(SpatialKey, Tile)] = spatialGroupRdd
      .map { x =>
        val list = x._2.toList
        val timeLength = list.length
        val (cols, rows) = (srcMetadata.tileCols, srcMetadata.tileRows)
        val wofsTile = FloatArrayTile(Array.fill(cols * rows)(0.0f), cols, rows)
        val rd = new RuntimeData(0, 16, 0)
        val assignedCellNum = cols / rd.defThreadCount
        val flag = Array.fill(rd.defThreadCount)(0)
        for(threadId <- 0 until rd.defThreadCount){
          new Thread(){
            override def run(): Unit = {
              for(i <- (threadId * assignedCellNum) until (threadId+1) * assignedCellNum; j <- 0 until rows){
                var waterCount:Double = 0.0
                var nanCount: Int = 0
                for(k <- list){
                  val value = k._2.get(i, j)
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
        (x._1, wofsTile)
      }
    wofsRdd.collect()
  }

  def main(args:Array[String]):Unit = {

    /**
     * API with args
     */
    val conf = new SparkConf()
      .setAppName("WOfS analysis")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "geotrellis.spark.store.kryo.KryoRegistrator")
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

    //wofs SampleGeneration
    val analysisBegin = System.currentTimeMillis()
    wofsSampleGeneration(tileLayerRddWithMeta, outputPath)
    val analysisEnd = System.currentTimeMillis()

    println("Query time: " + (queryEnd - queryBegin))
    println("Analysis time : " + (analysisEnd - analysisBegin))
  }
}
