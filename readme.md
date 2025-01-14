# A Graph Convolutional Neural Network-based Method for Predicting Computational Intensity of Geocomputation
This is the implementation for the paper "A Graph Convolutional Neural Network-based Method for Predicting Computational Intensity of Geocomputation".

The framework is Learning-based Computing Framework for Geospatial data(LCF-G).

This paper includes three case studies, each corresponding to a folder. Each folder contains four subfolders: data, CIPrediction, ParallelComputation and SampleGeneration.

- The **data** folder contains geospatail data.
- The **CIPrediction** folder contains model training code.
- The **ParallelComputation** folder contains geographic computation code.
- The **SampleGeneration** folder contains code for sample generation.



# Case1: Generation of DEM from point cloud data



## step 1: Data download

Dataset 1 has been uploaded to the directory `1point2dem/data`. The other two datasets, **Dataset 2** and **Dataset 3**, can be downloaded from the following website:

[OpenTopography](https://opentopography.org/)

Below are the steps for downloading **Dataset 2** and **Dataset 3**, along with the query parameters:

- Dataset 2:

1. **Visit OpenTopography Website**:
    Go to Dataset 2 Download Link.https://portal.opentopography.org/lidarDataset?opentopoID=OTLAS.112018.2193.1

2. **Coordinates & Classification**:

   In the section "1. Coordinates & Classification", select the option **"Manually enter selection coordinates"**.

3. **Set the coordinates as follows**:

   - **Xmin** = 1372495.692761
   - **Ymin** = 5076006.86821
   - **Xmax** = 1378779.529766
   - **Ymax** = 5085586.39531

3. **Point Cloud Data Download**:
   Under section "2. Point Cloud Data Download", choose the option **"Point cloud data in LAS format"**.

4. **Submit**:
   Click on **"SUBMIT"** to initiate the download.

- Dataset 3:
1. **Visit OpenTopography Website**:

  Go to Dataset 3 Download Link: https://portal.opentopography.org/lidarDataset?opentopoID=OTLAS.052016.26912.1

2. **Coordinates & Classification**:
    In the section "1. Coordinates & Classification", select the option **"Manually enter selection coordinates"**.
3. **Set the coordinates as follows**:

  - **Xmin** = 470047.153826

  - **Ymin** = 4963418.512121

  - **Xmax** = 479547.16556

  - **Ymax** = 4972078.92768

4. **Point Cloud Data Download**:
   Under section "2. Point Cloud Data Download", choose the option **"Point cloud data in LAS format"**.

5. **Submit**:
   Click on **"SUBMIT"** to initiate the download.

## step 2: Sample generation

This step involves data preparation, and samples can be generated using the provided code. Since the samples have already been uploaded to `1point2dem/SampleGeneration/data`, this step is optional.

```bash
cd 1point2dem/SampleGeneration
g++ PointCloud2DEMSampleGeneration.cpp -o PointCloud2DEMSampleGeneration
mpiexec -n {number_processes} ./PointCloud2DEMSampleGeneration ../data/pcd path/to/output
```

## step 3: Model training

This step involves training three models (GAN, CNN, GAT). The model results are saved in `1point2dem/SampleGeneration/result`, and the results for **Table 3** in the paper are derived from this output.

```bash
cd 1point2dem/CIPrediction
python -u point_prediction.py --model [GCN|ChebNet|GATNet]
```

## step 4: Parallel computation

This step uses the trained models to optimize parallel computation. The results for **Figures 11-13** in the paper are generated from the output of this command.

```bash
cd 1point2dem/ParallelComputation
g++ ParallelPointCloud2DEM.cpp -o ParallelPointCloud2DEM
mpiexec -n {number_processes} ./ParallelPointCloud2DEM ../data/pcd
```



# Case 2: Spatial intersection of vector data



## step 1: Data download

Some data from the paper has been uploaded to `2intersection/data`. The remaining OSM data can be downloaded from GeoFabrik. Below are the download steps and parameters:

- Directly click the following link to download the OSM data: [GeoFabrik - Czech Republic OSM Data](https://download.geofabrik.de/europe/czech-republic-latest-free.shp.zip)

## step 2: Sample generation

This step involves data preparation, and samples can be generated using the provided code. Since the samples have already been uploaded to `2intersection/SampleGeneration/data`, this step is optional.

```bash
cd 2intersection/SampleGeneration
g++ ParallelIntersection.cpp -o ParallelIntersection
mpiexec -n {number_processes} ./ParallelIntersection ../data/shpfile ../data/shpfile
```

## step 3: Model training

This step involves training three models (GAN, CNN, GAT). The model results are saved in `2intersection/SampleGeneration/result`, and the results for **Table 5** in the paper are derived from this output.

```bash
cd 2intersection/CIPrediction
python -u vector_prediction.py --model [GCN|ChebNet|GATNet]
```

## step 4: Parallel computation

This step uses the trained models to optimize parallel computation. The results for **Figures 14-16** in the paper are generated from the output of this command.

```
cd 2intersection/ParallelComputation
g++ ParallelIntersection.cpp -o ParallelIntersection
mpiexec -n {number_processes} ./ParallelIntersection ../data/shpfile1 ../data/shpfile2
```

# Case 3: WOfS analysis using raster data



## step 1: Data download

Some data from the paper has been uploaded to `3wofs/data`. The remaining data can be downloaded from http://openge.org.cn/advancedRetrieval?type=dataset. Below are the query parameters:

- **Product Selection**: Select `LC08_L1TP` and `LC08_L1GT`
- **Latitude and Longitude Selection**:
  - Minimum Longitude: 112.5
  - Maximum Longitude: 115.5
  - Minimum Latitude: 29.5
  - Maximum Latitude: 31.5
- **Time Range**: 2013-01-01 to 2018-12-31
- Other parameters: Default

## step 2: Sample generation

This step involves data preparation, and samples can be generated using the provided code. Since the samples have already been uploaded to `3wofs/SampleGeneration/data`, this step is optional.

```bash
cd 3wofs/SampleGeneration
sbt packeage
spark-submit --master {host1,host2,host3} --class whu.edu.cn.core.cube.raster.WOfSSampleGeneration path/to/package.jar
```

## step 3: Model training

This step involves training three models (GAN, CNN, GAT). The model results are saved in `3wofs/SampleGeneration/result`, and the results for **Table 6** in the paper are derived from this output.

```bash
cd 3wofs/CIPrediction
python -u raster_prediction.py --model [GCN|ChebNet|GATNet]
```

## step 4: Parallel computation

This step uses the trained models to optimize parallel computation. The results for **Figures 18, 19** in the paper are generated from the output of this command.

```bash
cd 3wofs/ParallelComputation
sbt packeage
spark-submit --master {host1,host2,host3} --class whu.edu.cn.core.cube.raster.WOfSOptimizedByDL path/to/package.jar path/to/output
```

## Statement about case 3

**The experiment Case 3 presented in this paper was conducted with improvements made on the GeoCube platform.**

- **Code Name**: GeoCube
- **Code Link**: [GeoCube Source Code](https://doi.org/10.6084/m9.figshare.15032847.v1)
- **License Information**: The GeoCube project is openly available under the **CC BY 4.0 license**.
  The GeoCube project is licensed under **CC BY 4.0**, which is the **Creative Commons Attribution 4.0 International License**, allowing anyone to freely share, modify, and distribute the platform's code.
- **Citation**:
  Gao, Fan (2022). A multi-source spatio-temporal data cube for large-scale geospatial analysis. figshare. Software. https://doi.org/10.6084/m9.figshare.15032847.v1

**Clarification Statement**:
**The authors of this code are not affiliated with this manuscript**. The **innovations** and **steps** in Case 3, including data download, sample generation, and parallel computation optimization, were independently developed and are not dependent on the GeoCube’s code.



# Requirements

The codes use the following dependencies with Python 3.8
- torch==2.0.0
- torch_geometric==2.5.3
- networkx==2.6.3
- pyshp==2.3.1
- tensorrt==8.6.1
- matplotlib==3.7.2
- scipy==1.10.1
- scikit-learn==1.3.0
- geopandas==0.13.2

