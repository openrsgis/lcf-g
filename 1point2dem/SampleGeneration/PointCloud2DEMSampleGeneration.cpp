#include <Python.h>
#include "stdio.h" 
#include<string.h>    
#include"mpi.h"  
#include<omp.h> 
#include <malloc.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include "pcl/kdtree/kdtree_flann.h"
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <sstream>

int range_Minx = 0;
int range_Miny = 0;
int range_Maxx = 0;
int range_Maxy = 0;
int init_depth = 4;
int global_depth = 0;
#define threads_num 6
typedef enum
{
	UR = 0,
	UL = 1,
	LL = 2,
	LR = 3
}QuadrantEnum;


typedef struct NodeRange
{
	int maxX;
	int minX;
	int maxY;
	int minY;
}NodeRange;

typedef struct QuadNode
{
	NodeRange     Box;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud;
	int     nChildCount;
	QuadNode* children[4];
	QuadNode* father;
	int depth_location;
	double compute_density;
}QuadNode;

typedef struct quadtree_t
{
	QuadNode* root;
	int         depth;
}QuadTree;

typedef struct Arr
{
	QuadNode* pBase[1200];
	int len;

	int cnt;

}Arr;

struct Arr arr;

QuadNode* InitQuadNode();


void CreateQuadTree(int depth, pcl::PointXY* pt, QuadTree* pQuadTree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


void CreateQuadBranch(int depth, NodeRange& rect, QuadNode** node, int nPointCount);

void init_arr(struct Arr* pArr, int length);

bool is_empty(struct Arr* pArr);

bool append_arr(struct Arr* pArr, int val);

void mpi_createTrainingData(int argc, char* argv[], int gridDimX, int gridDimY);

void init_arr(struct Arr* pArr, int length) {

	if (NULL == pArr->pBase)
	{
		printf("free error, dynamic memory allocation failed！\n");
		exit(-1);
	}
	else
	{
		pArr->len = length;
		pArr->cnt = 0;
	}
	return;
}
bool is_full(struct Arr* pArr)
{
	if (pArr->cnt == pArr->len) {
		printf("%s\n", "The current allocated array length is full");
		return true;
	}
	else
	{
		return false;
	}
}

bool append_arr(struct Arr* pArr, QuadNode* val)
{
	if (is_full(pArr))
		return false;
	pArr->pBase[pArr->cnt] = val;
	(pArr->cnt)++;
	return true;
}


QuadNode* InitQuadNode()
{
	QuadNode* node = new QuadNode;
	node->Box.maxX = 0;
	node->Box.maxY = 0;
	node->Box.minX = 0;
	node->Box.minY = 0;

	for (int i = 0; i < 4; i++)
	{
		node->children[i] = NULL;
	}
	node->father = NULL;
	node->nChildCount = 0;
	node->depth_location = 0;
	node->compute_density = 0;
	return node;
}

void QuadBox_init(NodeRange* box, int minX, int minY, int maxX, int maxY)
{
	box->minX = minX;
	box->maxX = maxX;
	box->minY = minY;
	box->maxY = maxY;
}
void NodeRangeSplit(NodeRange& Box, NodeRange* box)
{
	int Maxx = Box.maxX;
	int Minx = Box.minX;
	int Maxy = Box.maxY;
	int Miny = Box.minY;
	if (global_depth == 1)
	{

		int x_step = 1;
		int y_step = 1;
		for (int i = 0; i < init_depth; i++)
		{
			x_step = x_step * 2;
			y_step = y_step * 2;
		}
		x_step = (int)(98 / x_step);
		y_step = (int)(68 / y_step);
		Maxx = Minx + 2 * x_step;
		Maxy = Miny + 2 * y_step;
	}
	int dx = (int)(Maxx - Minx) / 2;
	int dy = (int)(Maxy - Miny) / 2;
	QuadBox_init(&box[3], Minx + dx, Miny + dy, Maxx, Maxy);
	QuadBox_init(&box[2], Minx, Miny + dy, Minx + dx, Maxy);
	QuadBox_init(&box[0], Minx, Miny, Minx + dx, Miny + dy);
	QuadBox_init(&box[1], Minx + dx, Miny, Maxx, Miny + dy);



}
void CreateQuadBranch(int depth, NodeRange& rect, QuadNode* father, QuadNode** node)
{
	if (depth != 0)
	{
		if (depth == init_depth)
		{
			*node = InitQuadNode(); // Create the root node
			QuadNode* pNode = *node;
			pNode->Box = rect;
			pNode->nChildCount = 4;
			pNode->father = father;
			pNode->cloud = father->cloud;
			pNode->depth_location = init_depth - depth;
			NodeRange boxs[4];
			NodeRangeSplit(pNode->Box, boxs);
			for (int i = 0; i < 4; i++)
			{
				// Create four nodes and insert the corresponding MBR (Minimum Bounding Rectangle)
				pNode->children[i] = InitQuadNode();
				pNode->children[i]->Box = boxs[i];
				pNode->children[i]->father = pNode;
				pNode->children[i]->depth_location = pNode->depth_location + 1;

				// Set the boundary values for neighbor grid filtering
				int neighbor_min_x = boxs[i].minX - 10;
				int neighbor_max_x = boxs[i].maxX + 10;
				int neighbor_min_y = boxs[i].minY - 10;
				int neighbor_max_y = boxs[i].maxY + 10;

				// Filter neighbor grid points
				pcl::PointCloud<pcl::PointXYZ>::Ptr n_x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_x_pass_;
				n_x_pass_.setInputCloud(pNode->cloud);            // Set the input point cloud
				n_x_pass_.setFilterFieldName("x");                // Set the field name (X axis) for filtering
				n_x_pass_.setFilterLimits(neighbor_min_x, neighbor_max_x);        // Set the filtering range on X axis
				n_x_pass_.filter(*n_x_cloud_filtered_);           // Apply the filter and store the result

				pcl::PointCloud<pcl::PointXYZ>::Ptr n_xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_xy_pass_;
				n_xy_pass_.setInputCloud(n_x_cloud_filtered_);    // Set the input cloud after X-axis filtering
				n_xy_pass_.setFilterFieldName("y");                // Set the field name (Y axis) for filtering
				n_xy_pass_.setFilterLimits(neighbor_min_y, neighbor_max_y);        // Set the filtering range on Y axis
				n_xy_pass_.filter(*n_xy_cloud_filtered_);          // Apply the filter and store the result
				pNode->children[i]->cloud = n_xy_cloud_filtered_; // Assign the filtered cloud to the child node

				// Filter local grid points
				pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> x_pass_;
				x_pass_.setInputCloud(n_xy_cloud_filtered_);       // Set the input cloud after XY-axis filtering
				x_pass_.setFilterFieldName("x");                   // Set the field name (X axis) for filtering
				x_pass_.setFilterLimits(boxs[i].minX, boxs[i].maxX);  // Set the filtering range on X axis for the local grid
				x_pass_.filter(*x_cloud_filtered_);                // Apply the filter and store the result

				pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> xy_pass_;
				xy_pass_.setInputCloud(x_cloud_filtered_);         // Set the input cloud after X-axis filtering
				xy_pass_.setFilterFieldName("y");                   // Set the field name (Y axis) for filtering
				xy_pass_.setFilterLimits(boxs[i].minY, boxs[i].maxY);  // Set the filtering range on Y axis for the local grid
				xy_pass_.filter(*xy_cloud_filtered_);              // Apply the filter and store the result
				pNode->children[i]->local_cloud = xy_cloud_filtered_; // Assign the filtered local cloud to the child node

				// Recursively create the next level of the quadtree branch
				CreateQuadBranch(depth - 1, boxs[i], pNode->children[i]->father, &(pNode->children[i]));
			}
		}
		else
		{
			QuadNode* pNode = *node;
			NodeRange boxs[4];
			global_depth = depth;
			NodeRangeSplit(pNode->Box, boxs);
			pNode->depth_location = init_depth - depth;

			for (int i = 0; i < 4; i++)
			{
				// Create four nodes and insert the corresponding MBR (Minimum Bounding Rectangle)
				pNode->children[i] = InitQuadNode();
				pNode->children[i]->Box = boxs[i];
				pNode->children[i]->father = pNode;
				pNode->children[i]->depth_location = pNode->depth_location + 1;

				// Set the boundary values for neighbor grid filtering
				int neighbor_min_x = boxs[i].minX - 10;
				int neighbor_max_x = boxs[i].maxX + 10;
				int neighbor_min_y = boxs[i].minY - 10;
				int neighbor_max_y = boxs[i].maxY + 10;

				// Filter neighbor grid points
				pcl::PointCloud<pcl::PointXYZ>::Ptr n_x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_x_pass_;
				n_x_pass_.setInputCloud(pNode->cloud);            // Set the input point cloud
				n_x_pass_.setFilterFieldName("x");                // Set the field name (X axis) for filtering
				n_x_pass_.setFilterLimits(neighbor_min_x, neighbor_max_x);        // Set the filtering range on X axis
				n_x_pass_.filter(*n_x_cloud_filtered_);           // Apply the filter and store the result

				pcl::PointCloud<pcl::PointXYZ>::Ptr n_xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_xy_pass_;
				n_xy_pass_.setInputCloud(n_x_cloud_filtered_);    // Set the input cloud after X-axis filtering
				n_xy_pass_.setFilterFieldName("y");                // Set the field name (Y axis) for filtering
				n_xy_pass_.setFilterLimits(neighbor_min_y, neighbor_max_y);        // Set the filtering range on Y axis
				n_xy_pass_.filter(*n_xy_cloud_filtered_);          // Apply the filter and store the result
				pNode->children[i]->cloud = n_xy_cloud_filtered_; // Assign the filtered cloud to the child node

				// Filter local grid points
				pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> x_pass_;
				x_pass_.setInputCloud(n_xy_cloud_filtered_);       // Set the input cloud after XY-axis filtering
				x_pass_.setFilterFieldName("x");                   // Set the field name (X axis) for filtering
				x_pass_.setFilterLimits(boxs[i].minX, boxs[i].maxX);  // Set the filtering range on X axis for the local grid
				x_pass_.filter(*x_cloud_filtered_);                // Apply the filter and store the result

				pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> xy_pass_;
				xy_pass_.setInputCloud(x_cloud_filtered_);         // Set the input cloud after X-axis filtering
				xy_pass_.setFilterFieldName("y");                   // Set the field name (Y axis) for filtering
				xy_pass_.setFilterLimits(boxs[i].minY, boxs[i].maxY);  // Set the filtering range on Y axis for the local grid
				xy_pass_.filter(*xy_cloud_filtered_);              // Apply the filter and store the result
				pNode->children[i]->local_cloud = xy_cloud_filtered_; // Assign the filtered local cloud to the child node

				// Recursively create the next level of the quadtree branch
				CreateQuadBranch(depth - 1, boxs[i], pNode->children[i]->father, &(pNode->children[i]));
			}
		}

	}
	else
	{
		append_arr(&arr, *node);
	}
}
void CreateQuadTree(int depth, QuadTree* pQuadTree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	//init_arr(&arr, 1200);
	pQuadTree->depth = depth;
	NodeRange rect;
	rect.minX = range_Minx;
	rect.minY = range_Miny;
	rect.maxX = range_Maxx;
	rect.maxY = range_Maxy;
	QuadNode* root_father = InitQuadNode();
	root_father->Box = rect;
	root_father->cloud = cloud;
	CreateQuadBranch(depth, rect, root_father, &(pQuadTree->root));

}

void SearchQuadTree(QuadNode* pNode)
{
	int cp_sum = 0;

	if (pNode != NULL)
	{
		std::cout << "------------" << pNode->depth_location << "th layer------------" << std::endl;
		std::cout << "minx, miny, maxx, maxy: " << pNode->Box.minX << ", "
			<< pNode->Box.minY << ", " << pNode->Box.maxX << ", " << pNode->Box.maxY << std::endl;
		std::cout << "cloud point count: " << pNode->cloud->size() << std::endl;
		for (int i = 0; i < 4; i++)
		{
			if (pNode->children[i] != NULL)
			{
				SearchQuadTree(pNode->children[i]);
			}
		}
	}

}


void mpi_createTrainingData(int argc, char* argv[], int gridDimX, int gridDimY)
{
	double start, end, cost;
	if (argc < 6) {
		std::cout << "args not enough, only " << argc << std::endl;
		return;
	}
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	std::cout << rank << "th process is reading data..." << std::endl;
	std::string srcPt = argv[3];//input data

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(srcPt, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return;
	}
	// Declare a character array to store the destination time information
	char dstTime[100]; // sample_data

	// Use strncpy to safely copy the argument into dstTime, ensuring we do not exceed the buffer size
	strncpy(dstTime, argv[4], sizeof(dstTime) - 1);

	// Ensure the string is null-terminated by explicitly setting the last character to '\0'
	dstTime[sizeof(dstTime) - 1] = '\0';

	// Create a stringstream to convert the gridDimX integer into a string
	std::stringstream ss;
	ss << gridDimX;
	std::string s_gridDimX = ss.str();

	// Concatenate the string representation of gridDimX to dstTime
	strcat(dstTime, s_gridDimX.c_str());
	strcat(dstTime, "-");

	// Create a stringstream to convert gridDimY integer into a string
	std::stringstream ss1;
	ss1 << gridDimY;
	std::string s_gridDimY = ss1.str();

	// Concatenate the string representation of gridDimY to dstTime
	strcat(dstTime, s_gridDimY.c_str());
	strcat(dstTime, "/PCDEMPartitionTrain");

	// Create a stringstream to convert the rank integer into a string
	std::stringstream ss2;
	ss2 << rank;
	std::string s = ss2.str();

	// Concatenate the string representation of rank to dstTime and append ".txt" extension
	strcat(dstTime, s.c_str());
	strcat(dstTime, ".txt");

	// Output the resulting dstTime string
	std::cout << dstTime << std::endl;

	// Open the file stream with the generated dstTime string as the file name
	const char* dstTime_ = dstTime;
	std::ofstream out;
	out.open(dstTime_);

	// Declare a character array to store the destination DEM information
	char dstDem[100]; // output dem

	// Use strncpy to safely copy the argument into dstDem, ensuring we do not exceed the buffer size
	strncpy(dstDem, argv[5], sizeof(dstDem) - 1);

	// Ensure the string is null-terminated by explicitly setting the last character to '\0'
	dstDem[sizeof(dstDem) - 1] = '\0';

	// Concatenate the string representation of rank to dstDem and append ".txt" extension
	strcat(dstDem, s.c_str());
	strcat(dstDem, ".txt");

	// Use dstDem as the output file name
	const char* dstDem_ = dstDem;


	//caculate the extent
	double* maxX;
	maxX = new double[size];
	double* minX;
	minX = new double[size];
	double* maxY;
	maxY = new double[size];
	double* minY;
	minY = new double[size];
	int MIN_X, MIN_Y;
	int MAX_X, MAX_Y;

	int x_interval, pt_interval, pt_size;

	pt_size = cloud->size();
	pt_interval = pt_size / size;

	double minx = 99999999; double maxx = -99999999;
	double miny = 99999999; double maxy = -99999999;

	for (size_t i = rank * pt_interval; i < (rank + 1) * pt_interval; i++)
	{
		if (cloud->points[i].x < minx) minx = cloud->points[i].x;
		if (cloud->points[i].x > maxx) maxx = cloud->points[i].x;
		if (cloud->points[i].y < miny) miny = cloud->points[i].y;
		if (cloud->points[i].y > maxy) maxy = cloud->points[i].y;
	}

	MPI_Gather(&minx, 1, MPI_DOUBLE, minX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //All collected by process 0

	MPI_Gather(&maxx, 1, MPI_DOUBLE, maxX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //All collected by process 0

	MPI_Gather(&miny, 1, MPI_DOUBLE, minY, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //All collected by process 0

	MPI_Gather(&maxy, 1, MPI_DOUBLE, maxY, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //All collected by process 0
	MPI_Barrier(MPI_COMM_WORLD);

	if (!rank) {
		double min_x = 99999999;
		double max_x = -99999999;
		double min_y = 99999999;
		double max_y = -99999999;
		for (int i = 0; i < size; i++) {
			if (minX[i] < min_x) min_x = minX[i];
			if (maxX[i] > max_x) max_x = maxX[i];
			if (minY[i] < min_y) min_y = minY[i];
			if (maxY[i] > max_y) max_y = maxY[i];
		}
		MIN_X = int(min_x);
		MAX_X = int(max_x + 1);
		MIN_Y = int(min_y);
		MAX_Y = int(max_y + 1);

	}
	//bcast the extend
	MPI_Bcast(&MIN_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&MAX_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&MIN_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&MAX_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	range_Minx = MIN_X;
	range_Miny = MIN_Y;
	range_Maxx = MAX_X;
	range_Maxy = MAX_Y;

	delete[] minX;
	minX = NULL;
	delete[] minY;
	minY = NULL;
	delete[] maxX;
	maxX = NULL;
	delete[] maxY;
	maxY = NULL;
	if (rank == 1)
		std::cout << "extent: " << range_Minx << "," << range_Maxx << "," << range_Miny << "," << range_Maxy << std::endl;

	int sample_interval_x = int((range_Maxx - range_Minx) / gridDimX);//sample_interval_x
	int sample_interval_y = int((range_Maxy - range_Miny) / gridDimY);//sample_interval_y
	int batch_y = int((range_Maxy - range_Miny) / size);

	if (rank == 1)
		std::cout << "sample_interval_x, sample_interval_y, batch_y: " << sample_interval_x << " " << sample_interval_y << " " << batch_y << std::endl;

	//generate samples
	NodeRange rect;
	int count = 0;
	for (int i = range_Minx; i < range_Maxx; i += sample_interval_x)
	{
		rect.minX = i;
		rect.maxX = i + sample_interval_x;

		for (int j = 0; j < batch_y; j += sample_interval_y)
		{
			std::ofstream out_;
			out_.open(dstDem_);

			rect.minY = range_Miny + int(rank * batch_y / sample_interval_y) * sample_interval_y + j;
			rect.maxY = range_Miny + int(rank * batch_y / sample_interval_y) * sample_interval_y + j + sample_interval_y;
			start = clock();
			int neighbor_min_x = rect.minX - 10;
			int neighbor_max_x = rect.maxX + 10;
			int neighbor_min_y = rect.minY - 10;
			int neighbor_max_y = rect.maxY + 10;

			pcl::PointCloud<pcl::PointXYZ>::Ptr n_x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> n_x_pass_;
			n_x_pass_.setInputCloud(cloud);
			n_x_pass_.setFilterFieldName("x");
			n_x_pass_.setFilterLimits(neighbor_min_x, neighbor_max_x);
			n_x_pass_.filter(*n_x_cloud_filtered_);

			pcl::PointCloud<pcl::PointXYZ>::Ptr n_xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> n_xy_pass_;
			n_xy_pass_.setInputCloud(n_x_cloud_filtered_);
			n_xy_pass_.setFilterFieldName("y");
			n_xy_pass_.setFilterLimits(neighbor_min_y, neighbor_max_y);
			n_xy_pass_.filter(*n_xy_cloud_filtered_);

			pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> x_pass_;
			x_pass_.setInputCloud(n_xy_cloud_filtered_);
			x_pass_.setFilterFieldName("x");
			x_pass_.setFilterLimits(rect.minX, rect.maxX);
			x_pass_.filter(*x_cloud_filtered_);

			pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> xy_pass_;
			xy_pass_.setInputCloud(x_cloud_filtered_);
			xy_pass_.setFilterFieldName("y");
			xy_pass_.setFilterLimits(rect.minY, rect.maxY);
			xy_pass_.filter(*xy_cloud_filtered_);

			unsigned int local_point_num = xy_cloud_filtered_->size();

			double sum_value_x = 0;
			double sum_value_y = 0;
			double mean_value_x = 0;
			double mean_value_y = 0;
			double var_value = 0;
			pcl::PointCloud <pcl::PointXY> xycloud;
			pcl::PointXY pt;
			for (size_t p = 0; p < n_xy_cloud_filtered_->size(); p++)
			{
				pt.x = n_xy_cloud_filtered_->points[p].x;
				pt.y = n_xy_cloud_filtered_->points[p].y;
				xycloud.push_back(pt);
			}

			for (size_t p = 0; p < n_xy_cloud_filtered_->size(); p++)
			{
				sum_value_x += n_xy_cloud_filtered_->points[p].x;
				sum_value_y += n_xy_cloud_filtered_->points[p].y;
			}
			mean_value_x = sum_value_x / (double)(n_xy_cloud_filtered_->size() + 1);
			mean_value_y = sum_value_y / (double)(n_xy_cloud_filtered_->size() + 1);
			for (size_t p = 0; p < n_xy_cloud_filtered_->size(); p++)
			{
				var_value += ((double)n_xy_cloud_filtered_->points[p].x - mean_value_x) * ((double)n_xy_cloud_filtered_->points[p].x - mean_value_x) + ((double)n_xy_cloud_filtered_->points[p].y - mean_value_y) * ((double)n_xy_cloud_filtered_->points[p].y - mean_value_y);
			}
			var_value = sqrt(var_value / (double)(n_xy_cloud_filtered_->size() + 1));

			int x_size = rect.maxX - rect.minX;
			int y_size = rect.maxY - rect.minY;
			std::cout << "x_size: " << x_size << "y_size: " << y_size << std::endl;
			double* node_z = new double[x_size * y_size];

			if (n_xy_cloud_filtered_->size() >= 10)
			{
				pcl::KdTreeFLANN<pcl::PointXY> kdtree;
				kdtree.setInputCloud(xycloud.makeShared());
				for (size_t p = 0; p < x_size; p++)
				{
					pt.x = rect.minX + p * 1;
					for (size_t l = 0; l < y_size; l++)
					{
						double t_pz = 0;
						double t_p = 0;
						std::vector<int> pointIdxRadiusSearch;
						std::vector<float> pointRadiusquaredDistance;
						pt.y = rect.minY + l * 1;
						float radius = 3;
						while (kdtree.radiusSearch(pt, radius, pointIdxRadiusSearch, pointRadiusquaredDistance) < 0 || pointIdxRadiusSearch.size() < 10)
						{
							radius = radius + 0.005;
							if (radius > 10) break;
						}

						if (kdtree.radiusSearch(pt, radius, pointIdxRadiusSearch, pointRadiusquaredDistance) > 0 && pointIdxRadiusSearch.size() >= 10)
						{
							int pointNumIDW = pointIdxRadiusSearch.size();
							if (pointIdxRadiusSearch.size() > 30)
							{
								pointNumIDW = 30;
							}
							for (size_t k = 0; k < pointNumIDW; k++)
							{
								double dis = (pt.x - n_xy_cloud_filtered_->points[pointIdxRadiusSearch[k]].x) * (pt.x - n_xy_cloud_filtered_->points[pointIdxRadiusSearch[k]].x) + (pt.y - n_xy_cloud_filtered_->points[pointIdxRadiusSearch[k]].y) * (pt.y - n_xy_cloud_filtered_->points[pointIdxRadiusSearch[k]].y);
								double pp = 1 / dis;
								t_pz += n_xy_cloud_filtered_->points[pointIdxRadiusSearch[k]].z / dis;
								t_p += pp;
							}
							double z = t_pz / t_p;
							node_z[p * y_size + l] = z;
							out_ << node_z[p * y_size + l] << " ";
						}
						else
						{
							node_z[p * y_size + l] = -9999.999999;
							out_ << "-9999.999999" << " ";
						}
					}
					out_ << std::endl;
				}
			}
			else
			{
				for (size_t p = 0; p < x_size; p++)
				{
					for (size_t l = 0; l < y_size; l++)
					{
						node_z[p * y_size + l] = -9999.999999;
						out_ << node_z[p * y_size + l] << " ";
					}
					out_ << std::endl;
				}
			}
			delete[]node_z;
			node_z = NULL;
			end = clock();
			cost = end - start;

			int local_grid_number = x_size * y_size;
			int neighbor_grid_number = x_size * y_size + 2 * x_size * 10 + 2 * y_size * 10 + 4 * 10 * 10;
			out << local_point_num << "\t" << n_xy_cloud_filtered_->size() - local_point_num << "\t" << local_grid_number << "\t" << neighbor_grid_number - local_grid_number <<
				"\t" << (double)(n_xy_cloud_filtered_->size()) / (double)(neighbor_grid_number) << "\t" << var_value << "\t" << cost << std::endl;

			n_x_cloud_filtered_->clear();
			n_xy_cloud_filtered_->clear();
			x_cloud_filtered_->clear();
			xy_cloud_filtered_->clear();
			xycloud.clear();
			out_.close();
			std::cout << "the process " << rank << ":" << count << " tile has been calculated successfully" << std::endl;
			count++;
		}
	}
	out.close();
	MPI_Finalize();
}

int main(int argc, char* argv[])
{
	int gridDimX, gridDimY;
	sscanf(argv[1], "%d", &gridDimX);
	sscanf(argv[2], "%d", &gridDimY);

	double sum_begin = clock();
	mpi_createTrainingData(argc, argv, gridDimX, gridDimY);
	double sum_end = clock();

	std::cout << "sum time cost:" << sum_end - sum_begin << std::endl;
	system("pause");
	return 0;
}
