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
#include <pcl/common/common.h>

#include <nlohmann/json.hpp>


using json = nlohmann::json;


int range_Minx = 0;
int range_Miny = 0;
int range_Maxx = 0;
int range_Maxy = 0;
int init_depth = 4;
int global_depth = 0;
#define threads_num 6

struct DataEntry {
	std::string key;
	double value;
};
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

void arrDelete(Arr* p);

void arrDelete(Arr& p);

void CreateQuadTree(int depth, pcl::PointXY* pt, QuadTree* pQuadTree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


void CreateQuadBranch(int depth, NodeRange& rect, QuadNode** node, int nPointCount);

void init_arr(struct Arr* pArr, int length);

bool is_empty(struct Arr* pArr);

bool append_arr(struct Arr* pArr, int val);

void arrDelete(Arr* p)
{
	for (int i = 0; i < 1200; i++)
	{
		p->pBase[i]->cloud->clear();
		p->pBase[i]->local_cloud->clear();
	}
}
void arrDelete(Arr& p)
{
	for (int i = 0; i < 1200; i++)
	{
		(&p)->pBase[i]->cloud->clear();
		(&p)->pBase[i]->local_cloud->clear();
	}
}


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
		x_step = (int)((range_Maxx - range_Minx) / x_step);
		y_step = (int)((range_Maxy - range_Miny) / y_step);
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
// Function to create a quad-tree branch recursively
void CreateQuadBranch(int depth, NodeRange& rect, QuadNode* father, QuadNode** node)
{
	// If depth is not zero, continue dividing the node
	if (depth != 0)
	{
		if (depth == init_depth)
		{
			// Initialize the root node at the initial depth
			*node = InitQuadNode();  // Create the root node  
			QuadNode* pNode = *node;
			pNode->Box = rect;  // Set the bounding box for the node
			pNode->nChildCount = 4;  // Set the number of children (quad-tree)
			pNode->father = father;  // Set the parent node
			pNode->cloud = father->cloud;  // Set the point cloud of the parent node
			pNode->depth_location = init_depth - depth;  // Set the depth location

			// Split the bounding box into 4 smaller boxes (quadrants)
			NodeRange boxs[4];
			NodeRangeSplit(pNode->Box, boxs);

			// Create the 4 child nodes for the current node
			for (int i = 0; i < 4; i++)
			{
				// Initialize each child node and assign the respective bounding box
				pNode->children[i] = InitQuadNode();
				pNode->children[i]->Box = boxs[i];
				pNode->children[i]->father = pNode;
				pNode->children[i]->depth_location = pNode->depth_location + 1;

				// Define neighbor range with a 10-unit padding around each box
				int neighbor_min_x = boxs[i].minX - 10;
				int neighbor_max_x = boxs[i].maxX + 10;
				int neighbor_min_y = boxs[i].minY - 10;
				int neighbor_max_y = boxs[i].maxY + 10;

				// Filter out points that lie within the neighbor grid along the X axis
				pcl::PointCloud<pcl::PointXYZ>::Ptr n_x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_x_pass_;
				n_x_pass_.setInputCloud(pNode->cloud);  // Set input point cloud
				n_x_pass_.setFilterFieldName("x");  // Filter along the X axis
				n_x_pass_.setFilterLimits(neighbor_min_x, neighbor_max_x);  // Set filtering limits
				n_x_pass_.filter(*n_x_cloud_filtered_);  // Apply filtering

				// Further filter the points along the Y axis
				pcl::PointCloud<pcl::PointXYZ>::Ptr n_xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_xy_pass_;
				n_xy_pass_.setInputCloud(n_x_cloud_filtered_);  // Set input point cloud
				n_xy_pass_.setFilterFieldName("y");  // Filter along the Y axis
				n_xy_pass_.setFilterLimits(neighbor_min_y, neighbor_max_y);  // Set filtering limits
				n_xy_pass_.filter(*n_xy_cloud_filtered_);  // Apply filtering
				pNode->children[i]->cloud = n_xy_cloud_filtered_;  // Store the filtered cloud in the child node

				// Filter out points that lie within the local grid along the X axis
				pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> x_pass_;
				x_pass_.setInputCloud(n_xy_cloud_filtered_);  // Set input point cloud
				x_pass_.setFilterFieldName("x");  // Filter along the X axis
				x_pass_.setFilterLimits(boxs[i].minX, boxs[i].maxX);  // Set filtering limits
				x_pass_.filter(*x_cloud_filtered_);  // Apply filtering

				// Filter the points along the Y axis
				pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> xy_pass_;
				xy_pass_.setInputCloud(x_cloud_filtered_);  // Set input point cloud
				xy_pass_.setFilterFieldName("y");  // Filter along the Y axis
				xy_pass_.setFilterLimits(boxs[i].minY, boxs[i].maxY);  // Set filtering limits
				xy_pass_.filter(*xy_cloud_filtered_);  // Apply filtering
				pNode->children[i]->local_cloud = xy_cloud_filtered_;  // Store the filtered cloud in the child node

				// Recursively create the next branch at the next depth
				CreateQuadBranch(depth - 1, boxs[i], pNode->children[i]->father, &(pNode->children[i]));
			}
		}
		else
		{
			// If not at the initial depth, process a sub-tree node
			QuadNode* pNode = *node;
			NodeRange boxs[4];
			global_depth = depth;
			NodeRangeSplit(pNode->Box, boxs);  // Split the bounding box
			pNode->depth_location = init_depth - depth;  // Update depth location

			// Create 4 child nodes and process each quadrant
			for (int i = 0; i < 4; i++)
			{
				// Initialize each child node and assign the respective bounding box
				pNode->children[i] = InitQuadNode();
				pNode->children[i]->Box = boxs[i];
				pNode->children[i]->father = pNode;
				pNode->children[i]->depth_location = pNode->depth_location + 1;

				// Define neighbor range with a 10-unit padding around each box
				int neighbor_min_x = boxs[i].minX - 10;
				int neighbor_max_x = boxs[i].maxX + 10;
				int neighbor_min_y = boxs[i].minY - 10;
				int neighbor_max_y = boxs[i].maxY + 10;

				// Filter out points that lie within the neighbor grid along the X and Y axes (repeat process as above)
				pcl::PointCloud<pcl::PointXYZ>::Ptr n_x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_x_pass_;
				n_x_pass_.setInputCloud(pNode->cloud);  // Set input point cloud
				n_x_pass_.setFilterFieldName("x");  // Filter along the X axis
				n_x_pass_.setFilterLimits(neighbor_min_x, neighbor_max_x);  // Set filtering limits
				n_x_pass_.filter(*n_x_cloud_filtered_);  // Apply filtering

				pcl::PointCloud<pcl::PointXYZ>::Ptr n_xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> n_xy_pass_;
				n_xy_pass_.setInputCloud(n_x_cloud_filtered_);  // Set input point cloud
				n_xy_pass_.setFilterFieldName("y");  // Filter along the Y axis
				n_xy_pass_.setFilterLimits(neighbor_min_y, neighbor_max_y);  // Set filtering limits
				n_xy_pass_.filter(*n_xy_cloud_filtered_);  // Apply filtering
				pNode->children[i]->cloud = n_xy_cloud_filtered_;  // Store the filtered cloud in the child node

				// Filter out points that lie within the local grid along the X and Y axes (repeat process as above)
				pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> x_pass_;
				x_pass_.setInputCloud(n_xy_cloud_filtered_);  // Set input point cloud
				x_pass_.setFilterFieldName("x");  // Filter along the X axis
				x_pass_.setFilterLimits(boxs[i].minX, boxs[i].maxX);  // Set filtering limits
				x_pass_.filter(*x_cloud_filtered_);  // Apply filtering

				pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PassThrough<pcl::PointXYZ> xy_pass_;
				xy_pass_.setInputCloud(x_cloud_filtered_);  // Set input point cloud
				xy_pass_.setFilterFieldName("y");  // Filter along the Y axis
				xy_pass_.setFilterLimits(boxs[i].minY, boxs[i].maxY);  // Set filtering limits
				xy_pass_.filter(*xy_cloud_filtered_);  // Apply filtering
				pNode->children[i]->local_cloud = xy_cloud_filtered_;  // Store the filtered cloud in the child node

				// Recursively create the next branch at the next depth
				CreateQuadBranch(depth - 1, boxs[i], pNode->children[i]->father, &(pNode->children[i]));
			}
		}
	}
	else
	{
		// When depth reaches zero, append the node to the array
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


// callGCN fuction
void callGCN(const char* script, const char* funcName, const char* modelPath, const char* filename, const char* outputCIPath, const char* gpuIndex)
{
	std::cout << "call GCN..." << std::endl;
	std::string command = "python "
		+ std::string(script) + " "
		+ std::string(funcName) + " "
		+ std::string(modelPath) + " "
		+ std::string(filename) + " "
		+ std::string(outputCIPath) + " "
		+ std::string(gpuIndex);

	// Executing
	std::cout << "Executing command: " << command << std::endl;
	int result = system(command.c_str());

	if (result != 0) {
		std::cerr << "Failed to execute command" << std::endl;
	}
}

// Function to parse each line of data and return a DataEntry structure
DataEntry parseLine(const std::string& line) {
	DataEntry entry;
	std::stringstream ss(line);
	std::string token;

	// Split the line by commas
	std::getline(ss, token, ',');
	entry.key = token; // The first part is the key

	// Convert the second part to a double type
	std::getline(ss, token, ',');
	entry.value = std::stod(token); // The second part is the value, converted to double

	return entry; // Return the populated DataEntry structure
}


std::vector<std::pair<double, double>> compute_grid_centers(double min_x, double max_x, double min_y, double max_y, int neighbor_pixels, double resolution) {
	double all_min_x = min_x - neighbor_pixels;
	double all_min_y = min_y - neighbor_pixels;
	double all_max_x = max_x + neighbor_pixels;
	double all_max_y = max_y + neighbor_pixels;

	double x_range = all_max_x - all_min_x;
	double y_range = all_max_y - all_min_y;

	int num_x_grid = static_cast<int>(x_range / resolution);
	int num_y_grid = static_cast<int>(y_range / resolution);

	std::vector<std::pair<double, double>> centers;
	for (int i = 0; i < num_x_grid; ++i) {
		for (int j = 0; j < num_y_grid; ++j) {
			double x_center = all_min_x + 0.5 + i * resolution;
			double y_center = all_min_y + 0.5 + j * resolution;
			centers.emplace_back(x_center, y_center);
		}
	}

	return centers;
}

int main(int argc, char* argv[])
{
	double start, end, cost;
	start = clock();

	int x_interval, pt_interval, pt_size;
	if (argc != 8) {
		std::cerr << "Usage: " << argv[0] << " <filename> <script> <function> <modelPath> <outputCIPath> <gpuIndex> <srcPt>" << std::endl;
		return 1;
	}
	const char* filename = argv[1];
	const char* script = argv[2];
	const char* function = argv[3];
	const char* modelPath = argv[4];
	const char* outputCIPath = argv[5];
	const char* gpuIndex = argv[6];
	const char* srcPt = argv[7];

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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

	double* thread_cost;
	thread_cost = new double[size];
	double rank_cost;

	int* thread_tiles;
	thread_tiles = new int[size];

	std::cout << rank << "th process is reading data..." << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(srcPt, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}

	clock_t start_mpi = clock();

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
		clock_t end_mpi = clock();
		double elapsed_mpi = static_cast<double>(end_mpi - start_mpi) / CLOCKS_PER_SEC;

		std::cout << "MPI Extent: " << MIN_X << "," << MAX_X << "," << MIN_Y << "," << MAX_Y << std::endl;
		std::cout << "MPI Time taken: " << elapsed_mpi << " seconds" << std::endl;
	}


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


	// Allocate memory for group_point_thread to store the number of tiles each thread will receive
	int* group_point_thread = new int[1];

	// Allocate memory for group_thread to store the number of threads
	int* group_thread = new int[1];

	// Allocate memory for group_point to store the total intensity sum for each thread
	// The size of the array is determined by the number of threads (size)
	double* group_point = new double[size];

	// Initialize the group_point array with zeros
	for (int i = 0; i < size; i++)
	{
		group_point[i] = 0;  // Each thread starts with a total intensity of 0
	}

	// Allocate memory for split_node to store the indices corresponding to the dividing line
	// This will divide the tasks into sections that each thread can work on
	int* split_node = new int[size + 1];

	// Initialize the split_node array with zeros
	for (int i = 0; i < (size + 1); i++)
	{
		split_node[i] = 0;  // Default values (0) indicating no split initially
	}

	// Allocate memory for split_group to store the number of tiles each thread will handle
	int* split_group = new int[size];

	// Initialize the split_group array with zeros
	for (int i = 0; i < size; i++)
	{
		split_group[i] = 0;  // Initially, each thread will handle 0 tiles
	}


	if (!rank)
	{
		json rank_data;
		double split_start = clock();
		global_depth = 0;
		QuadTree* pQuadTree = new QuadTree;
		init_arr(&arr, 1200);
		pQuadTree->depth = init_depth;
		CreateQuadTree(pQuadTree->depth, pQuadTree, cloud);
		std::cout << "pointSum: " << cloud->size() << std::endl;
		std::cout << "gridNumfterInitialPartition: " << arr.cnt << std::endl;
		unsigned int pointSum = 0;
		for (int p = 0; p < arr.cnt; p++)
		{
			arr.pBase[p]->compute_density = 0.000000001;
			pointSum += arr.pBase[p]->local_cloud->size();
		}
		std::cout << std::endl;
		std::cout << "pointSumAfterInitialPartition: " << pointSum << std::endl;

		int backStep = 1;
		double tdr = 1.0;
		while (backStep < 2 && tdr > 0.15)
		{
			global_depth = 0;
			init_depth += backStep;
			Arr arrCopy;
			memcpy(&arrCopy, &arr, sizeof(struct Arr));
			init_arr(&arr, 1200);
			for (int p = 0; p < arrCopy.cnt; p++)
			{
				if (arrCopy.pBase[p]->compute_density > 0)
					CreateQuadBranch(1, arrCopy.pBase[p]->Box, arrCopy.pBase[p]->father, &(arrCopy.pBase[p]));
				else
					append_arr(&arr, arrCopy.pBase[p]);
			}
			clock_t start1 = clock(); // Start the clock to measure the execution time
			for (int p = 0; p < arr.cnt; p++) { // Loop through each element in the array
				// Extract the bounding box coordinates for the current element
				double rect_minx = arr.pBase[p]->Box.minX;
				double rect_miny = arr.pBase[p]->Box.minY;
				double rect_maxx = arr.pBase[p]->Box.maxX;
				double rect_maxy = arr.pBase[p]->Box.maxY;

				// Calculate the number of grid cells in x and y directions based on resolution
				int x_size = int((arr.pBase[p]->Box.maxX - arr.pBase[p]->Box.minX) / 1);
				int y_size = int((arr.pBase[p]->Box.maxY - arr.pBase[p]->Box.minY) / 1);

				// Prepare to convert the grid points into a graph structure in JSON format and save to a file
				std::string key = std::to_string(rank) + "_" + std::to_string(p); // Create a unique key for the current point
				json json_data; // Create a JSON object to store graph structure
				json_data["xyz"] = json::array(); // Initialize an array for storing point data
				cost = 0.0000001; // Set a small cost value
				json_data["cost"] = cost; // Add cost to JSON

				int neighbor_pixels = 10; // Number of neighboring pixels
				double resolution = 1.0; // Grid resolution (1 unit per grid cell)

				// Initialize a 2D array to store the count of points in each grid cell
				int num_x_grid = static_cast<int>((rect_maxx - rect_minx) / resolution);
				int num_y_grid = static_cast<int>((rect_maxy - rect_miny) / resolution);
				std::vector<std::vector<int>> grid_counts(num_x_grid, std::vector<int>(num_y_grid, 0));

				// Loop through all the points in the point cloud and update the grid cell counts
				for (const auto& point : arr.pBase[p]->cloud->points) {
					// Compute the grid indices based on the point's coordinates
					int x_idx = static_cast<int>((point.x - rect_minx) / resolution);
					int y_idx = static_cast<int>((point.y - rect_miny) / resolution);

					// Check if the point falls within the grid boundaries
					if (x_idx >= 0 && x_idx < num_x_grid && y_idx >= 0 && y_idx < num_y_grid) {
						grid_counts[x_idx][y_idx]++; // Increment the count for the corresponding grid cell
					}
				}

				int grid_with_points_count = 0; // Variable to store the number of grid cells containing points
				// Loop through the grid cells and generate the corresponding JSON data
				for (int i = 0; i < num_x_grid; ++i) {
					for (int j = 0; j < num_y_grid; ++j) {
						// Calculate the center coordinates of the current grid cell
						double center_x = rect_minx + (i + 0.5) * resolution;
						double center_y = rect_miny + (j + 0.5) * resolution;

						int point_count = grid_counts[i][j]; // Get the number of points in the current grid cell

						// Flag to check if the grid cell is within the bounding box
						int in_rect_flag = (rect_minx <= center_x && center_x <= rect_maxx && rect_miny <= center_y && center_y <= rect_maxy) ? 1 : 0;

						// Add the point data to the JSON object
						json_data["xyz"].push_back({ point_count, center_x, center_y, in_rect_flag });

						// If there are points in the current grid cell, increment the grid_with_points_count
						if (point_count > 0) {
							grid_with_points_count++;
						}
					}
				}

				// Add the count of grid cells containing points to the JSON object
				json_data["grid_with_points_count"] = grid_with_points_count;
				// Add bounding box information to the JSON object
				json_data["minX"] = rect_minx;
				json_data["maxX"] = rect_maxx;
				json_data["minY"] = rect_miny;
				json_data["maxY"] = rect_maxy;

				// Store the JSON data in the rank_data map using the generated key
				rank_data[key] = json_data;
			}
			clock_t end1 = clock(); // End the clock to measure execution time
			double elapsed1 = static_cast<double>(end1 - start1) / CLOCKS_PER_SEC; // Calculate the elapsed time
			std::cout << "Method 1 Time taken: " << elapsed1 << " seconds" << std::endl; // Output the execution time

			// Save the current rank's point cloud data to a separate JSON file
			std::ofstream outfile(filename); // Open the file for writing
			if (!outfile) { // Check if the file was opened successfully
				std::cerr << "Error opening output file: " << filename << std::endl; // Print an error message if the file cannot be opened
			}
			outfile << rank_data.dump(4);
			outfile.close();
			std::cout << "Filtered point clouds saved to " << filename << std::endl;
			double pc_preprocess_end = clock();
			std::cout << "pc_preprocess time:" << (pc_preprocess_end - split_start) << std::endl;
			callGCN(script, function, modelPath, filename, outputCIPath, gpuIndex);

			std::ifstream infile(outputCIPath);
			if (!infile) {
				std::cerr << "Error opening file: " << outputCIPath << std::endl;
			}
			std::cout << "arr.cnt: " << arr.cnt << std::endl;
			std::string line;
			while (std::getline(infile, line)) {
				DataEntry entry = parseLine(line);
				std::string key_prefix = std::to_string(rank) + "_";
				if (entry.key.compare(0, key_prefix.size(), key_prefix) == 0) {
					int index = std::stoi(entry.key.substr(key_prefix.size()));
					if (index >= 0 && index < arr.cnt) {
						arr.pBase[index]->compute_density = entry.value;
						if (arr.pBase[index]->compute_density <= 0) {
							arr.pBase[index]->compute_density = 0.0000001;
						}
					}
				}
			}
			infile.close();

			int sum_point = 0;
			double sum_density = 0;
			for (int i = 0; i < arr.cnt; i++)
			{
				sum_density += arr.pBase[i]->compute_density;
			}
			std::cout << "sum intensity:" << sum_density << std::endl;
			double* group_para = new double[1000];
			double op_group_para = 0;
			double op_min = 1100000000;


			for (int count = 0; count < 1000; count++)
			{
				group_para[count] = size + 0.01 * count; 
				double aver_group_density = (double)(sum_density / group_para[count]);

				int split = 0;

				for (int i = 0; i < size; i++)
				{
					split_node[i] = split;
					for (int j = split; j < arr.cnt; j++)
					{
						if (group_point[i] <= aver_group_density)
						{
							split++;
							group_point[i] += arr.pBase[j]->compute_density;
						}
						else if (i == (size - 1))
						{
							group_point[i] += arr.pBase[j]->compute_density;
							split++;
						}
						else
							break;
					}

				}
				split_node[size] = split;



				double cp_min = 1000000000;
				double cp_max = -1000000000;
				for (int i = 0; i < size; i++)
				{
					if (group_point[i] < cp_min)cp_min = group_point[i];
					if (group_point[i] > cp_max)cp_max = group_point[i];
				}
				std::cout << "cp_max: " << cp_max << ", cp_min: " << cp_min << std::endl;
				double density_difference = cp_max - cp_min; // Density Difference Calculation
				std::cout << "Density difference for group_para[" << count << "] = " << density_difference << std::endl; // Density Difference output

				if ((cp_max - cp_min) < op_min)
				{
					op_min = cp_max - cp_min;
					tdr = op_min / cp_max;
					op_group_para = group_para[count];
					std::cout << tdr << ", " << op_group_para << std::endl;
				}


				for (int i = 0; i < size; i++)
				{
					group_point[i] = 0;
				}
			}
			std::cout << "op_group_par = " << op_group_para << std::endl;

			double aver_group_density = (double)(sum_density / op_group_para);
			int split = 0;
			for (int i = 0; i < size; i++)
			{
				split_node[i] = split;
				for (int j = split; j < arr.cnt; j++)
				{
					if (group_point[i] <= aver_group_density)
					{
						split++;
						group_point[i] += arr.pBase[j]->compute_density;
					}
					else if (i == (size - 1))
					{
						group_point[i] += arr.pBase[j]->compute_density;
						split++;
					}
					else
						break;
				}

			}
			split_node[size] = split;

			for (int i = 0; i < size; i++)
			{
				std::cout << "group density " << i << ": " << group_point[i] << std::endl;
			}

			for (int i = 0; i < size; i++)
			{
				split_group[i] = split_node[i + 1] - split_node[i];
				std::cout << "tile size " << i << ": " << split_group[i] << std::endl;
			}
			backStep++;
			std::cout << "--------------------" << backStep << "," << tdr << "------------------" << std::endl;
		}

		double split_end = clock();
		std::cout << "split time:" << (split_end - split_start) << std::endl;

		std::cout << "gridNumAfterDynamicPartition: " << arr.cnt << std::endl;
		pointSum = 0;
		for (int p = 0; p < arr.cnt; p++)
		{
			pointSum += arr.pBase[p]->local_cloud->size();
		}
		std::cout << std::endl;
		std::cout << "pointSumAfterDynamicPartition: " << pointSum << std::endl;
	}

	MPI_Datatype my_range;
	int blocklens[] = { 1, 1, 1, 1 };
	MPI_Datatype old_types[] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT };
	MPI_Aint indices[] = { 0, sizeof(int), 2 * sizeof(int), 3 * sizeof(int) };
	MPI_Type_create_struct(4, blocklens, indices, old_types, &my_range);
	MPI_Type_commit(&my_range);

	MPI_Scatter(split_group, 1, MPI_INT, group_thread, 1, MPI_INT, 0, MPI_COMM_WORLD);
	NodeRange* nodeRange = new NodeRange[arr.cnt];
	NodeRange* nodeRange_thread = new NodeRange[group_thread[0]];
	for (int i = 0; i < arr.cnt; i++)
	{
		nodeRange[i] = arr.pBase[i]->Box;
	}
	int* disp = new int[size];

	for (int i = 0; i < size; i++)
	{
		disp[i] = split_node[i];
	}
	MPI_Scatterv(nodeRange, split_group, disp, my_range, nodeRange_thread, group_thread[0], my_range, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	double rank_start = clock();

	char dstDem[100] = "/path/to/result/partitionDem";

	std::stringstream ss;
	ss << rank;
	std::string s = ss.str();
	strcat(dstDem, s.c_str());
	strcat(dstDem, ".txt");
	const char* dstDem_ = dstDem;
	std::ofstream out;
	out.open(dstDem_);

	char dstTime[100] = "/path/to/result/actual_time";
	strcat(dstTime, s.c_str());
	strcat(dstTime, ".txt");
	const char* dstTime_ = dstTime;
	std::ofstream out_;
	out_.open(dstTime_);

	double sum_time_cost = 0;
	// Open a file in write mode
	std::ofstream outfile("/path/to/result/tile_extent.txt");
	for (int p = 0; p < group_thread[0]; p++)
	{
		if (p < 1024) {
			double start_tile = clock();
			int neighbor_min_x = nodeRange_thread[p].minX - 10;
			int neighbor_max_x = nodeRange_thread[p].maxX + 10;
			int neighbor_min_y = nodeRange_thread[p].minY - 10;
			int neighbor_max_y = nodeRange_thread[p].maxY + 10;

			pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> x_pass;
			x_pass.setInputCloud(cloud);



			x_pass.setFilterFieldName("x");
			x_pass.setFilterLimits(neighbor_min_x, neighbor_max_x);
			x_pass.filter(*x_cloud_filtered);


			pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> xy_pass;
			xy_pass.setInputCloud(x_cloud_filtered);
			xy_pass.setFilterFieldName("y");
			xy_pass.setFilterLimits(neighbor_min_y, neighbor_max_y);
			xy_pass.filter(*xy_cloud_filtered);


			pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> x_pass_;
			x_pass_.setInputCloud(xy_cloud_filtered);
			x_pass_.setFilterFieldName("x");
			x_pass_.setFilterLimits(nodeRange_thread[p].minX, nodeRange_thread[p].maxX);
			x_pass_.filter(*x_cloud_filtered_);


			pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PassThrough<pcl::PointXYZ> xy_pass_;
			xy_pass_.setInputCloud(x_cloud_filtered_);
			xy_pass_.setFilterFieldName("y");
			xy_pass_.setFilterLimits(nodeRange_thread[p].minY, nodeRange_thread[p].maxY);
			xy_pass_.filter(*xy_cloud_filtered_);

			pcl::PointCloud <pcl::PointXY> xycloud;
			pcl::PointXY pt;
			for (size_t i = 0; i < xy_cloud_filtered->size(); i++)
			{
				pt.x = xy_cloud_filtered->points[i].x;
				pt.y = xy_cloud_filtered->points[i].y;
				xycloud.push_back(pt);
			}


			double sum_value_x = 0;
			double sum_value_y = 0;
			double mean_value_x = 0;
			double mean_value_y = 0;
			double var_value = 0;
			for (size_t i = 0; i < xy_cloud_filtered->size(); i++)
			{
				sum_value_x += xy_cloud_filtered->points[i].x;
				sum_value_y += xy_cloud_filtered->points[i].y;
			}
			mean_value_x = sum_value_x / (double)(xy_cloud_filtered->size() + 1);
			mean_value_y = sum_value_y / (double)(xy_cloud_filtered->size() + 1);
			for (size_t i = 0; i < xy_cloud_filtered->size(); i++)
			{
				var_value += ((double)xy_cloud_filtered->points[i].x - mean_value_x) * ((double)xy_cloud_filtered->points[i].x - mean_value_x) + ((double)xy_cloud_filtered->points[i].y - mean_value_y) * ((double)xy_cloud_filtered->points[i].y - mean_value_y);
			}
			var_value = var_value / (double)(xy_cloud_filtered->size() + 1);




			double node_min_x = nodeRange_thread[p].minX;
			double node_min_y = nodeRange_thread[p].minY;
			double node_max_x = nodeRange_thread[p].maxX;
			double node_max_y = nodeRange_thread[p].maxY;

			int x_size = node_max_x - node_min_x;
			int y_size = node_max_y - node_min_y;

			double* node_z = new double[x_size * y_size];

			if (xy_cloud_filtered->size() >= 10)
			{
				pcl::KdTreeFLANN<pcl::PointXY> kdtree;
				kdtree.setInputCloud(xycloud.makeShared());
				for (size_t i = 0; i < x_size; i++) {
					pt.x = node_min_x + i * 1; // Set the x-coordinate for the current point

					for (size_t j = 0; j < y_size; j++) {
						double t_pz = 0; // Initialize the variable for accumulating weighted z values
						double t_p = 0;  // Initialize the variable for accumulating weights
						std::vector<int> pointIdxRadiusSearch; // Vector to hold the indices of neighboring points
						std::vector<float> pointRadiusquaredDistance; // Vector to hold the squared distances of neighboring points
						pt.y = node_min_y + j * 1; // Set the y-coordinate for the current point
						float radius = 3; // Start with a radius of 3 for the radius search

						// Perform radius search and adjust the radius until enough points are found or max radius is reached
						while (kdtree.radiusSearch(pt, radius, pointIdxRadiusSearch, pointRadiusquaredDistance) < 0 || pointIdxRadiusSearch.size() < 10) {
							radius = radius + 0.005; // Increase the radius by a small increment
							if (radius > 10) break; // Stop if the radius exceeds 10
						}

						int numNeighbors = kdtree.radiusSearch(pt, radius, pointIdxRadiusSearch, pointRadiusquaredDistance); // Perform the radius search

						// Check if there are enough neighbors (at least 10) and the search was successful
						if (kdtree.radiusSearch(pt, radius, pointIdxRadiusSearch, pointRadiusquaredDistance) > 0 && pointIdxRadiusSearch.size() >= 10) {
							int pointNumIDW = pointIdxRadiusSearch.size(); // Get the number of neighbors
							if (pointIdxRadiusSearch.size() > 30) {
								pointNumIDW = 30; // Limit the number of points to 30 for further calculation
							}

							// Reset accumulation variables for weighted average
							t_pz = 0;
							t_p = 0;

							for (size_t k = 0; k < pointNumIDW; k++) {
								// Calculate the squared distance between the current point and the neighbor
								double dis = (pt.x - xy_cloud_filtered->points[pointIdxRadiusSearch[k]].x) * (pt.x - xy_cloud_filtered->points[pointIdxRadiusSearch[k]].x) +
									(pt.y - xy_cloud_filtered->points[pointIdxRadiusSearch[k]].y) * (pt.y - xy_cloud_filtered->points[pointIdxRadiusSearch[k]].y);

								// Skip if the distance is zero to avoid division by zero
								if (dis == 0) {
									std::cout << "Zero distance encountered at index " << k << " for point (" << pt.x << ", " << pt.y << ")" << std::endl;
									continue;
								}

								// Calculate the inverse distance for weighting
								double pp = 1 / dis;

								// Accumulate the weighted z value and the weight
								t_pz += xy_cloud_filtered->points[pointIdxRadiusSearch[k]].z * pp;
								t_p += pp;
							}

							// If there is a valid total weight (t_p != 0), compute the weighted average z value
							if (t_p != 0) {
								double z = t_pz / t_p; // Calculate the weighted average z value
								node_z[i * y_size + j] = z; // Store the result in the node_z array
							}
							else {
								// If t_p is zero, it indicates no valid neighbors or invalid distance, set the value to an invalid value
								node_z[i * y_size + j] = -9999.999999; // Or other appropriate invalid value
							}

							out << node_z[i * y_size + j] << " "; // Write the result to the output
						}
						else {
							// If there are not enough neighbors, set the value to an invalid value
							node_z[i * y_size + j] = -9999.999999; // Or other appropriate invalid value
							out << node_z[i * y_size + j] << " "; // Write the result to the output
						}
					}
					out << std::endl; // Move to the next line in the output
				}
			}
			else
			{
				for (size_t i = 0; i < x_size; i++)
				{
					for (size_t j = 0; j < y_size; j++)
					{
						node_z[i * y_size + j] = -9999.999999;
						out << node_z[i * y_size + j] << " ";
					}
					out << std::endl;
				}
			}
			double end_tile = clock();
			double cost_tile = end_tile - start_tile;
			sum_time_cost += cost_tile;
			out_ << cost_tile << std::endl;

			std::cout << "thread " << rank << ": " << p << " in " << group_thread[0] << std::endl;

			// Write the values to the file
			outfile << p << " "
				<< node_min_x << " "
				<< node_min_y << " "
				<< node_max_x << " "
				<< node_max_y << std::endl;


			delete[]node_z;
			node_z = NULL;
			x_cloud_filtered_->clear();
			xy_cloud_filtered_->clear();
			x_cloud_filtered->clear();
			xy_cloud_filtered->clear();
			xycloud.clear();
		}
	}
	// Close the file
	outfile.close();
	std::cout << "Data written to file successfully." << std::endl;


	out_ << "the thread sum cost time: " << sum_time_cost << std::endl;
	double rank_end = clock();
	rank_cost = rank_end - rank_start;
	out.close();
	MPI_Gather(&rank_cost, 1, MPI_DOUBLE, thread_cost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //È«ÊÕ¼¯ÓÚ½ø³Ì0
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(&group_thread[0], 1, MPI_INT, thread_tiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (!rank)
	{
		end = clock();
		cost = end - start;
		std::cout << "sum time cost: " << cost << std::endl;
		for (size_t i = 0; i < size; i++) {
			std::cout << "thread " << i << " actual execution time: " << thread_cost[i] << std::endl;
		}
		for (size_t i = 0; i < size; i++)
		{
			std::cout << "thread " << i << " predicted execution time: " << group_point[i] << std::endl;
		}

	}

	MPI_Finalize();
	system("pause");
	return 0;
}
