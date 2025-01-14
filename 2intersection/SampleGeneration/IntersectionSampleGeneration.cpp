#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"
#include "mpi.h"
#include "Util.h"

using namespace std;

double singleGridSampleGeneration(int count, GeoGrid* geoGrid, double gridMinX, double gridMaxX, double gridMinY, double gridMaxY, OGRLayer* poLayer, OGRLayer* poLayer1, OGRLayer* poLayer2, char* outLabelDir, char* outImageDir, string strRank, OGRFeature* poDstFeature, GDALDriver* poDriver, OGRSpatialReference* poSpatialRef, ofstream& foutSample);

double singleGridSampleGeneration(int count, double gridMinX, double gridMaxX, double gridMinY, double gridMaxY, int sumRecord1, int sumRecord2, OGRLayer* poLayer, OGRLayer* poLayer1, OGRLayer* poLayer2, char* outLabelDir, char* outImageDir, string strRank, OGRFeature* poDstFeature, GDALDriver* poDriver, OGRSpatialReference* poSpatialRef, ofstream& foutSample);

void parallelIntersectionSampleGeneration1(const char* inPath1, const char* inPath2, char* outLabelDir, char* outImageDir, int gridDimX, int gridDimY);

int main(int argc, char* argv[]) 
{
        int gridDimX, gridDimY;
        sscanf(argv[5], "%d", &gridDimX);
        sscanf(argv[6], "%d", &gridDimY);
        parallelIntersectionSampleGeneration1(argv[1], argv[2], argv[3], argv[4], gridDimX, gridDimY);

        return 0;
}

double singleGridSampleGeneration(int count, GeoGrid *geoGrid, double gridMinX, double gridMaxX, double gridMinY, double gridMaxY, OGRLayer* poLayer, OGRLayer* poLayer1, OGRLayer* poLayer2, char* outLabelDir, char* outImageDir, string strRank, OGRFeature* poDstFeature, GDALDriver* poDriver, OGRSpatialReference* poSpatialRef, ofstream& foutSample)
{
	OGRFeature* poFeature1, * poFeature2;
	int polygonsNum1 = geoGrid->polygonsFID1.size();
	int polygonsNum2 = geoGrid->polygonsFID2.size();
	int verticesNum1 = 0, verticesNum2 = 0;
	double varianceLayer1 = 0.0, varianceLayer2 = 0.0;
	double avgDist;
	int referencePoint = 0;

	double xSumLayer1 = 0, ySumLayer1 = 0;
	double xSumLayer2 = 0, ySumLayer2 = 0;

	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);

		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		verticesNum1 += polygon1->getExteriorRing()->getNumPoints();
		for (int t = 0; t < polygon1->getNumInteriorRings(); t++)
			verticesNum1 += polygon1->getInteriorRing(t)->getNumPoints();
		OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
		xSumLayer1 += poPoint->getX(); ySumLayer1 += poPoint->getY();
	}
	double xAverage1 = xSumLayer1 / polygonsNum1;
	double yAverage1 = ySumLayer1 / polygonsNum1;
	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
		varianceLayer1 = varianceLayer1 + (poPoint->getX() - xAverage1) * (poPoint->getX() - xAverage1) +
			(poPoint->getY() - yAverage1) * (poPoint->getY() - yAverage1);
	}
	varianceLayer1 = varianceLayer1 / polygonsNum1;

	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);

		OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
		verticesNum2 += polygon2->getExteriorRing()->getNumPoints();
		for (int t = 0; t < polygon2->getNumInteriorRings(); t++)
			verticesNum2 += polygon2->getInteriorRing(t)->getNumPoints();
		OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
		xSumLayer2 += poPoint->getX(); ySumLayer2 += poPoint->getY();
	}
	double xAverage2 = xSumLayer2 / polygonsNum2;
	double yAverage2 = ySumLayer2 / polygonsNum2;
	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);
		OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
		OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
		varianceLayer2 = varianceLayer2 + (poPoint->getX() - xAverage2) * (poPoint->getX() - xAverage2) +
			(poPoint->getY() - yAverage2) * (poPoint->getY() - yAverage2);
	}
	varianceLayer2 = varianceLayer2 / polygonsNum2;
	avgDist = (xAverage1 - xAverage2) * (xAverage1 - xAverage2) + (yAverage1 - yAverage2) * (yAverage1 - yAverage2);

	if (polygonsNum1 == 0)varianceLayer1 = 0; if (polygonsNum2 == 0)varianceLayer2 = 0;
	if (polygonsNum1 == 0 || polygonsNum2 == 0)avgDist = 100000;


	double gridBegin = clock();
	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		OGREnvelope* envelopePolygon1 = new OGREnvelope();
		polygon1->getEnvelope(envelopePolygon1);

		for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
		{
			int FID2 = geoGrid->polygonsFID2[j2];
			poFeature2 = poLayer2->GetFeature(FID2);
			OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
			OGREnvelope* envelopePolygon2 = new OGREnvelope();
			polygon2->getEnvelope(envelopePolygon2);

			if (envelopePolygon2->Intersects(*envelopePolygon1))
			{
				double x_rp = (envelopePolygon1->MinX > envelopePolygon2->MinX) ? envelopePolygon1->MinX : envelopePolygon2->MinX;
				double y_rp = (envelopePolygon1->MinY > envelopePolygon2->MinY) ? envelopePolygon1->MinY : envelopePolygon2->MinY;
				if (x_rp < gridMaxX && x_rp > gridMinX&& y_rp < gridMaxY && y_rp > gridMinY)
				{
					referencePoint++;
					OGRGeometry* pIst = polygon2->Intersection(polygon1);
					OGRwkbGeometryType type = pIst->getGeometryType();
					if (pIst != NULL && (type == 3 || type == 6))
					{
						poDstFeature->SetGeometry(pIst);
						OGRErr fError = poLayer->CreateFeature(poDstFeature);

					}
				}
			}

			delete envelopePolygon2;
			envelopePolygon2 = NULL;
			OGRFeature::DestroyFeature(poFeature2);
		}

		delete envelopePolygon1;
		envelopePolygon1 = NULL;
		OGRFeature::DestroyFeature(poFeature1);
	}

	double gridEnd = clock();
	double gridCost = (double)(gridEnd - gridBegin) / CLOCKS_PER_SEC;

	std::stringstream ssGridNum;
	string strGridNum;
	ssGridNum << count;
	ssGridNum >> strGridNum;
	string index = strRank + "_" + strGridNum + "_dataset1";

	foutSample << index << "\t" << polygonsNum1 << "\t" << polygonsNum2 << "\t" << verticesNum1 << "\t" << verticesNum2 << "\t" << varianceLayer1 << "\t" << varianceLayer2 << "\t" << avgDist << "\t" << referencePoint << "\t" << gridCost << endl;


	string sampleShpPathStr = (string)outImageDir + index + ".shp";
	string sampleImagePathStr = (string)outImageDir + index + ".tif";

	GDALDataset* pSampleDS = poDriver->Create(sampleShpPathStr.c_str(), 0, 0, 0, GDT_Unknown, NULL);
	OGRLayer* poSampleLayer = pSampleDS->CreateLayer(index.c_str(), poSpatialRef, wkbMultiLineString, NULL);
	OGRFeatureDefn* poSampleDefn = poSampleLayer->GetLayerDefn();
	OGRFeature* poSampleFeature = OGRFeature::CreateFeature(poSampleDefn);

	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRGeometry* poGeoPolygon1 = poFeature1->GetGeometryRef();
		OGRGeometry* poGeoLine1 = polygon2polyline(poGeoPolygon1);

		poSampleFeature->SetGeometry(poGeoLine1);
		OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);
		OGRFeature::DestroyFeature(poFeature1);
	}
	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);
		OGRGeometry* poGeoPolygon2 = poFeature2->GetGeometryRef();
		OGRGeometry* poGeoLine2 = polygon2polyline(poGeoPolygon2);

		poSampleFeature->SetGeometry(poGeoLine2);
		OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);
		OGRFeature::DestroyFeature(poFeature2);
	}
	OGRLineString* gridLineString = (OGRLineString*)OGRGeometryFactory::createGeometry(wkbLineString);
	gridLineString->addPoint(gridMinX, gridMinY); gridLineString->addPoint(gridMinX, gridMaxY);
	gridLineString->addPoint(gridMaxX, gridMaxY); gridLineString->addPoint(gridMaxX, gridMinY);
	gridLineString->addPoint(gridMinX, gridMinY);
	poSampleFeature->SetGeometry(gridLineString);
	OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);

	GDALClose(pSampleDS);
	rasterize(sampleShpPathStr.c_str(), sampleImagePathStr.c_str());
	
	return gridCost;
}

double singleGridSampleGeneration(int count, double gridMinX, double gridMaxX, double gridMinY, double gridMaxY, int sumRecord1, int sumRecord2, OGRLayer* poLayer, OGRLayer* poLayer1, OGRLayer *poLayer2, char* outLabelDir, char* outImageDir, string strRank, OGRFeature* poDstFeature, GDALDriver* poDriver, OGRSpatialReference* poSpatialRef, ofstream &foutSample)
{
	OGRFeature* poFeature1, * poFeature2;
	GeoGrid* geoGrid = new GeoGrid;

	for (int FID1 = 0; FID1 < sumRecord1; FID1++)
	{
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRGeometry* poGeo = poFeature1->GetGeometryRef();
		OGRPolygon* polygon = (OGRPolygon*)poGeo;
		if (polygon == NULL)
			continue;
		if (!polygon->IsValid())
			continue;
		OGREnvelope* envelopePolygon = new OGREnvelope();
		polygon->getEnvelope(envelopePolygon);

		OGREnvelope* envelope = new OGREnvelope();
		envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
		envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
		if (envelope->Intersects(*envelopePolygon))
			geoGrid->polygonsFID1.push_back(FID1);

		delete envelope;
		envelope = NULL;
		delete envelopePolygon;
		envelopePolygon = NULL;
		OGRFeature::DestroyFeature(poFeature1);
	}

	for (int FID2 = 0; FID2 < sumRecord2; FID2++)
	{
		poFeature2 = poLayer2->GetFeature(FID2);
		OGRGeometry* poGeo = poFeature2->GetGeometryRef();
		OGRPolygon* polygon = (OGRPolygon*)poGeo;
		if (polygon == NULL)
			continue;
		if (!polygon->IsValid())
			continue;
		OGREnvelope* envelopePolygon = new OGREnvelope();
		polygon->getEnvelope(envelopePolygon);

		OGREnvelope* envelope = new OGREnvelope();
		envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
		envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
		if (envelope->Intersects(*envelopePolygon))
			geoGrid->polygonsFID2.push_back(FID2);

		delete envelope;
		envelope = NULL;
		delete envelopePolygon;
		envelopePolygon = NULL;
		OGRFeature::DestroyFeature(poFeature2);
	}


	int polygonsNum1 = geoGrid->polygonsFID1.size();
	int polygonsNum2 = geoGrid->polygonsFID2.size();
	int verticesNum1 = 0, verticesNum2 = 0;
	double varianceLayer1 = 0.0, varianceLayer2 = 0.0;
	double avgDist;
	int referencePoint = 0;

	double xSumLayer1 = 0, ySumLayer1 = 0;
	double xSumLayer2 = 0, ySumLayer2 = 0;

	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);

		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		verticesNum1 += polygon1->getExteriorRing()->getNumPoints();
		for (int t = 0; t < polygon1->getNumInteriorRings(); t++)
			verticesNum1 += polygon1->getInteriorRing(t)->getNumPoints();
		OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
		xSumLayer1 += poPoint->getX(); ySumLayer1 += poPoint->getY();
	}
	double xAverage1 = xSumLayer1 / polygonsNum1;
	double yAverage1 = ySumLayer1 / polygonsNum1;
	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
		varianceLayer1 = varianceLayer1 + (poPoint->getX() - xAverage1) * (poPoint->getX() - xAverage1) +
			(poPoint->getY() - yAverage1) * (poPoint->getY() - yAverage1);
	}
	varianceLayer1 = varianceLayer1 / polygonsNum1;

	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);

		OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
		verticesNum2 += polygon2->getExteriorRing()->getNumPoints();
		for (int t = 0; t < polygon2->getNumInteriorRings(); t++)
			verticesNum2 += polygon2->getInteriorRing(t)->getNumPoints();
		OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
		xSumLayer2 += poPoint->getX(); ySumLayer2 += poPoint->getY();
	}
	double xAverage2 = xSumLayer2 / polygonsNum2;
	double yAverage2 = ySumLayer2 / polygonsNum2;
	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);
		OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
		OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
		varianceLayer2 = varianceLayer2 + (poPoint->getX() - xAverage2) * (poPoint->getX() - xAverage2) +
			(poPoint->getY() - yAverage2) * (poPoint->getY() - yAverage2);
	}
	varianceLayer2 = varianceLayer2 / polygonsNum2;
	avgDist = (xAverage1 - xAverage2) * (xAverage1 - xAverage2) + (yAverage1 - yAverage2) * (yAverage1 - yAverage2);

	if (polygonsNum1 == 0)varianceLayer1 = 0; if (polygonsNum2 == 0)varianceLayer2 = 0;
	if (polygonsNum1 == 0 || polygonsNum2 == 0)avgDist = 100000;


	double gridBegin = clock();
	//The vectors intersect and get the running time
	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
		OGREnvelope* envelopePolygon1 = new OGREnvelope();
		polygon1->getEnvelope(envelopePolygon1);

		for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
		{
			int FID2 = geoGrid->polygonsFID2[j2];
			poFeature2 = poLayer2->GetFeature(FID2);
			OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
			OGREnvelope* envelopePolygon2 = new OGREnvelope();
			polygon2->getEnvelope(envelopePolygon2);

			if (envelopePolygon2->Intersects(*envelopePolygon1))
			{
				double x_rp = (envelopePolygon1->MinX > envelopePolygon2->MinX) ? envelopePolygon1->MinX : envelopePolygon2->MinX;
				double y_rp = (envelopePolygon1->MinY > envelopePolygon2->MinY) ? envelopePolygon1->MinY : envelopePolygon2->MinY;
				if (x_rp < gridMaxX && x_rp > gridMinX&& y_rp < gridMaxY && y_rp > gridMinY)
				{
					referencePoint++;
					OGRGeometry* pIst = polygon2->Intersection(polygon1);
					OGRwkbGeometryType type = pIst->getGeometryType();
					if (pIst != NULL && (type == 3 || type == 6))
					{
						poDstFeature->SetGeometry(pIst);
						OGRErr fError = poLayer->CreateFeature(poDstFeature);

					}
				}
			}

			delete envelopePolygon2;
			envelopePolygon2 = NULL;
			OGRFeature::DestroyFeature(poFeature2);
		}

		delete envelopePolygon1;
		envelopePolygon1 = NULL;
		OGRFeature::DestroyFeature(poFeature1);
	}

	double gridEnd = clock();
	double gridCost = (double)(gridEnd - gridBegin) / CLOCKS_PER_SEC;

	std::stringstream ssGridNum;
	string strGridNum;
	ssGridNum << count;
	ssGridNum >> strGridNum;
	string index = strRank + "_" + strGridNum + "_dataset1";

	foutSample << index << "\t" << polygonsNum1 << "\t" << polygonsNum2 << "\t" << verticesNum1 << "\t" << verticesNum2 << "\t" << varianceLayer1 << "\t" << varianceLayer2 << "\t" << avgDist << "\t" << referencePoint << "\t" << gridCost << endl;


	string sampleShpPathStr = (string)outImageDir + index + ".shp";
	string sampleImagePathStr = (string)outImageDir + index + ".tif";

	GDALDataset* pSampleDS = poDriver->Create(sampleShpPathStr.c_str(), 0, 0, 0, GDT_Unknown, NULL);
	OGRLayer* poSampleLayer = pSampleDS->CreateLayer(index.c_str(), poSpatialRef, wkbMultiLineString, NULL);
	OGRFeatureDefn* poSampleDefn = poSampleLayer->GetLayerDefn();
	OGRFeature* poSampleFeature = OGRFeature::CreateFeature(poSampleDefn);

	for (int j1 = 0; j1 < geoGrid->polygonsFID1.size(); j1++)
	{
		int FID1 = geoGrid->polygonsFID1[j1];
		poFeature1 = poLayer1->GetFeature(FID1);
		OGRGeometry* poGeoPolygon1 = poFeature1->GetGeometryRef();
		OGRGeometry* poGeoLine1 = polygon2polyline(poGeoPolygon1);

		poSampleFeature->SetGeometry(poGeoLine1);
		OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);
		OGRFeature::DestroyFeature(poFeature1);
	}
	for (int j2 = 0; j2 < geoGrid->polygonsFID2.size(); j2++)
	{
		int FID2 = geoGrid->polygonsFID2[j2];
		poFeature2 = poLayer2->GetFeature(FID2);
		OGRGeometry* poGeoPolygon2 = poFeature2->GetGeometryRef();
		OGRGeometry* poGeoLine2 = polygon2polyline(poGeoPolygon2);

		poSampleFeature->SetGeometry(poGeoLine2);
		OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);
		OGRFeature::DestroyFeature(poFeature2);
	}
	OGRLineString* gridLineString = (OGRLineString*)OGRGeometryFactory::createGeometry(wkbLineString);
	gridLineString->addPoint(gridMinX, gridMinY); gridLineString->addPoint(gridMinX, gridMaxY);
	gridLineString->addPoint(gridMaxX, gridMaxY); gridLineString->addPoint(gridMaxX, gridMinY);
	gridLineString->addPoint(gridMinX, gridMinY);
	poSampleFeature->SetGeometry(gridLineString);
	OGRErr fError = poSampleLayer->CreateFeature(poSampleFeature);

	GDALClose(pSampleDS);
	rasterize(sampleShpPathStr.c_str(), sampleImagePathStr.c_str());

	return gridCost;
}

void parallelIntersectionSampleGeneration1(const char* inPath1, const char* inPath2, char* outLabelDir, char* outImageDir, int gridDimX, int gridDimY)
{
	cout << "Parallel task is running..." << endl;
	double begin = clock();
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Datatype* MPI_Envelope = new MPI_Datatype;
	MPI_Status* status = new MPI_Status;
	MPI_Request request;
	MPI_Type_contiguous(4, MPI_DOUBLE, MPI_Envelope);
	MPI_Type_commit(MPI_Envelope);

	double rank_cost, rank_start, rank_end;
	rank_start = clock();
	GDALAllRegister();
	OGRRegisterAll();

	int sumRecord1 = 0;
	int sumRecord2 = 0;
	OGREnvelope* extent = new OGREnvelope();

	GDALDataset* poDS1, * poDS2;
	CPLSetConfigOption("SHAPE_ENCODING", "");
	poDS1 = (GDALDataset*)GDALOpenEx(inPath1, GDAL_OF_VECTOR, NULL, NULL, NULL);
	poDS2 = (GDALDataset*)GDALOpenEx(inPath2, GDAL_OF_VECTOR, NULL, NULL, NULL);
	if (poDS1 == NULL || poDS2 == NULL)
	{
		printf("Open failed.\n%s");
		return;
	}
	OGRLayer* poLayer1, * poLayer2;
	poLayer1 = poDS1->GetLayer(0);
	poLayer2 = poDS2->GetLayer(0);
	OGRSpatialReference* poSpatialRef = poLayer1->GetSpatialRef();

	if (!rank)
	{
		OGREnvelope* extent1 = new OGREnvelope();
		OGREnvelope* extent2 = new OGREnvelope();

		sumRecord1 = poLayer1->GetFeatureCount();// The number of vecoters in layer1
		poLayer1->GetExtent(extent1);

		sumRecord2 = poLayer2->GetFeatureCount();// The number of vecoters in layer2
		poLayer2->GetExtent(extent2);

		cout << extent1->MaxX << "," << extent1->MinX << "," << extent1->MaxY << "," << extent1->MinY << endl;
		cout << extent2->MaxX << "," << extent2->MinX << "," << extent2->MaxY << "," << extent2->MinY << endl;

		double extentMinX = (extent1->MinX < extent2->MinX) ? extent1->MinX : extent2->MinX;
		double extentMinY = (extent1->MinY < extent2->MinY) ? extent1->MinY : extent2->MinY;
		double extentMaxX = (extent1->MaxX > extent2->MaxX) ? extent1->MaxX : extent2->MaxX;
		double extentMaxY = (extent1->MaxY > extent2->MaxY) ? extent1->MaxY : extent2->MaxY;
		extent->MinX = extentMinX; extent->MaxX = extentMaxX;//Maximum extent of two layers
		extent->MinY = extentMinY; extent->MaxY = extentMaxY;
	}
	MPI_Bcast(&sumRecord1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sumRecord2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(extent, 1, *MPI_Envelope, 0, MPI_COMM_WORLD);

	if (rank)
	{
		std::stringstream ssRank;
		string strRank;
		ssRank << rank;
		ssRank >> strRank;

		string outLabelPath = (string)outLabelDir + "sample_" + strRank + ".txt";
		std::ofstream foutSample(outLabelPath.c_str(), std::ios::out);

		string outShpPath = (string)outLabelDir + "sample_" + strRank + ".shp";
		GDALDriver* poDriver = OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName("ESRI Shapefile");
		if (poDriver == NULL) return;
		GDALDataset* poDstDS = poDriver->Create(outShpPath.c_str(), 0, 0, 0, GDT_Unknown, NULL);
		OGRLayer* poLayer = poDstDS->CreateLayer("Result", poSpatialRef, wkbMultiPolygon, NULL);
		OGRFeatureDefn* poDefn = poLayer->GetLayerDefn();
		OGRFeature* poDstFeature = OGRFeature::CreateFeature(poDefn);

		double extentMinX = extent->MinX; double extentMaxX = extent->MaxX;
		double extentMinY = extent->MinY; double extentMaxY = extent->MaxY;
		double xSize = (extentMaxX - extentMinX) / (double)gridDimX;
		double ySize = (extentMaxY - extentMinY) / (double)gridDimY;

		int averageGridNum = (gridDimX * gridDimY) / (size - 1);
		int startGrid = (rank - 1) * averageGridNum;// The start grid of each process
		int endGrid = (rank - 1 + 1) * averageGridNum;// The end grid of each process
		if (rank == (size - 1)) endGrid = gridDimX * gridDimY;
		
		GeoGrid** geoGrid = new GeoGrid * [endGrid - startGrid];// geoGrid : The vector IDs that each grid contains.
		for (int i = 0; i < (endGrid - startGrid); i++)
			geoGrid[i] = new GeoGrid;
		OGRFeature* poFeature;
		//Divide the vectors from layer1 into grids
		for (int FID = 0; FID < sumRecord1; FID++)
		{
			poFeature = poLayer1->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);

			for (int i = startGrid; i < endGrid; i++) 
			{
				int x_coordinate = i / gridDimX;
				int y_coordinate = i % gridDimX;
				double gridMinX = extentMinX + x_coordinate * xSize;
				double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
				double gridMinY = extentMinY + y_coordinate * ySize;
				double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;

				OGREnvelope* envelope = new OGREnvelope();
				envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
				envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
				if (envelope->Intersects(*envelopePolygon))
					geoGrid[i - startGrid]->polygonsFID1.push_back(FID);
				delete envelope;
				envelope = NULL;
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
			OGRFeature::DestroyFeature(poFeature);
		}
		//Divide the vectors from layer2 into grids
		for (int FID = 0; FID < sumRecord2; FID++)
		{
			poFeature = poLayer2->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);

			for (int i = startGrid; i < endGrid; i++)
			{
				int x_coordinate = i / gridDimX;
				int y_coordinate = i % gridDimX;
				double gridMinX = extentMinX + x_coordinate * xSize;
				double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
				double gridMinY = extentMinY + y_coordinate * ySize;
				double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;

				OGREnvelope* envelope = new OGREnvelope();
				envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
				envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
				if (envelope->Intersects(*envelopePolygon))
					geoGrid[i - startGrid]->polygonsFID2.push_back(FID);
				delete envelope;
				envelope = NULL;
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
			OGRFeature::DestroyFeature(poFeature);
		}

		int count = 0;
		vector<pair <double, double> > supplyGridLfCornerPoints;
		for (int i = startGrid; i < endGrid; i++)
		{
			int x_coordinate = i / gridDimX;
			int y_coordinate = i % gridDimX;
			double gridMinX = extentMinX + x_coordinate * xSize;
			double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
			double gridMinY = extentMinY + y_coordinate * ySize;
			double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;

			double gridCost = singleGridSampleGeneration(count, geoGrid[i - startGrid], gridMinX, gridMaxX, gridMinY, gridMaxY, poLayer, poLayer1, poLayer2, outLabelDir, outImageDir, strRank, poDstFeature, poDriver, poSpatialRef, foutSample);
			count += 1;

			int focalStep = 0;

			double focalSizeX = xSize / 400;
			double focalSizeY = ySize / 400;
			if (gridCost < 2.1) continue;
			if (gridCost >= 2.1 && gridCost < 3.15) focalStep = 1;
			if (gridCost >= 3.15 && gridCost < 4.2) focalStep = 3;
			if (gridCost >= 4.2 && gridCost < 6.3) focalStep = 5;
			if (gridCost >= 6.3 && gridCost < 7.35) focalStep = 7;
			if (gridCost >= 7.35 && gridCost < 8.4) focalStep = 10;
			if (gridCost >= 8.4 && gridCost < 9.45) focalStep = 15;
			if (gridCost >= 9.45) focalStep = 20;
			
			int start = 0 - focalStep; 
			int end = focalStep;
			for (int x = start; x <= end; x++)
			{
				gridMinX = gridMinX + x * focalSizeX;
				gridMaxX = gridMaxX + x * focalSizeX;
				for (int y = start; y <= end; y++)
				{
					gridMinY = gridMinY + y * focalSizeY;
					gridMaxY = gridMaxY + y * focalSizeY;
					supplyGridLfCornerPoints.push_back(make_pair(gridMinX, gridMinY));
				}
				
			}

		}
		
		int supplySum = supplyGridLfCornerPoints.size();
		GeoGrid** supplyGeoGrid = new GeoGrid * [supplySum];
		for (int i = 0; i < supplySum; i++)
			supplyGeoGrid[i] = new GeoGrid;

		cout << rank << ": focal grid sample sum is " << supplySum << " and is been generating..." << endl;	
		for (int FID = 0; FID < sumRecord1; FID++)
		{
			poFeature = poLayer1->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);
			for (int i = 0; i < supplySum; i++)
			{
				double gridMinX = supplyGridLfCornerPoints[i].first;
				double gridMaxX = gridMinX + xSize;
				double gridMinY = supplyGridLfCornerPoints[i].second;
				double gridMaxY = gridMinY + ySize;

				OGREnvelope* envelope = new OGREnvelope();
				envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
				envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
				if (envelope->Intersects(*envelopePolygon))
					supplyGeoGrid[i]->polygonsFID1.push_back(FID);
				delete envelope;
				envelope = NULL;
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
			OGRFeature::DestroyFeature(poFeature);
		}
		for (int FID = 0; FID < sumRecord2; FID++)
		{
			poFeature = poLayer2->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);

			for (int i = 0; i < supplySum; i++)
			{
				double gridMinX = supplyGridLfCornerPoints[i].first;
				double gridMaxX = gridMinX + xSize;
				double gridMinY = supplyGridLfCornerPoints[i].second;
				double gridMaxY = gridMinY + ySize;

				OGREnvelope* envelope = new OGREnvelope();
				envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
				envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
				if (envelope->Intersects(*envelopePolygon))
					supplyGeoGrid[i]->polygonsFID2.push_back(FID);
				delete envelope;
				envelope = NULL;
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
			OGRFeature::DestroyFeature(poFeature);
		}

		for (int i = 0; i < supplySum; i++)
		{
			double gridMinX = supplyGridLfCornerPoints[i].first;
			double gridMaxX = gridMinX + xSize;
			double gridMinY = supplyGridLfCornerPoints[i].second;
			double gridMaxY = gridMinY + ySize;

			double gridCost = singleGridSampleGeneration(count, supplyGeoGrid[i], gridMinX, gridMaxX, gridMinY, gridMaxY,
				poLayer, poLayer1, poLayer2, outLabelDir, outImageDir, strRank, poDstFeature, poDriver, poSpatialRef, foutSample);
			count += 1;
		}
		
		foutSample.close();
		OGRFeature::DestroyFeature(poDstFeature);
		GDALClose(poDstDS);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank)
	{
		removeFile(outImageDir, "shp");
		removeFile(outImageDir, "shx");
		removeFile(outImageDir, "dbf");
		removeFile(outImageDir, "prj");
		double end = clock();
		double cost = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "sum time cost: " << cost << "s" << endl;
	}
	MPI_Finalize();
}
