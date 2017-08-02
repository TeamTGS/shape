/**
* Simple testing Code for shape modelling
*  - Read in a set of landmarked shapes and save as a Robust SSM data file
*/

#include "itkVTKPolyDataReader.h"
#include "itkMesh.h"
#include "itkPointSet.h"

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/algo/vnl_generalized_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>

#include <vector>

#include <iostream>
#include <fstream>

typedef float PrecisionType;

bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames);
void EstimatePCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors, vnl_matrix<PrecisionType> &m_A);
void ApplyStandardPCA(const vnl_matrix<PrecisionType> &data, vnl_matrix<PrecisionType> &eigenVecs, vnl_vector<PrecisionType> &eigenVals);
void IPCAModelParameters(unsigned m_NumberOfMeasures, unsigned batchSize, unsigned eigenvectorSize, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors, vnl_matrix<PrecisionType> &m_A);

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		std::cerr << "Shape Modelling App" << std::endl;
		std::cerr << "Assumes meshes in MVB are Polydata." << std::endl;
		std::cerr << "Usage:" << std::endl;
		std::cerr << "mvb file" << std::endl;
		std::cerr << "BatchPCA Size " << std::endl;
		std::cerr << "Eigenvector Size" << std::endl;
		std::cerr << "trainingSets Size control" << std::endl;
		return EXIT_FAILURE;
	}
	std::string inputFileName = argv[1];
	int batchSize = atoi(argv[2]);
	int eigenvectorSize = atof(argv[3]);
	int trainingSetsSizeControl = atoi(argv[4]);
	//int preAligned = 1;
	//if (argc > 5)
	//	preAligned = atoi(argv[5]);

	std::vector<int> ids;
	std::vector<std::string> filenames;
	std::string fileExtension("mvb");

	if (inputFileName.find(fileExtension) != std::string::npos)
	{
		std::cout << "MVB Extension found " << inputFileName << std::endl;
		bool success = ReadSurfaceFileNames(inputFileName.c_str(), ids, filenames);
		if (success == EXIT_FAILURE)
		{
			std::cout << "Failed to read " << inputFileName.c_str() << std::endl;
			return EXIT_FAILURE;
		}
	}
	else
	{
		std::cout << "InputFile Not an MVB file " << inputFileName.c_str() << std::endl;
		return EXIT_FAILURE;
	}

	typedef itk::Mesh<PrecisionType, 3> MeshType;
	typedef itk::VTKPolyDataReader< MeshType > ReaderType;
	typedef ReaderType::PointType PointType;
	typedef vnl_vector<PrecisionType> VectorType;
	typedef vnl_matrix<PrecisionType> MatrixType;

	int count = 1;
	std::vector<VectorType> trainingSets;
	//for (unsigned int i = 0; i < filenames.size(); i++)
	for (unsigned int i = 0; i < trainingSetsSizeControl; i++)
	{
		ReaderType::Pointer  polyDataReader = ReaderType::New();
		polyDataReader->SetFileName(filenames[i].c_str());
		std::cout << "Adding ID " << count << " " << filenames[i].c_str() << std::endl;
		count++;
		try
		{
			polyDataReader->Update();
		}
		catch (itk::ExceptionObject & excp)
		{
			std::cerr << "Error during Update() " << std::endl;
			std::cerr << excp << std::endl;
			return EXIT_FAILURE;
		}

		MeshType::Pointer mesh = polyDataReader->GetOutput();
		unsigned int numberOfPoints = mesh->GetNumberOfPoints();
		unsigned int numberOfCells = mesh->GetNumberOfCells();
		//std::cout << "numberOfPoints= " << numberOfPoints << std::endl;
		//std::cout << "numberOfCells= " << numberOfCells << std::endl;

		// Retrieve points
		VectorType pointsVector(3 * numberOfPoints); //for each x, y, z values
		for (unsigned int i = 0; i < numberOfPoints; i++)
		{
			PointType pp;
			bool pointExists = mesh->GetPoint(i, &pp);
			if (pointExists)
			{
				//std::cout << "Point is = " << pp << std::endl;
				pointsVector[(i * 3)] = pp[0];
				pointsVector[(i * 3) + 1] = pp[1];
				pointsVector[(i * 3) + 2] = pp[2];
			}
		}
		trainingSets.push_back(pointsVector);
	}

	//Compute model
	VectorType    means;
	MatrixType    eigenVectors;
	VectorType    eigenValues;
	MatrixType	  A;

	unsigned measures = trainingSets[0].size();

	// start with selected number images

	if (batchSize > trainingSets.size())
	{
		std::cerr << "batch size could not be larger than training set" << std::endl;
		return EXIT_FAILURE;
	}
	EstimatePCAModelParameters(measures, batchSize, trainingSets, means, eigenValues, eigenVectors, A);

	//std::cout << "eigenValues: " << eigenValues << std::endl;
	//std::cout << "eigenValues sum: " << eigenValues.sum() << std::endl;

	// IPCA the rest of the surfaces
	// 10 to end
	IPCAModelParameters(measures, batchSize, eigenvectorSize, trainingSets.size(), trainingSets, means, eigenValues, eigenVectors, A);
	std::cout << "eigenValues: " << eigenValues << std::endl;
	//std::cout << "eigenValues sum: " << eigenValues.sum() << std::endl;

	std::ofstream myfile;
	myfile.open("C:\\Users\\Alex\\Desktop\\eigenvalue.csv");
	myfile << "Mode,Value,\n";
	for(int i = 0; i < trainingSets.size(); i++)
	{ 
		myfile << i+1 << "," << eigenValues.get(i) << "\n";
	}
	myfile.close();
	std::cout << "Complete" << std::endl;
}

bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames)
{
	std::fstream inFile(filename, std::ios::in);
	if (inFile.fail())
	{
		std::cerr << "Cannot read input data file " << filename << std::endl;
		return false;
	}

	std::string bufferKey;
	inFile >> bufferKey;
	std::string szKey = "MILXVIEW_BATCH_FILE";
	if (szKey != bufferKey)
	{
		std::cerr << "Invalid input data file " << filename << std::endl;
		std::cerr << bufferKey << std::endl;
		std::cerr << szKey << std::endl;
		return false;
	}

	std::string key, name;
	std::string descKey = "CASE_SURFACE";
	std::string descKey2 = "CASE_IMAGE";
	int id;
	while (inFile >> key >> id >> name)
	{
		if (key == descKey || key == descKey2)
		{
			ids.push_back(id);
			filenames.push_back(name);
			std::cout << id << " " << name << " " << std::endl;
		}
		else
		{
			std::cerr << "Seems to have incorrect data" << std::endl;
			return false;
		}
	}
	inFile.close();
	std::cout << "ReadImageFileNames Finished" << std::endl;

	return EXIT_SUCCESS;
}

void EstimatePCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors, vnl_matrix<PrecisionType> &m_A)
{
	//-------------------------------------------------------------------------
	//Calculate the Means
	//-------------------------------------------------------------------------
	//std::cout << "PCAModelEstimator" << m_NumberOfMeasures << std::endl;
	m_Means.set_size(m_NumberOfMeasures);
	m_Means.fill(0);
	for (unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
	{
		m_Means += m_TrainingSets[i];
	}
	m_Means /= (PrecisionType)(m_NumberOfTrainingSets);
	//std::cout << "PCAModelEstimator: Mean Performed " << m_Means.size() << " " << m_NumberOfTrainingSets << std::endl;
	//std::cout << "PCAModelEstimator: Make D" << std::endl;
	vnl_matrix<PrecisionType> D, D_Weighted;
	D.set_size(m_NumberOfMeasures, m_NumberOfTrainingSets);
	D.fill(0);

	// remove mean and make matrix
	for (unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
	{
		const vnl_vector<PrecisionType> tmpSet = m_TrainingSets[i] - m_Means;
		D.set_column(i, tmpSet);
	}

	m_EigenValues.set_size(m_NumberOfTrainingSets);
	m_EigenVectors.set_size(m_NumberOfMeasures, m_NumberOfTrainingSets);
	//std::cout << "PCAModelEstimator: D Performed " << D.rows() << "x" << D.columns() << std::endl;

	ApplyStandardPCA(D, m_EigenVectors, m_EigenValues);

	m_A.set_size(m_NumberOfTrainingSets, m_NumberOfTrainingSets);
	for (unsigned int i = 0; i < m_NumberOfTrainingSets; i++)
	{
		m_A.set_column(i, m_EigenVectors.transpose() * D.get_column(i));
	}
}

//! Function for applying the PCA of matrices provided using SVD
void ApplyStandardPCA(const vnl_matrix<PrecisionType> &data, vnl_matrix<PrecisionType> &eigenVecs, vnl_vector<PrecisionType> &eigenVals)
{
	const PrecisionType norm = 1.0 / (data.cols() - 1);
	const vnl_matrix<PrecisionType> T = (data.transpose()*data)*norm; //D^T.D is smaller so more efficient

	//SVD
	vnl_svd<PrecisionType> svd(T); //!< Form Projected Covariance matrix and compute SVD, ZZ^T
	svd.zero_out_absolute(); ///Zero out values below 1e-8 but greater than zero

	///pinverse unnecessary?
	//  eigenVecs = data*vnl_matrix_inverse<double>(svd.U()).pinverse().transpose(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
	eigenVecs = data*svd.U(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
	eigenVecs.normalize_columns();

	eigenVals = svd.W().diagonal();
}

void IPCAModelParameters(unsigned m_NumberOfMeasures, unsigned batchSize, unsigned eigenvectorSize, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors, vnl_matrix<PrecisionType> &m_A)
{
	// n x m rols x cols
	// m_NumberOfMeasures:		 69138
	// m_NumberOfTrainingSets:	 20 (depends on .mvb file)
	// m_TrainingSets:			 (vector) 69138 x 20, whole set of surface
	// m_Means:					 (vector) 69138
	// m_EigenValues:			 (vector) initially 10 from Batch PCA
	// m_EigenVectors:			 (matrix) initially 69138 x 10 from Batch PCA
	// m_A:						 (matrix) coefficient, initially 10 x 10 
	
	//vnl_matrix<PrecisionType> D;
	//D.set_size(m_NumberOfMeasures, m_NumberOfTrainingSets);
	//D.fill(0);
	//vnl_matrix<PrecisionType> U = m_EigenVectors; // 69138 x 10
	vnl_matrix<PrecisionType> UT; // 10 x 69138
	vnl_matrix<PrecisionType> x;
	vnl_matrix<PrecisionType> m;
	vnl_matrix<PrecisionType> tmpSet;
	vnl_matrix<PrecisionType> a;
	vnl_matrix<PrecisionType> y;
	vnl_matrix<PrecisionType> r;
	vnl_matrix<PrecisionType> Ud;
	vnl_matrix<PrecisionType> Udd;
	vnl_vector<PrecisionType> lamdadd;
	vnl_matrix<PrecisionType> Ad;
	vnl_matrix<PrecisionType> Anew;
	vnl_matrix<PrecisionType> rn;
	vnl_vector<PrecisionType> udd;
	vnl_matrix<PrecisionType> tmpAd;

	// start from batchSize to the end of the sets
	for (unsigned int i = batchSize; i < m_NumberOfTrainingSets; i++)
	{
		std::cout << "ipca loop" << std::endl;
		// 1. Project new surface from D to current eigenspace, a = UT(x-mean)
		UT = m_EigenVectors.transpose();
		x.set_size(m_NumberOfMeasures, 1); // 69138 x 1
		x.set_column(0, m_TrainingSets[i]); // set first column as current image
		tmpSet.set_size(m_NumberOfMeasures, 1); // 69138 x 1
		tmpSet.set_column(0, (m_TrainingSets[i] - m_Means)); // remove mean from current image
		a.set_size(i,1); // i x 1
		a.set_columns(0, UT * tmpSet); // project to eigenspace

		// 2. Reconstruct new image, y = U a + mean
		m.set_size(m_Means.size(), 1);
		m.set_column(0, m_Means);
		y = m_EigenVectors*a + m;

		// 3. Compute the residual vector, r is orthogonal to U
		r = x - y;
		std::ofstream myfile("C:\\r.csv", std::ios_base::app | std::ios_base::out);
		myfile << r.get_column(0) << "\n";
		myfile.close();

		// 4. Append r as a  new basis vector
		Ud.set_size(m_NumberOfMeasures, m_EigenVectors.cols() + 1);
		Ud.fill(0);
		Ud.set_columns(0, m_EigenVectors);
		rn = r.normalize_columns();
		Ud.set_columns(Ud.cols() - 1, rn);

		// 5. New coefficients
		Ad.set_size(m_A.rows() + 1, m_A.cols() + 1); // i+1 x i+1
		Ad.fill(0);
		// add A
		Ad.update(m_A, 0, 0);
		// add a
		Ad.update(a, 0, Ad.cols() - 1);
		// add ||r||
		/*double r_mag = 0;
		for (unsigned int j = 0; j < r.size(); j++)
		{
			r_mag += (r.get(j, 0)*r.get(j, 0));
		}
		r_mag = sqrt(r_mag);
*/
		// #1 r_mag
		//Ad.put(Ad.rows() - 1, Ad.cols() - 1, r_mag);
		//std::cout << "r_mag: " << r_mag << std::endl;
		// #2 r.array_two_norm()
		Ad.put(Ad.rows() - 1, Ad.cols() - 1, sqrt(r.array_two_norm()));

		// 6. Perform PCA on Ad
		// udd is mean of Ad, one column, Ad rows
		udd.set_size(Ad.cols());
		udd.fill(0);
		for (unsigned int k = 0; k < Ad.cols(); k++)
		{
			udd += Ad.get_column(k);
		}
		udd /= (PrecisionType)(Ad.cols());
		for (unsigned int l = 0; l < Ad.cols(); l++)
		{
			const vnl_vector<PrecisionType> tmpSet = Ad.get_column(l) - udd;
			Ad.set_column(l, tmpSet);
		}
		ApplyStandardPCA(Ad, Udd, lamdadd);

		// 7. Project the coefficient vectors to new basis
		// remove means from all columns of Ad
		// Ad size: i+1 x i+1 (i start at batchsize)
		/*tmpAd.set_size(Ad.rows(), Ad.cols());
		for (unsigned int m = 0; m < i; m++)
		{
			tmpAd.set_column(m, Ad.get_column(m) - udd);
		}*/
		m_A = Udd.transpose() * Ad;
		//std::cout << m_A << std::endl;
		// 8. Rotate the subspace
		m_EigenVectors = Ud * Udd;

		// 9. Update the mean
		m_Means = m_Means + Ud * udd;

		// 10. New eigenvalues
		m_EigenValues = lamdadd;
		
		
	}
	// trim eigenvectorSize if needed
	// trim 
}

//C:\Users\Alex\Desktop\shape\build\bin\Release\shape_pca_itk.exe C:\aligned.mvb 10 20
