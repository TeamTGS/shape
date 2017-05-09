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

typedef float PrecisionType;
  
bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames);
void EstimatePCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors);
void ApplyStandardPCA(const vnl_matrix<PrecisionType> &data, vnl_matrix<PrecisionType> &eigenVecs, vnl_vector<PrecisionType> &eigenVals);
void IPCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors);

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
    std::cerr << "Shape Modelling App" << std::endl;
    std::cerr << "Assumes meshes in MVB are Polydata." << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "mvb file" << std::endl;
    std::cerr << "Model Name" << std::endl;
    std::cerr << "Pre-Aligned Meshes (Boolean Optional, Default 1)" << std::endl;
    return EXIT_FAILURE;
    }
  std::string inputFileName = argv[1];
  std::string modelName = argv[2];
  float tolerance = atof(argv[3]);
  int maxIterations = atoi(argv[4]);
  int preAligned = 1;
  if(argc > 5)
    preAligned = atoi(argv[5]);

  std::vector<int> ids;
  std::vector<std::string> filenames;
  std::string fileExtension("mvb");

  if(inputFileName.find(fileExtension) != std::string::npos )
    {
    std::cout << "MVB Extension found " << inputFileName << std::endl;
    bool success = ReadSurfaceFileNames(inputFileName.c_str(), ids, filenames);
    if(success == EXIT_FAILURE)
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
  for(unsigned int i = 0; i < 20/*filenames.size()*/; i++)
  //for (unsigned int i = 0; i < 1; i++)
  {
    ReaderType::Pointer  polyDataReader = ReaderType::New();
    polyDataReader->SetFileName(filenames[i].c_str());
    std::cout << "Adding ID " << count << " " << filenames[i].c_str() << std::endl;
    count++;
    try
      {
      polyDataReader->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
      std::cerr << "Error during Update() " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
      }
    
    MeshType::Pointer mesh = polyDataReader->GetOutput();
    unsigned int numberOfPoints = mesh->GetNumberOfPoints();
    unsigned int numberOfCells  = mesh->GetNumberOfCells();
    // std::cout << "numberOfPoints= " << numberOfPoints << std::endl;
    // std::cout << "numberOfCells= " << numberOfCells << std::endl;
      
    // Retrieve points
	VectorType pointsVector(3 * numberOfPoints); //for each x, y, z values
    for(unsigned int i = 0; i < numberOfPoints; i++)
      {
      PointType pp;
      bool pointExists = mesh->GetPoint(i, &pp);
      if(pointExists) 
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

  unsigned measures = trainingSets[0].size();
 
  //EstimatePCAModelParameters(measures, trainingSets.size(), trainingSets, means, eigenValues, eigenVectors);
  EstimatePCAModelParameters(measures, 10, trainingSets, means, eigenValues, eigenVectors);
  
  //std::cout << "eigenValues: " << eigenValues << std::endl;
  //std::cout << "eigenValues sum: " << eigenValues.sum() << std::endl;
  IPCAModelParameters(measures, trainingSets.size(), trainingSets, means, eigenValues, eigenVectors);
  
  // code here
  std::cout << "Complete" << std::endl;
}
  
bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames)
{
  std::fstream inFile(filename, std::ios::in);
  if(inFile.fail())
  {
    std::cerr << "Cannot read input data file " << filename << std::endl;
    return false;
  }

  std::string bufferKey;
  inFile >> bufferKey;
  std::string szKey = "MILXVIEW_BATCH_FILE";
  if(szKey != bufferKey)
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
  while(inFile >> key >> id >> name)
  {
    if(key == descKey || key == descKey2)
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

void EstimatePCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors)
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

void IPCAModelParameters(unsigned m_NumberOfMeasures, unsigned m_NumberOfTrainingSets, std::vector< vnl_vector<PrecisionType> > m_TrainingSets, vnl_vector<PrecisionType> &m_Means, vnl_vector<PrecisionType> &m_EigenValues, vnl_matrix<PrecisionType> &m_EigenVectors)
{
	vnl_matrix<PrecisionType> D;
	D.set_size(m_NumberOfMeasures, m_NumberOfTrainingSets);
	D.fill(0);
	vnl_matrix<PrecisionType> U = m_EigenVectors;
	vnl_matrix<PrecisionType> UT = m_EigenVectors.transpose();
	vnl_matrix<PrecisionType> a;
	vnl_matrix<PrecisionType> y;
	vnl_matrix<PrecisionType> m;
	m.set_size(m_Means.size(), 1);
	m.set_column(0, m_Means);
	vnl_matrix<PrecisionType> x;
	x.set_size(m_TrainingSets[0].size(), 1);
	vnl_matrix<PrecisionType> r;
	vnl_matrix<PrecisionType> Ud;
	Ud.set_size(m_NumberOfMeasures, m_EigenVectors.cols()+1);
	
	vnl_matrix<PrecisionType> A;
	vnl_matrix<PrecisionType> Ad;
	vnl_matrix<PrecisionType> Anew;

	// Remove the mean from new surface
	for (unsigned int i = 10; i < m_NumberOfTrainingSets ; i++)
	{
		const vnl_vector<PrecisionType> tmpSet = m_TrainingSets[i] - m_Means;
		D.set_column(i, tmpSet);
	}

	//std::cout << "m row " << m.rows() << std::endl;
	//std::cout << "m cols " << m.cols() << std::endl;
	
	for (unsigned int i = 10; i < m_NumberOfTrainingSets; i++)
	{
		// Project new surface from D to current eigenspace
		a = UT * D.get_n_columns(i, 1);

		// Reconstruct new image
		y = m_EigenVectors*a + m;

		// Residual vector
		x.set_column(0, m_TrainingSets[i]);
		r = x - y;
		std::cout << "r norm check " << r.get(0, 0) << std::endl;
			std::cout << r.normalize_columns().get(0,0) << std::endl;

		// Append r as a  new basis vector
		// google search vnl matrix append
		Ud.set_columns(0,m_EigenVectors);
		Ud.set_columns(m_EigenVectors.cols(), r.normalize_columns());

		// New coefficients
		//A.set_size();
		//Ad.set_size(A.rows() + 1, A.cols() + 1);
		//Ad.fill(0);
		// add A,
		//Ad.set

		// Peform PCA, get means, eigenvectors, eigenvalues
		m_Means.set_size(m_NumberOfMeasures);
		m_Means.fill(0);
		for (unsigned int j = 0; j < m_NumberOfTrainingSets; j++)
		{
			m_Means += m_TrainingSets[j];
		}
		m_Means /= (PrecisionType)(m_NumberOfTrainingSets);
		//ApplyStandardPCA(Ad, m_EigenVectors, m_EigenValues);

		// Project the coefficient vectors to new basis
		//Anew = m_EigenVectors.transpose() * (Ad - m_Means)

		// Rotate the subspace
		//m_EigenVectors = Ud * m_EigenVectors

		// Update the mean


		// New eigenvalues

	}
	
	

}

//C:\Users\Alex\Desktop\shape\build\bin\Release\shape_pca_itk.exe C:\aligned.mvb 1 1 1