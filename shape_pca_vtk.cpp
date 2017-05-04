/**
  * Simple testing Code for shape modelling
  *  - Read in a set of landmarked shapes and save as a Robust SSM data file
  */
  
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataCollection.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/algo/vnl_generalized_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
  
bool ReadSurfaceFileNames(const char * filename, std::vector<int> &ids, std::vector<std::string> &filenames);
void ApplyStandardPCA(const vnl_matrix<double> &data, vnl_matrix<double> &eigenVecs, vnl_vector<double> &eigenVals);

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
    
  int count = 0;
  vtkSmartPointer<vtkPolyDataCollection> collection = vtkSmartPointer<vtkPolyDataCollection>::New();
  for(unsigned int i = 0; i < filenames.size(); i++)
  {
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(filenames[i].c_str());
    std::cout << "Adding ID " << count << " " << filenames[i].c_str() << std::endl;
    count++;
    reader->Update();
    std::cout << "    with points " << reader->GetOutput()->GetNumberOfPoints() << std::endl;

    //add to collection
    collection->AddItem(reader->GetOutput());
  }

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

//! Function for applying the PCA of matrices provided using SVD
void ApplyStandardPCA(const vnl_matrix<double> &data, vnl_matrix<double> &eigenVecs, vnl_vector<double> &eigenVals)
{
  const double norm = 1.0/(data.cols()-1);
  const vnl_matrix<double> T = (data.transpose()*data)*norm; //D^T.D is smaller so more efficient

  //SVD
  vnl_svd<double> svd(T); //!< Form Projected Covariance matrix and compute SVD, ZZ^T
  svd.zero_out_absolute(); ///Zero out values below 1e-8 but greater than zero

  ///pinverse unnecessary?
//  eigenVecs = data*vnl_matrix_inverse<double>(svd.U()).pinverse().transpose(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
  eigenVecs = data*svd.U(); //!< Extract eigenvectors from U, noting U = V^T since covariance matrix is real and symmetric
  eigenVecs.normalize_columns();

  eigenVals = svd.W().diagonal();
}
