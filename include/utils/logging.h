#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;

namespace logging {

void stdVectorToLog(const std::string &filename, std::vector<double> &data)
{
  std::ofstream f(filename);
  for (int i = 0; i < data.size(); i++)
  {
    f.write((char *) &data[i], sizeof(double));
  }
  f.close();
}

void matrixToLog(const std::string &filename, MatrixXd &matrix)
{
  std::ofstream f(filename);
  int numRows = matrix.rows();
  int numCols = matrix.cols();
  for (int i = 0; i < numCols; i++)
  {
    f.write((char *) (matrix.data() + numRows * i), sizeof(double) * numRows);
  }
  f.close();
}

int logToStdVector(const std::string &filename, std::vector<double> &data)
{
  std::ifstream f(filename);
  double read;
  int size = 0;
  while (!f.fail())
  {
    f.read((char *)&read, sizeof(double));
    if (!f.fail())
    {
      data.push_back(read);
      size++;
    }
  }
  f.close();
  return size;
}

void logToMatrix(const std::string &filename, MatrixXd &matrix, int rowSize)
{
  std::vector<double> data;
  int size = logToStdVector(filename, data);
  int num_cols = size / rowSize;
  matrix.resize(rowSize, num_cols);

  for (int i = 0; i < num_cols; i++)
  {
    int idx = i * rowSize;
    for (int j = 0; j < rowSize; j++)
    {
      matrix(j, i) = data[idx+j];
    }
  }
}


} // end namespace logging
