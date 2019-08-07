#pragma once

#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include <random>
#include <experimental/filesystem>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <vector>

#define UTILS_PI 3.141592653589793

namespace utils {

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;

typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

template <typename T>
bool get_yaml_node(const std::string key, const std::string filename, T& val, bool print_error = true)
{
  // Try to load the YAML file
  YAML::Node node;
  try
  {
    node = YAML::LoadFile(filename);
  }
  catch (...)
  {
    std::cout << "Failed to Read yaml file " << filename << std::endl;
  }

  // Throw error if unable to load a parameter
  if (node[key])
  {
    val = node[key].as<T>();
    return true;
  }
  else
  {
    if (print_error)
    {
      throw std::runtime_error("Unable to load " + key + " from " + filename);
    }
    return false;
  }
}
template <typename Derived1>
bool get_yaml_eigen(const std::string key, const std::string filename, Eigen::MatrixBase<Derived1>& val, bool print_error=true)
{
  YAML::Node node = YAML::LoadFile(filename);
  std::vector<double> vec;
  if (node[key])
  {
    vec = node[key].as<std::vector<double>>();
    if (vec.size() == (val.rows() * val.cols()))
    {
      int k = 0;
      for (int i = 0; i < val.rows(); i++)
      {
        for (int j = 0; j < val.cols(); j++)
        {
          val(i,j) = vec[k++];
        }
      }
      return true;
    }
    else
    {
      throw std::runtime_error("Eigen Matrix Size does not match parameter size for " + key + " in " + filename +
                               ". Requested " + std::to_string(Derived1::RowsAtCompileTime) + "x" + std::to_string(Derived1::ColsAtCompileTime) +
                               ", Found " + std::to_string(vec.size()));
      return false;
    }
  }
  else if (print_error)
  {
    throw std::runtime_error("Unable to load " + key + " from " + filename);
  }
  return false;
}

template <typename Derived>
bool get_yaml_diag(const std::string key, const std::string filename, Eigen::MatrixBase<Derived>& val, bool print_error=true)
{
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> diag;
  if (get_yaml_eigen(key, filename, diag, print_error))
  {
    val = diag.asDiagonal();
    return true;
  }
  return false;
}

template <typename T>
bool get_yaml_priority(const std::string key, const std::string file1, const std::string file2, T& val)
{
  if (get_yaml_node(key, file1, val, false))
  {
    return true;
  }
  else
  {
    return get_yaml_node(key, file2, val, true);
  }
}

template <typename Derived1>
bool get_yaml_priority_eigen(const std::string key, const std::string file1, const std::string file2, Eigen::MatrixBase<Derived1>& val)
{
  if (get_yaml_eigen(key, file1, val, false))
  {
    return true;
  }
  else
  {
    return get_yaml_eigen(key, file2, val, true);
  }
}

inline bool createDirIfNotExist(const std::string& dir)
{
  if(!std::experimental::filesystem::exists(dir))
    return std::experimental::filesystem::create_directory(dir);
  else
    return false;
}

inline std::vector<std::string> split(const std::string& s, const char* delimeter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimeter[0]))
   {
      tokens.push_back(token);
   }
   return tokens;
}

inline std::string baseName(const std::string& path)
{
  std::string filename = split(path, "/").back();
  return split(filename, ".")[0];
}

// LEGACY skew
inline Eigen::Matrix3d skew(const Eigen::Vector3d v)
{
  Eigen::Matrix3d mat;
  mat << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
  return mat;
}


static const Eigen::Matrix<double, 2, 3> I_2x3 = [] {
  Eigen::Matrix<double, 2, 3> tmp;
  tmp << 1.0, 0, 0,
         0, 1.0, 0;
  return tmp;
}();

static const Eigen::Matrix3d I_3x3 = [] {
  Eigen::Matrix3d tmp = Eigen::Matrix3d::Identity();
  return tmp;
}();

static const Eigen::Matrix2d I_2x2 = [] {
  Eigen::Matrix2d tmp = Eigen::Matrix2d::Identity();
  return tmp;
}();


static const Eigen::Vector3d e_x = [] {
  Eigen::Vector3d tmp;
  tmp << 1.0, 0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_y = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 1.0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_z = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 1.0;
  return tmp;
}();

template <typename T>
Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,1>& v)
{
  Eigen::Matrix<T,3,3> mat;
  mat << (T)0.0, -v(2), v(1),
         v(2), (T)0.0, -v(0),
         -v(1), v(0), (T)0.0;
  return mat;
}

template <typename Derived>
void setNormalRandom(Eigen::MatrixBase<Derived>& M, std::normal_distribution<double>& N, std::default_random_engine& g)
{
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = 0; j < M.cols(); j++)
    {
      M(i,j) = N(g);
    }
  }
}

template <typename T, int R, int C>
Eigen::Matrix<T, R, C> randomNormal(std::normal_distribution<T>& N, std::default_random_engine& g)
{
  Eigen::Matrix<T,R,C> out;
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      out(i,j) = N(g);
    }
  }
  return out;
}

template <typename T, int R, int C>
Eigen::Matrix<T, R, C> randomUniform(std::uniform_real_distribution<T>& N, std::default_random_engine& g)
{
  Eigen::Matrix<T,R,C> out;
  for (int i = 0; i < R; i++)
  {
    for (int j = 0; j < C; j++)
    {
      out(i,j) = N(g);
    }
  }
  return out;
}

template <typename T>
int sign(T in)
{
  return (in >= 0) - (in < 0);
}

template <typename T>
inline T random(T max, T min)
{
  T f = (T)rand() / RAND_MAX;
  return min + f * (max - min);
}

template <typename T>
class dirtyDerivative
{
private:
  double sigma_;
  bool initialized;
  T deriv_curr_;
  T deriv_prev_;
  T val_prev_;
public:
  dirtyDerivative(void) : sigma_(0.05)
  {
    resetCalculator();
  }
  dirtyDerivative(const double sigma)
  {
    sigma_ = sigma;
    resetCalculator();
  }
  void resetCalculator()
  {
    deriv_curr_ = (T) 0.0;
    deriv_prev_ = (T) 0.0;
    val_prev_ = (T) 0.0;
    initialized = false;
  }
  T calculate(const T &val, const double Ts)
  {
    if (initialized)
    {
      deriv_curr_ = (2 * sigma_ - Ts) / (2 * sigma_ + Ts) * deriv_prev_ +
                    2 / (2 * sigma_ + Ts) * (val - val_prev_);
      deriv_prev_ = deriv_curr_;
      val_prev_ = val;
    }
    else
    {
      deriv_curr_ = (T) 0.0;
      deriv_prev_ = deriv_curr_;
      val_prev_ = val;
      initialized = true;
    }
    return deriv_curr_;
  }
};
typedef dirtyDerivative<double> dirtyDerivatived;

template <typename T>
class dirtyDerivativeMat
{
private:
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatXT;
  std::vector<std::vector<dirtyDerivative<T>>> derivatives;
  MatXT ddirs_;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  dirtyDerivativeMat(void)
  {
    for (int i = 0; i < 2; i++)
    {
      std::vector<dirtyDerivative<T>> row;
      derivatives.push_back(row);
      for (int j = 0; j < 1; j++)
      {
        derivatives[i].push_back(dirtyDerivative<T>());
      }
    }
    ddirs_ = MatXT::Zero(2, 1);
  }
  dirtyDerivativeMat(const MatXT &sigmas)
  {
    for (int i = 0; i < sigmas.rows(); i++)
    {
      std::vector<dirtyDerivative<T>> row;
      derivatives.push_back(row);
      for (int j = 0; j < sigmas.cols(); j++)
      {
        derivatives[i].push_back(dirtyDerivative<T>(sigmas(i, j)));
      }
    }
    ddirs_ = MatXT::Zero(sigmas.rows(), sigmas.cols());
  }
  MatXT calculate(const MatXT &val, const double Ts)
  {
    for (int i = 0; i < val.rows(); i++)
    {
      for (int j = 0; j <val.cols(); j++)
      {
        ddirs_(i, j) = derivatives[i][j].calculate(val(i, j), Ts);
      }
    }
    return ddirs_;
  }
};
typedef dirtyDerivativeMat<double> dirtyDerivativeMatd;

template<typename T>
inline T wrap_angle_npi2pi(T theta)
{
  T ret_theta = theta;
  if (theta > UTILS_PI)
  {
    ret_theta -= 2 * UTILS_PI;
  }
  else if (theta <= -UTILS_PI)
  {
    ret_theta += 2 * UTILS_PI;
  }
  return ret_theta;
}

} // end namespace utils
