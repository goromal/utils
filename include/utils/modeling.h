#pragma once

#include <Eigen/Core>
#include <math.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>
//#include <cstdarg>

using namespace Eigen;

namespace modeling {

// ===========================================================================================================
// Fourth order Runge-Kutta integration ======================================================================
// ===========================================================================================================

template <typename xT, typename uT>
class RK4
{
private:
    boost::function<void(xT&, xT&, const uT&)> f_;
    xT k1_;
    xT k2_;
    xT k3_;
    xT k4_;
    xT x2_;
    xT x3_;
    xT x4_;
    bool initialized_;
public:
    RK4() : initialized_(false) {}
    RK4(void (*f)(xT&, xT&, const uT&))
    {
        f_ = boost::bind(f, _1, _2, _3);
        initialized_ = true;
    }
    template <class C>
    RK4(void (C::*f)(xT&, xT&, const uT&), C *obj)
    {
        f_ = boost::bind(f, obj, _1, _2, _3);
        initialized_ = true;
    }
    void setDynamics(void (*f)(xT&, xT&, const uT&))
    {
        f_ = boost::bind(f, _1, _2, _3);
        initialized_ = true;
    }
    template <class C>
    void setDynamics(void (C::*f)(xT&, xT&, const uT&), C *obj)
    {
        f_ = boost::bind(f, obj, _1, _2, _3);
        initialized_ = true;
    }
    void run(xT &x, const uT &u, const double &dt)
    {
        if (initialized_)
        {
            f_(k1_, x, u);

            x2_ = x;
            x2_ += k1_ * (dt/2.0);
            f_(k2_, x2_, u);

            x3_ = x;
            x3_ += k2_ * (dt/2.0);
            f_(k3_, x3_, u);

            x4_ = x;
            x4_ += k3_ * dt;
            f_(k4_, x4_, u);

            x += (k1_ + k2_*2.0 + k3_*2.0 + k4_) * (dt / 6.0);
        }
    }

};

// ===========================================================================================================
// Linear Time Invariant System Simulation ===================================================================
// ===========================================================================================================

class LTIModelSISO
{
private:
    bool initialized_;
    MatrixXd A_;
    MatrixXd B_;
    MatrixXd C_;
    MatrixXd D_;
    MatrixXd x_;
    RK4<MatrixXd, double> rk4_;
    void createStateSpace(MatrixXd &x0, const MatrixXd &alpha_vals, const MatrixXd &beta_vals)
    {
        if (alpha_vals.cols() == 1 && beta_vals.cols() == 1 &&
                alpha_vals.rows() >= beta_vals.rows() && x0.rows() == alpha_vals.rows() - 1)
        {
            rk4_.setDynamics(&LTIModelSISO::f, this);
            int n = x0.rows();
            int m = beta_vals.rows() - 1;
            MatrixXd beta_vals_buffered(n+1, 1);
            for (int i = 0; i <= n; i++)
            {
                if (i <= m)
                    beta_vals_buffered(i, 0) = beta_vals(i, 0);
                else
                    beta_vals_buffered(i, 0) = 0.0;
            }
            x_.resize(n, 1);
            x_ = x0;
            A_.resize(n, n);
            B_.resize(n, 1);
            C_.resize(1, n);
            D_.resize(1, 1);

            D_(0, 0) = beta_vals_buffered(n, 0) / alpha_vals(n, 0);
            B_.setZero();
            B_(n-1, 0) = 1.0 / alpha_vals(n, 0);
            A_.setZero();
            for (int i = 0; i < n; i++)
            {
                C_(0, i) = beta_vals_buffered(i, 0) - beta_vals_buffered(n, 0) * alpha_vals(i, 0) / alpha_vals(n, 0);
                A_(n-1, i) = -alpha_vals(i, 0) / alpha_vals(n, 0);
            }
            for (int i = 1; i < n; i++)
                A_(i-1, i) = 1.0;

            initialized_ = true;
        }
    }
    void f(MatrixXd &x_dot, MatrixXd &x, const double &u)
    {
        x_dot = A_ * x + B_ * u;
    }
public:
    LTIModelSISO() : initialized_(false) {}
    LTIModelSISO(MatrixXd &x0, const MatrixXd &alpha_vals, const MatrixXd &beta_vals)
    {
        createStateSpace(x0, alpha_vals, beta_vals);
    }
    void setCoefficients(MatrixXd &x0, const MatrixXd &alpha_vals, const MatrixXd &beta_vals)
    {
        createStateSpace(x0, alpha_vals, beta_vals);
    }
    double run(const double &u, const double &dt)
    {
        if (initialized_)
        {
            rk4_.run(x_, u, dt);
            return (C_ * x_ + D_ * u)(0,0);
        }
        else
            return 0.0;
    }
    inline MatrixXd A() const { return A_; }
    inline MatrixXd B() const { return B_; }
    inline MatrixXd C() const { return C_; }
    inline MatrixXd D() const { return D_; }
    inline MatrixXd x() const { return x_; }
};

inline std::ostream& operator<< (std::ostream& os, const LTIModelSISO& lms)
{
    os << "SISO-LTI System:\nA =\n" << lms.A() << "\nB =\n" << lms.B() << "\nC =\n" << lms.C()
       << "\nD =\n" << lms.D();
    return os;
}

} // end namespace modeling
