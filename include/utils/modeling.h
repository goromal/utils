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
    const boost::function<void(xT&, xT&, const uT&)> f_;
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
    RK4(boost::function<void(xT&, xT&, const uT&)> &f)
    {
        f_ = boost::bind(f, _1, _2, _3);
        initialized_ = true;
    }
    template <class C>
    RK4(boost::function<void(xT&, xT&, const uT&)> &f, C *obj)
    {
        f_ = boost::bind(f, obj, _1, _2, _3);
        initialized_ = true;
    }
    void setDynamics(boost::function<void(xT&, xT&, const uT&)> &f)
    {
        f_ = boost::bind(f, _1, _2, _3);
        initialized_ = true;
    }
    template <class C>
    void setDynamics(boost::function<void(xT&, xT&, const uT&)> &f, C *obj) // VoidConstPtr &obj)
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

} // end namespace modeling
