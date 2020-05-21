
#include "ParticleFilter.h"

namespace amcl3d
{
ParticleFilter::ParticleFilter() : generator_(rd_())
{
}

ParticleFilter::~ParticleFilter()
{
}

void ParticleFilter::buildParticlesPoseMsg(geometry_msgs::PoseArray& msg) const
{
  msg.poses.resize(p_.size());

  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    msg.poses[i].position.x = static_cast<double>(p_[i].x);
    msg.poses[i].position.y = static_cast<double>(p_[i].y);
    msg.poses[i].position.z = static_cast<double>(p_[i].z);
    msg.poses[i].orientation.x = 0.;
    msg.poses[i].orientation.y = 0.;
    msg.poses[i].orientation.z = sin(static_cast<double>(p_[i].a * 0.5f));
    msg.poses[i].orientation.w = cos(static_cast<double>(p_[i].a * 0.5f));
  }
}

void ParticleFilter::init(const int num_particles, const Eigen::Vector3d trans,
                          const Eigen::Matrix3d rot, const double t_dev, const double a_dev)
{
  /*  Resize particle set */
  p_.resize(abs(num_particles));

  /*  Sample the given pose */
  const double gauss_const_1 = 1. / (t_dev * sqrt(2 * M_PI));
  const double gauss_const_2 = 1. / (2 * t_dev * t_dev);

  Eigen::Vector3d yawpitchroll = rot.eulerAngles(2, 1, 0); // 2 -z-yaw
  p_[0].x = trans.x();
  p_[0].y = trans.y();
  p_[0].z = trans.z();
  p_[0].a = yawpitchroll[0];
  p_[0].w = gauss_const_1;

  double wt = p_[0].w;
  double dist;

  for (uint32_t i = 1; i < p_.size(); ++i)
  {
    p_[i].x = p_[0].x + ranGaussian(0, t_dev);
    p_[i].y = p_[0].y + ranGaussian(0, t_dev);
    p_[i].z = p_[0].z + ranGaussian(0, t_dev);
    p_[i].a = p_[0].a + ranGaussian(0, a_dev);

    dist = sqrt((p_[i].x - p_[0].x) * (p_[i].x - p_[0].x) + (p_[i].y - p_[0].y) * (p_[i].y - p_[0].y) +
                (p_[i].z - p_[0].z) * (p_[i].z - p_[0].z));

    p_[i].w = gauss_const_1 * exp(-dist * dist * gauss_const_2);

    wt += p_[i].w;
  }

  Particle mean_p;
  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    p_[i].w /= wt;

    mean_p.x += p_[i].w * p_[i].x;
    mean_p.y += p_[i].w * p_[i].y;
    mean_p.z += p_[i].w * p_[i].z;
    mean_p.a += p_[i].w * p_[i].a;
  }
  mean_ = mean_p;

  initialized_ = true;
}

void ParticleFilter::init(const int num_particles, const Eigen::Vector3d trans, const double t_dev)
{
  /*  Resize particle set */
  p_.resize(abs(num_particles));
  p_[0].x = trans.x();
  p_[0].y = trans.y();
  p_[0].z = trans.z();
  for (uint32_t i = 1; i < p_.size(); ++i)
  {
    p_[i].x = p_[0].x + ranGaussian(0, t_dev);
    p_[i].y = p_[0].y + ranGaussian(0, t_dev);
    p_[i].z = p_[0].z + ranGaussian(0, t_dev);
    p_[i].a = 0;
  }
}

void ParticleFilter::predict(const double odom_x_mod, const double odom_y_mod, const double odom_z_mod,
                             const double odom_a_mod, const double delta_x, const double delta_y, const double delta_z,
                             const double delta_a)
{
  const double x_dev = fabs(delta_x * odom_x_mod);
  const double y_dev = fabs(delta_y * odom_y_mod);
  const double z_dev = fabs(delta_z * odom_z_mod);
  const double a_dev = fabs(delta_a * odom_a_mod);

  /*  Make a prediction for all particles according to the odometry */
  double sa, ca, rand_x, rand_y;
  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    sa = sin(p_[i].a);
    ca = cos(p_[i].a);
    rand_x = delta_x + ranGaussian(0, x_dev);
    rand_y = delta_y + ranGaussian(0, y_dev);
    p_[i].x += ca * rand_x - sa * rand_y;
    p_[i].y += sa * rand_x + ca * rand_y;
    p_[i].z += delta_z + ranGaussian(0, z_dev);
    p_[i].a += delta_a + ranGaussian(0, a_dev);
  }
}

void ParticleFilter::predict(Eigen::Matrix3d & delta_R, Eigen::Vector3d & delta_T, const double odom_t_mod, const double odom_r_mod)
{
  Eigen::Vector3d trans_dev = odom_t_mod* delta_T.cwiseAbs();
  Eigen::Vector3d ang_deltaR = delta_R.eulerAngles(2, 1, 0); //2- z-yaw
  double a_dev = fabs(odom_r_mod*ang_deltaR[0]);

  /*  Make a prediction for all particles according to the odometry */
  double sa, ca, rand_x, rand_y;
  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    sa = sin(p_[i].a);
    ca = cos(p_[i].a);
    rand_x = delta_T[0] + ranGaussian(0, trans_dev[0]);
    rand_y = delta_T[1] + ranGaussian(0, trans_dev[1]);
    p_[i].x += ca * rand_x - sa * rand_y;
    p_[i].y += sa * rand_x + ca * rand_y;
    p_[i].z += delta_T[2] + ranGaussian(0, trans_dev[2]);
    p_[i].a += ang_deltaR[0] + ranGaussian(0, a_dev);
  }
}

// void ParticleFilter::update(std::vector<line3d> &lines3d, std::vector<line2d> &lines2d,
//                             Eigen::Matrix3d &K, Eigen::Matrix3d & R,
//                             double &theta, double &threshold)
// {
//   /*  Incorporate measurements */
//   double wtp = 0;

//   Eigen::Vector3d yawpitchroll = R.eulerAngles(2, 1, 0); // 2-z, 1-y, 0-x

//   //test
// 	// Eigen::Matrix3d RfA;
//   // RfA = Eigen::AngleAxisd(yawpitchroll[0], Eigen::Vector3d::UnitZ())
//   //     * Eigen::AngleAxisd(yawpitchroll[1], Eigen::Vector3d::UnitY())
//   //     * Eigen::AngleAxisd(yawpitchroll[2], Eigen::Vector3d::UnitX()); 
// 	// std::cout << "before R=" << R << std::endl;
// 	// std::cout << "yaw=" << yawpitchroll[0] << ", roll=" << yawpitchroll[1] << ", pitch=" << yawpitchroll[2] << std::endl;
// 	// std::cout << "After R=" << RfA << std::endl;

//   clock_t begin_for1 = clock();
//   for (uint32_t i = 0; i < p_.size(); ++i)
//   {
//     /*  Get particle information */
//     Eigen::Vector3d t=Eigen::Vector3d(p_[i].x, p_[i].y, p_[i].z);
//     Eigen::Matrix3d rot;
//     rot = Eigen::AngleAxisd(p_[i].a, Eigen::Vector3d::UnitZ())
//       * Eigen::AngleAxisd(yawpitchroll[1], Eigen::Vector3d::UnitY())
//       * Eigen::AngleAxisd(yawpitchroll[2], Eigen::Vector3d::UnitX()); 

//     /*  Evaluate the weight of the point cloud */
//     /* use particle to obtain 2D-3D correspondences */
//     std::vector<pairsmatch> updatemaches=updatecorrespondence(lines3d, lines2d, K, rot, t, theta, threshold);
//     p_[i].wp = updatemaches.size();

//     /*  Increase the summatory of weights */
//     wtp += p_[i].wp;
//   }
//   clock_t end_for1 = clock();
//   double elapsed_secs = double(end_for1 - begin_for1) / CLOCKS_PER_SEC;
//   ROS_DEBUG("Update time 1: [%lf] sec", elapsed_secs);

//   /*  Normalize all weights */
//   double wt = 0;
//   for (uint32_t i = 0; i < p_.size(); ++i)
//   {
//     if (wtp > 0)
//       p_[i].wp /= wtp;
//     else
//       p_[i].wp = 0;

//     p_[i].w = p_[i].wp;
//     wt += p_[i].w;
//   }
//   //update particles center before resampling
//   // Particle mean_p;
//   // for (uint32_t i = 0; i < p_.size(); ++i)
//   // {
//   //   if (wt > 0)
//   //     p_[i].w /= wt;
//   //   else
//   //     p_[i].w = 0;

//   //   mean_p.x += p_[i].w * p_[i].x;
//   //   mean_p.y += p_[i].w * p_[i].y;
//   //   mean_p.z += p_[i].w * p_[i].z;
//   //   mean_p.a += p_[i].w * p_[i].a;
//   // }
//   // mean_ = mean_p;
// }

std::vector<pairsmatch> ParticleFilter::update(std::vector<line3d> &lines3d, std::vector<line2d> &lines2d,
               Eigen::Matrix3d &K,  Eigen::Matrix3d & R, Eigen::Vector3d & T_updated,
               double &theta,  double &threshold)
{
  int num_matches=0;
  std::vector<pairsmatch> inliers;

  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    /*  Get particle information */
    Eigen::Vector3d t=Eigen::Vector3d(p_[i].x, p_[i].y, p_[i].z);
    /* use particle to obtain 2D-3D correspondences */
    std::vector<pairsmatch> updatemaches=updatecorrespondence(lines3d, lines2d, K, R, t, theta, threshold);
    if(updatemaches.size()>num_matches)
    {
      inliers=updatemaches;
      num_matches=updatemaches.size();
      T_updated=t;
    }
  }
  return inliers;
}

void ParticleFilter::resample(Eigen::Matrix3d & R, Eigen::Vector3d & T)
{
  std::vector<Particle> new_p(p_.size());
  const double factor = 1.f / p_.size();
  const double r = factor * rngUniform(0, 1);
  double c = p_[0].w;
  double u;

  //! Do resamplig
  for (uint32_t m = 0, i = 0; m < p_.size(); ++m)
  {
    u = r + factor * m;
    while (u > c)
    {
      if (++i >= p_.size())
        break;
      c += p_[i].w;
    }
    new_p[m] = p_[i];
    new_p[m].w = factor;
  }

  //! Asign the new particles set
  p_ = new_p;
  //update particles center after resampling
  Particle mean_p;
  for (uint32_t i = 0; i < p_.size(); ++i)
  {
    mean_p.x += p_[i].w * p_[i].x;
    mean_p.y += p_[i].w * p_[i].y;
    mean_p.z += p_[i].w * p_[i].z;
    mean_p.a += p_[i].w * p_[i].a;
  }
  mean_ = mean_p;  
  // return trans and rotation.
  T=Eigen::Vector3d(mean_p.x, mean_p.y, mean_p.z);
  Eigen::Vector3d yawpitchroll = R.eulerAngles(2, 1, 0);
  R = Eigen::AngleAxisd(mean_p.a, Eigen::Vector3d::UnitZ())
      * Eigen::AngleAxisd(yawpitchroll[1], Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(yawpitchroll[2], Eigen::Vector3d::UnitX()); 

}

double ParticleFilter::ranGaussian(const double mean, const double sigma)
{
  std::normal_distribution<double> distribution(mean, sigma);
  return distribution(generator_);
}

double ParticleFilter::rngUniform(const double range_from, const double range_to)
{
  std::uniform_real_distribution<double> distribution(range_from, range_to);
  return distribution(generator_);
}

}  // namespace amcl3d
