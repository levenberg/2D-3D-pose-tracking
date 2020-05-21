
#pragma once

#include <random>

#include <ros/ros.h>

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "line.h"

namespace amcl3d
{
/*! \brief Struct that contains the data concerning one particle.
 */
struct Particle
{
  double x; /*!< Position x */
  double y; /*!< Position y */
  double z; /*!< Position z */
  double a; /*!< Yaw angle */

  double w;  /*!< Total weight */
  double wp; /*!< Weight by the 3d point cloud sensor */

  Particle() : x(0), y(0), z(0), a(0), w(0), wp(0)
  {
  }
};

/*! \brief Class that contains the stages of the Particle Filter.
 */
class ParticleFilter
{
public:
  /*! \brief ParticleFilter class constructor.
   */
  explicit ParticleFilter();
  /*! \brief ParticleFilter class destructor.
   */
  virtual ~ParticleFilter();

  /*! \brief To inicialite the grid map.
   *
   * \return <b>bool=False</b> - If it has not been initialized.
   * \return <b>bool=True</b> - If it has been initialized.
   *
   * It only return the variable initialized_, and this is modified in the code when the filter does the
   * ParticleFilter::init method.
   */
  bool isInitialized() const
  {
    return initialized_;
  }

  /*! \brief To get the information from the Particle struct.
   *
   * \return Particle - Particle struct.
   */
  Particle getMean() const
  {
    return mean_;
  }

  /*! \brief To build the particles pose message.
   *
   * \param msg Type of message that it is wanted to build.
   */
  void buildParticlesPoseMsg(geometry_msgs::PoseArray& msg) const;

  /*! \brief This function implements the PF init stage.
   *
   * \param num_particles Particle number in the filter.
   * \param trans Init x, y, z-axis position.
   * \param rot Init orientation.
   * \param t_dev Init thresholds of x,y,z-axis position.
   * \param a_dev Init thresholds of yaw angle orientation.
   *
   * It restructures the particle vector to adapt it to the number of selected particles. Subsequently, it initializes
   * it using a Gaussian distribution and the deviation introduced. Subsequently, it calculates what would be the
   * average particle that would simulate the estimated position of the UAV.
   */
  void init(const int num_particles, const Eigen::Vector3d trans,
            const Eigen::Matrix3d rot, const double t_dev, const double a_dev);
  void init(const int num_particles, const Eigen::Vector3d trans, const double t_dev);

      /*! \brief This function implements the PF prediction stage.
   * (Translation in X, Y and Z in meters and yaw angle incremenet in rad.)
   *
   * \param odom_x_mod Thresholds of x-axis position in prediction.
   * \param odom_y_mod Thresholds of y-axis position in prediction.
   * \param odom_z_mod Thresholds of z-axis position in prediction.
   * \param odom_a_mod Thresholds of yaw angle orientation in prediction.
   * \param delta_x Increased odometry in the position of the x-axis.
   * \param delta_y Increased odometry in the position of the x-axis.
   * \param delta_z Increased odometry in the position of the x-axis.
   * \param delta_a Increased odometry in the position of the x-axis.
   *
   * It calculates the increase that has occurred in the odometry and makes predictions of where it is possible that the
   * UAV is, taking into account selected thresholds.
   */
      void predict(const double odom_x_mod, const double odom_y_mod, const double odom_z_mod, const double odom_a_mod,
                   const double delta_x, const double delta_y, const double delta_z, const double delta_a);

  /*! \brief This function implements the PF prediction stage.
   *
   * \param delta_R increased odometry in rotation.
   * \param delta_T increased odometry in translation.
   * \param odom_mod adding Guassian nose threshold of poistion in direction.
   *
   * It takes the positions of the particles to change if they are on the map. Then, it evaluates the weight of the
   * particle according to the 2D-3D correspondences. Finally, it normalizes the weights
   * for all particles and finds the average for the composition of the UAV pose.
   */
  void predict(Eigen::Matrix3d & delta_R, Eigen::Vector3d & delta_T, const double odom_t_mod, const double odom_r_mod);

  /*! \brief This function implements the PF update stage.
   *
   * \param lines3d 3D lines from point cloud map.
   * \param lines2d 2D lines from individual images.
   * \param threshold threshold for the 2D-3D matching inliers.
   *
   * It takes the positions of the particles to change if they are on the map. Then, it evaluates the weight of the
   * particle according to the 2D-3D correspondences. Finally, it normalizes the weights
   * for all particles and finds the average for the composition of the UAV pose.
   */
  // void update(std::vector<line3d> &lines3d, std::vector<line2d> &lines2d,
  //              Eigen::Matrix3d &K,  Eigen::Matrix3d & R, 
  //              double &theta,  double &threshold);
  std::vector<pairsmatch> update(std::vector<line3d> &lines3d, std::vector<line2d> &lines2d,
               Eigen::Matrix3d &K,  Eigen::Matrix3d & R, Eigen::Vector3d & T_updated,
               double &theta,  double &threshold);

  /*! \brief This function implements the PF resample stage.
   * Translation in X, Y and Z in meters and yaw angle incremenet in rad.
   *
   * \param num_particles Particle number in the filter.
   * \param x_dev Init thresholds of x-axis position.
   * \param y_dev Init thresholds of y-axis position.
   * \param z_dev Init thresholds of z-axis position.
   * \param a_dev Init thresholds of yaw angle orientation.
   *
   * Sample the particle set again using low variance samples. So that the particles with less weights are discarded. To
   * complete the number of particles that the filter must have, new ones are introduced taking the average of those
   * that passed the resampling and applying the same variance thresholds that is applied in the prediction.
   */
  void resample(Eigen::Matrix3d & R, Eigen::Vector3d & T);

private:

  /*! \brief To generate the random value by the Gaussian distribution.
   *
   * \param mean Average of the distribution.
   * \param sigma Desviation of the distribution.
   * \return <b>double</b> - Random value.
   */
  double ranGaussian(const double mean, const double sigma);

  /*! \brief To generate the random between two values.
   *
   * \param range_from Lower end of range.
   * \param range_to Upper end of range.
   * \return <b>double</b> - Random value.
   */
  double rngUniform(const double range_from, const double range_to);

  bool initialized_{ false }; /*!< To indicate the initialition of the filter */

  std::vector<Particle> p_; /*!< Vector of particles */
  Particle mean_;           /*!< Particle to show the mean of all the particles */

  std::random_device rd_;  /*!< Random device */
  std::mt19937 generator_; /*!< Generator of random values */
};


}  // namespace amcl3d
