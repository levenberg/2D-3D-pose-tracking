
#include <mutex>
#include <queue>
#include <thread>
#include "estimator.h"

bool cloud_fusion = true;
std::string IMAGE_TOPIC;
Eigen::Matrix3d Ori_R;
Eigen::Vector3d Ori_T;

queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<sensor_msgs::ImageConstPtr> image_buf;
queue<afm::lines2d::ConstPtr> afmline_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
std::mutex m_buf;
std::mutex m_process;

ros::Publisher pub_godom, pub_pose_visual;
ros::Publisher pub_featimg, pub_matches;
ros::Publisher pub_path;
nav_msgs::Path path;
ros::Publisher pub_pointclouds;

estimator Estimator;
CameraPoseVisualization cameraposevisual(0, 0, 0, 1);
bool valid_pose=false;
bool show_feat=false;
void readParameters(ros::NodeHandle &n)
{
    string config_file, line3d_name, cloud_name;

    n.param("lines_map", line3d_name, std::string(""));
    n.param("cloud_name", cloud_name, std::string(""));
    n.param("config_file", config_file, std::string(""));
    Estimator.setParameters(config_file, line3d_name, cloud_name);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    
    fsSettings["image_topic"] >> IMAGE_TOPIC;  
    fsSettings["show"]>> show_feat;
    
    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    TRACK_RESULT_PATH = OUTPUT_PATH + "/tracking_result.csv";
    std::cout << "result path " << TRACK_RESULT_PATH << std::endl;
    std::ofstream fout(TRACK_RESULT_PATH, std::ios::out);
    fout.close();

    cameraposevisual.setScale(0.5);
    cameraposevisual.setLineWidth(0.05);
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //printf(" image time %f \n", image_msg->header.stamp.toSec());
    if (!cloud_fusion)
        return;
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf(" image time %f \n", image_msg->header.stamp.toSec());
    
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    //ROS_INFO("point_callback!");
    if (!cloud_fusion)
        return;
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}
void afm_line_callback(const afm::lines2d::ConstPtr &afm_line_msg)
{
    //printf(" line time %f \n", afm_line_msg->header.stamp.toSec());
    if (!cloud_fusion)
        return;
    m_buf.lock();
    afmline_buf.push(afm_line_msg);
    m_buf.unlock();
}
void base_pose_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    Ori_R = Quaterniond(odom_msg->pose.pose.orientation.w,
                        odom_msg->pose.pose.orientation.x,
                        odom_msg->pose.pose.orientation.y,
                        odom_msg->pose.pose.orientation.z);
    Ori_T = Vector3d{odom_msg->pose.pose.position.x,
                     odom_msg->pose.pose.position.y,
                     odom_msg->pose.pose.position.z};
    if (valid_pose == false)
    {
        Estimator.loadExtrinsictf(Ori_T, Ori_R);
        valid_pose = true;
    }
}

//vio_callback actually subscribe vio result
void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //printf(" pose time %f \n", pose_msg->header.stamp.toSec());
    if (!cloud_fusion)
        return;
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void pubGodometry(const std_msgs::Header &header)
{
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "body";

    int indx=Estimator.frame_count;
    
    Eigen::Quaterniond Quat(Estimator.R_w[indx]);
    odometry.pose.pose.position.x = Estimator.T_w[indx].x();
    odometry.pose.pose.position.y = Estimator.T_w[indx].y();
    odometry.pose.pose.position.z = Estimator.T_w[indx].z();
    odometry.pose.pose.orientation.x = Quat.x();
    odometry.pose.pose.orientation.y = Quat.y();
    odometry.pose.pose.orientation.z = Quat.z();
    odometry.pose.pose.orientation.w = Quat.w();

    pub_godom.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.pose = odometry.pose.pose;
    path.header = odometry.header;
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);

    Vector3d P = Estimator.T_w[indx] + Estimator.R_w[indx] * Estimator.b2c_T;
    Quaterniond R = Quaterniond(Estimator.R_w[indx] * Estimator.b2c_R);
    cameraposevisual.reset();
    cameraposevisual.add_pose(P, R);
    cameraposevisual.publish_by(pub_pose_visual, odometry.header);
}

void pubFeatureimg(const std_msgs::Header &header)
{
    cv::Mat tmp1_img, temp_img;
    int indx = Estimator.frame_count;
    cv::cvtColor(Estimator.image[indx], temp_img, CV_GRAY2RGB);
    cv::undistort(temp_img,tmp1_img, Estimator.cv_CMatrix, Estimator.cv_dist);
    // tmp1_img=temp_img; 
    cv::Mat tmp2_img= tmp1_img.clone();
    // get transformation from map frame(3d lines) to camera frame (2d lines)
    Eigen::Matrix3d Rot = Estimator.b2c_R.transpose() * Estimator.R_w[indx].transpose();
    Eigen::Vector3d Trans = -Rot* Estimator.T_w[indx] - Estimator.b2c_R.transpose()*Estimator.b2c_T;
    // use vio to get the transformation matrix, with error projection
    Eigen::Matrix3d R_vio = Estimator.b2c_R.transpose() * Estimator.vio_R[indx].transpose();
	Eigen::Vector3d T_vio = -R_vio* Estimator.vio_T[indx] - Estimator.b2c_R.transpose()*Estimator.b2c_T;

    // publish 2D and 3D features in images
    // Note: All the 3D line is too many, only use the matched 3D lines to project.

    for (size_t i = 0; i < Estimator.matches2d3d[indx].size(); i++)
    {
        line2d p_l2d = Estimator.matches2d3d[indx][i].line3dt.transform3D(R_vio, T_vio).project3D(Estimator.K);
        cv::Point2d pt1(p_l2d.ptstart.x(), p_l2d.ptstart.y());
        cv::Point2d pt2(p_l2d.ptend.x(), p_l2d.ptend.y());
        cv::line(tmp1_img, pt1, pt2, cv::Scalar(0, 255, 0), 3);

        line2d l2d = Estimator.matches2d3d[indx][i].line2dt;
        cv::Point2d pt3(l2d.ptstart.x(), l2d.ptstart.y());
        cv::Point2d pt4(l2d.ptend.x(), l2d.ptend.y());
        cv::line(tmp1_img, pt3, pt4, cv::Scalar(0, 0, 255), 2);
    }
    // for (size_t i = 0; i < Estimator.undist_lines2d[indx].size(); i++)
    // {
        
    // }
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", tmp1_img).toImageMsg();
    pub_featimg.publish(msg);

    // cv::Mat tmp2_img;
    // cv::cvtColor(Estimator.image[indx], tmp2_img, CV_GRAY2RGB);
    //publish 2D-3D mathces
    for (size_t i = 0; i < Estimator.matches2d3d[indx].size(); i++)
    {
        pairsmatch match = Estimator.matches2d3d[indx][i];
        cv::Point2d pt1(match.line2dt.ptstart.x(), match.line2dt.ptstart.y());
        cv::Point2d pt2(match.line2dt.ptend.x(), match.line2dt.ptend.y());
        cv::line(tmp2_img, pt1, pt2, cv::Scalar(0, 0, 255), 3);

        line2d p_l2d = match.line3dt.transform3D(Rot, Trans).project3D(Estimator.K);
        cv::Point2d pt3(p_l2d.ptstart.x(), p_l2d.ptstart.y());
        cv::Point2d pt4(p_l2d.ptend.x(), p_l2d.ptend.y());
        cv::line(tmp2_img, pt3, pt4, cv::Scalar(0, 255, 0), 2);
    }
    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(header, "bgr8", tmp2_img).toImageMsg();
    pub_matches.publish(msg2);

}

void pubGpointcloud(const std_msgs::Header &header, vector<Vector3d> &_points)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    int indx=Estimator.frame_count;
    for (unsigned int i = 0; i < _points.size(); i++)
    {
        // transfom local point clouds from body frame to world frame.
        Eigen::Vector3d pt_w = Estimator.R_w[indx] * _points[i] + Estimator.T_w[indx];
        geometry_msgs::Point32 p;
        p.x = pt_w(0);
        p.y = pt_w(1);
        p.z = pt_w(2);
        point_cloud.points.push_back(p);
    }
    pub_pointclouds.publish(point_cloud);
}

void process2d3d()
{
    if (!cloud_fusion)
        return;
    while (true) // how to find the exact image frame with point and pose
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        afm::lines2d::ConstPtr afmline_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        // find out the latest image msg
        m_buf.lock();

        if (!image_buf.empty() &&!afmline_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (afmline_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose becuase no line extracted\n"); // line extraction is slow than vio
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec()&&
            point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() &&
             afmline_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                {
                    pose_buf.pop();
                }
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (afmline_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    afmline_buf.pop();
                afmline_msg = afmline_buf.front();
                afmline_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();
        if (pose_msg != NULL &&valid_pose)
        {
            // printf(" LINE time %f \n", afmline_msg->header.stamp.toSec());
            // printf(" image time %f \n", image_msg->header.stamp.toSec());
            // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1") //gray img
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else //color img
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat image = ptr->image; 

            // read 2d line message
            vector<line2d> lines2d;
            for (size_t i=0; i<afmline_msg->startx.size(); i++)
            {
                line2d lnd(Eigen::Vector4d(afmline_msg->startx[i],afmline_msg->starty[i], afmline_msg->endx[i], afmline_msg->endy[i]));
                lines2d.push_back(lnd);
            }

            Vector3d vio_T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d vio_R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).normalized().toRotationMatrix();

            //reconstructed 3d points in body frame
            vector<Eigen::Vector3d> points;
            for (unsigned int i = 0; i < point_msg->points.size(); i++)
            {
                Eigen::Vector3d pt(point_msg->points[i].x, point_msg->points[i].y, point_msg->points[i].z);
                Eigen::Vector3d pt_b = vio_R.transpose() * (pt - vio_T);
                points.push_back(pt_b);
            }

            Estimator.processImage(pose_msg->header.stamp.toSec(), vio_T, vio_R, image, lines2d);
            pubGodometry(pose_msg->header);
            if (show_feat)
            {
                pubFeatureimg(pose_msg->header);
                pubGpointcloud(pose_msg->header, points);
            }
        }
    }
}

void process3d3d()
{
    if (!cloud_fusion)
        return;
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        afm::lines2d::ConstPtr afmline_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        // find out the latest image msg
        m_buf.lock();

        if (!image_buf.empty() &&!afmline_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (afmline_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose becuase no line extracted\n"); // line extraction is slow than vio
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() &&
                     point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() &&
                     afmline_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                {
                    pose_buf.pop();
                }
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();

                while (afmline_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    afmline_buf.pop();
                afmline_msg = afmline_buf.front();
                afmline_buf.pop();
            }
        }
        m_buf.unlock();
        if (pose_msg != NULL && valid_pose)
        {
            // printf(" LINE time %f \n", afmline_msg->header.stamp.toSec()); //line time is not aligned very accurate
            // printf(" point time %f \n", point_msg->header.stamp.toSec());
            // printf(" image time %f \n", image_msg->header.stamp.toSec());
            // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1") //gray img
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else //color img
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat image = ptr->image;
            //VIO pose
            Vector3d vio_T = Vector3d(pose_msg->pose.pose.position.x,
                                      pose_msg->pose.pose.position.y,
                                      pose_msg->pose.pose.position.z);
            Matrix3d vio_R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                         pose_msg->pose.pose.orientation.x,
                                         pose_msg->pose.pose.orientation.y,
                                         pose_msg->pose.pose.orientation.z)
                                 .normalized()
                                 .toRotationMatrix();

            //reconstructed 3d points in body frame
            vector<Eigen::Vector3d> points;
            for (unsigned int i = 0; i < point_msg->points.size(); i++)
            {
                Eigen::Vector3d pt(point_msg->points[i].x, point_msg->points[i].y, point_msg->points[i].z);
                Eigen::Vector3d pt_b = vio_R.transpose() * (pt - vio_T);
                points.push_back(pt_b);
            }
            Estimator.processPoints(pose_msg->header.stamp.toSec(), vio_T, vio_R, image,  points);
            pubGodometry(pose_msg->header);
            if (show_feat)
            {
                pubFeatureimg(pose_msg->header);
                pubGpointcloud(pose_msg->header, points);
            }
        }
    }
}
void process_fusion()
{
    if (!cloud_fusion)
        return;
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        afm::lines2d::ConstPtr afmline_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        // find out the latest image msg
        m_buf.lock();

        if (!image_buf.empty() &&!afmline_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (afmline_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose becuase no line extracted\n"); // line extraction is slow than vio
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() &&
                     point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() &&
                     afmline_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                {
                    pose_buf.pop();
                }
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();

                while (afmline_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    afmline_buf.pop();
                afmline_msg = afmline_buf.front();
                afmline_buf.pop();
            }
        }
        m_buf.unlock();
        if (pose_msg != NULL && valid_pose)
        {
            // printf(" LINE time %f \n", afmline_msg->header.stamp.toSec()); //line time is not aligned very accurate
            // printf(" point time %f \n", point_msg->header.stamp.toSec());
            // printf(" image time %f \n", image_msg->header.stamp.toSec());
            // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            //image
            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1") //gray img
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else //color img
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat image = ptr->image; 
            //VIO pose
            Vector3d vio_T = Vector3d(pose_msg->pose.pose.position.x,
                                      pose_msg->pose.pose.position.y,
                                      pose_msg->pose.pose.position.z);
            Matrix3d vio_R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                         pose_msg->pose.pose.orientation.x,
                                         pose_msg->pose.pose.orientation.y,
                                         pose_msg->pose.pose.orientation.z)
                                 .normalized()
                                 .toRotationMatrix();
            // read 2d line message
            vector<line2d> lines2d;
            for (size_t i=0; i<afmline_msg->startx.size(); i++)
            {
                line2d lnd(Eigen::Vector4d(afmline_msg->startx[i],afmline_msg->starty[i], afmline_msg->endx[i], afmline_msg->endy[i]));
                lines2d.push_back(lnd);
            }

            //reconstructed 3d points in body frame
            vector<Eigen::Vector3d> points;
            for (unsigned int i = 0; i < point_msg->points.size(); i++)
            {
                Eigen::Vector3d pt(point_msg->points[i].x, point_msg->points[i].y, point_msg->points[i].z);
                Eigen::Vector3d pt_b = vio_R.transpose() * (pt - vio_T);
                points.push_back(pt_b);
            }
            Estimator.processPtsAdLines(pose_msg->header.stamp.toSec(), vio_T, vio_R, image, lines2d, points);
            pubGodometry(pose_msg->header);
            if (show_feat)
            {
                pubFeatureimg(pose_msg->header);
                pubGpointcloud(pose_msg->header, points);
            }
        }
    }
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_fusion");
    ros::NodeHandle n("~");
    readParameters(n);

    pub_godom = n.advertise<nav_msgs::Odometry>("/tracking_node/global_odometry", 1000);
    pub_path = n.advertise<nav_msgs::Path>("/tracking_node/path", 1000);
    pub_featimg = n.advertise<sensor_msgs::Image>("/tracking_node/feat_img",1000);
    pub_matches = n.advertise<sensor_msgs::Image>("/tracking_node/feat_matches",1000);
    pub_pose_visual = n.advertise<visualization_msgs::MarkerArray>("/tracking_node/pose_visual", 1000);
    pub_pointclouds = n.advertise<sensor_msgs::PointCloud>("/tracking_node/point_cloud", 1000);
    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 1000, vio_callback);
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 1000, image_callback);
    ros::Subscriber sub_lineafm=n.subscribe("/Lines2d", 1000, afm_line_callback);
    ros::Subscriber sub_point = n.subscribe("/vins_estimator/point_cloud", 1000, point_callback); // 
    ros::Subscriber sub_basepose=n.subscribe("/benchmark_publisher/base_pose", 1000, base_pose_callback);

    
    std::thread joint_process;

    //only use 2D-3D pose tracking
    // joint_process = std::thread(process2d3d);

    //only use 3D-3D pose tracking
    // joint_process = std::thread(process3d3d);

    // use both 2D-3D and 3D-3D
    joint_process = std::thread(process_fusion);
    // ros::Rate r(20);
    ros::spin();
}
