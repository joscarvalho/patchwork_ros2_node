#define PCL_NO_PRECOMPILE

#include "rclcpp/rclcpp.hpp"
#include "alfa_node/alfa_node.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>

#include "patchwork.hpp"

#include <iostream>
#include <fstream>

using PointTypePatchwork = PointXYZILID;
using namespace std;

#define GROUNDSEG_ID 0

#define PARAMETER_MULTIPLIER 1000

/* These datasets ground truth is found through the RGB values */

#define IITH_GROUND_COLOUR 4294901760 // r: 255; g: 0; b: 0

#define KITTI_GROUND_COLOUR_1 16750335 // r: 255; g: 150; b: 255
#define KITTI_GROUND_COLOUR_2 16711935 // r: 255; g: 0; b: 255
#define KITTI_GROUND_COLOUR_3 49152750 // r: 75; g: 0; b: 75

/* These datasets ground truth is found through an object class / label */

#define PARIS_LILLE_GROUND_CLASS 202020000 // Road only!

alfa_msg::msg::MetricMessage precision;
alfa_msg::msg::MetricMessage recall;
alfa_msg::msg::MetricMessage f1;

vector<float> precision_vec, recall_vec, f1_vec, total_points_vec, removed_vec, tp_vec, fp_vec, tn_vec, fn_vec, htime_vec, ptime_vec;
int frames;

void callback_shutdown()
{
    if(frames > 0)
    {
        std::cout << std::endl << "------ METRICS -------" << std::endl;
        std::cout << "Number of frames: " << frames << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout.precision(4);
        std::cout << "Precision: " << (std::accumulate(precision_vec.begin(), precision_vec.end(), 0.0) / precision_vec.size()) << " %" << std::endl;
        std::cout << "Recall: " << (std::accumulate(recall_vec.begin(), recall_vec.end(), 0.0) / recall_vec.size()) << " %" << std::endl;
        std::cout << "F1-Score: " << (std::accumulate(f1_vec.begin(), f1_vec.end(), 0.0) / f1_vec.size()) << " %" << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "Total points: " << static_cast<int>(std::accumulate(total_points_vec.begin(), total_points_vec.end(), 0.0) / total_points_vec.size()) << std::endl;
        std::cout << "Removed points: " << static_cast<int>(std::accumulate(removed_vec.begin(), removed_vec.end(), 0.0) / removed_vec.size()) << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "Handler Time: " << (std::accumulate(htime_vec.begin(), htime_vec.end(), 0.0) / htime_vec.size()) << " ms" << std::endl;
        std::cout << "Full Processing Time: " << (std::accumulate(ptime_vec.begin(), ptime_vec.end(), 0.0) / ptime_vec.size()) << " ms" << std::endl;
        std::cout << "----------------------" << std::endl;
    }
    else
        std::cout << std::endl << "No frames were processed yet!" << std::endl;
}

void metrics (float _precision, float _recall, int total_points, int removed_points)
{   
    float _f1 = 2 * ( _precision * _recall ) / ( _precision + _recall ); 

    precision_vec.push_back(_precision);
    recall_vec.push_back(_recall);
    f1_vec.push_back(_f1);
    total_points_vec.push_back(static_cast<float>(total_points)); 
    removed_vec.push_back(static_cast<float>(removed_points));
}

void handler (AlfaNode * node)
{
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud;
    //input_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    //pcl::PointCloud<pcl::PointXYZL>::Ptr input_cloud;
    //input_cloud.reset(new pcl::PointCloud<pcl::PointXYZL>);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    //input_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<PointTypePatchwork>::Ptr input_cloud;
    input_cloud.reset(new pcl::PointCloud<PointTypePatchwork>);
    pcl::fromROSMsg(*(node->ros_pointcloud),*input_cloud);

    frames++;
    std::cout << "\r\e[K" << "Point cloud [" << frames << "] received!" << std::flush;

    boost::shared_ptr<PatchWork<PointTypePatchwork> > PatchworkGroundSeg;
    PatchworkGroundSeg.reset(new PatchWork<PointXYZILID>());

    pcl::PointCloud<PointTypePatchwork> pc_ground;
    pcl::PointCloud<PointTypePatchwork> pc_non_ground;
    double time_taken;

    PatchworkGroundSeg->estimate_ground(*input_cloud, pc_ground, pc_non_ground, time_taken);

    // Estimation
    double precision, recall, precision_wo_veg, recall_wo_veg;
    calculate_precision_recall(*input_cloud, pc_ground, precision, recall);
    calculate_precision_recall_without_vegetation(*input_cloud, pc_ground, precision_wo_veg, recall_wo_veg);

    /*
    std::cout << std::fixed << "\n\nth, " << " takes : " << time_taken << " | " 
            << input_cloud->size() << " -> " << pc_ground.size() << "\033[0m" << endl;

    std::cout << "\033[1;32m P: " << precision << " | R: " << recall << "\033[0m" << endl;
    std::cout << "\033[1;32m P: " << precision_wo_veg << " | R: " << recall_wo_veg << "\033[0m" << endl;
    */

    std::uint8_t r = 255, g = 0, b = 0;
    uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
    for (const auto& point : pc_ground.points) // PUBLISH
    {
        pcl::PointXYZRGB p;

        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        p.rgb = *reinterpret_cast<float*>(&rgb);

        node->push_point_output_cloud(p);
    }

    r = 0;
    g = 255;
    rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
    for (const auto& point : pc_non_ground.points) // PUBLISH
    {
        pcl::PointXYZRGB p;

        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        p.rgb = *reinterpret_cast<float*>(&rgb);
        node->push_point_output_cloud(p);
    }

    //metrics(precision, recall, input_cloud->size(), pc_ground.size());
    metrics(precision_wo_veg, recall_wo_veg, input_cloud->size(), pc_ground.size());
}

void post_processing (AlfaNode * node)
{
    node->publish_output_cloud();

    alfa_msg::msg::AlfaMetrics output_metrics;

    output_metrics.metrics.push_back(node->get_handler_time());
    output_metrics.metrics.push_back(node->get_full_processing_time());

    htime_vec.push_back(node->get_handler_time().metric);
    ptime_vec.push_back(node->get_full_processing_time().metric);

    /*
    output_metrics.metrics.push_back(precision);
    output_metrics.metrics.push_back(recall);
    output_metrics.metrics.push_back(f1);
    */
   
   
    node->publish_metrics(output_metrics);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    frames = 0;

    //Prepare the distance filter default configuration
    AlfaExtensionParameter parameters [10];

    /*
    parameters[0].parameter_name = "zeta";     //threshold
    parameters[0].parameter_value = 0.0;       //[m]

    parameters[1].parameter_name = "epsilon";  //threshold
    parameters[1].parameter_value = 0.3;       //[m]

    parameters[2].parameter_name = "delta";    //threshold
    parameters[2].parameter_value = 0.12;      //[m]

    parameters[3].parameter_name = "frac";     //fraction of removed points
    parameters[3].parameter_value = 5.0;

    parameters[4].parameter_name = "csize";    //squared cell size
    parameters[4].parameter_value = 0.3;       //[m]

    precision.units = "%";
    precision.metric_name = "Precision";
    precision.metric = 0;

    recall.units = "%";
    recall.metric_name = "Recall";
    recall.metric = 0;

    f1.units = "%";
    f1.metric_name = "F1-Score";
    f1.metric = 0;
    */

    //Launch Ground Segmentation with:
    std::cout << "Starting Ground Segmentation node with the following characteristics" << std::endl;
    std::cout << "Subscriber topic: /velodyne_points" << std::endl;
    std::cout << "Name of the node: patchwork" << std::endl;
    std::cout << "Parameters: parameter list" << std::endl;
    std::cout << "ID: 0" << std::endl;
    std::cout << "Hardware Driver (SIU): false" << std::endl;
    std::cout << "Hardware Extension: false" << std::endl;
    std::cout << "Distance Resolution: 1 cm" << std::endl;
    std::cout << "Intensity Multiplier: 1x" << std::endl;
    std::cout << "Handler Function: handler" << std::endl;
    std::cout << "Post Processing Function: post_processing" << std::endl << std::endl;

    rclcpp::on_shutdown(&callback_shutdown);
    //rclcpp::spin(std::make_shared<AlfaNode>("/velodyne_points","patchwork", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/cloud_ply","patchwork", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/kitti/point_cloud","patchwork", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    rclcpp::spin(std::make_shared<AlfaNode>("/kitti/velo/pointcloud","patchwork", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    rclcpp::shutdown();
    return 0;
}
