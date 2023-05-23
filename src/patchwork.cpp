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

alfa_msg::msg::MetricMessage true_positive_rate;
alfa_msg::msg::MetricMessage true_negative_rate;
alfa_msg::msg::MetricMessage positive_predictive_value;
alfa_msg::msg::MetricMessage negative_predictive_value;
alfa_msg::msg::MetricMessage f1;
alfa_msg::msg::MetricMessage true_positive_rate_mean;
alfa_msg::msg::MetricMessage true_negative_rate_mean;
alfa_msg::msg::MetricMessage positive_predictive_value_mean;
alfa_msg::msg::MetricMessage negative_predictive_value_mean;
alfa_msg::msg::MetricMessage f1_mean;
alfa_msg::msg::MetricMessage true_positive_rate_dev;
alfa_msg::msg::MetricMessage true_negative_rate_dev;
alfa_msg::msg::MetricMessage positive_predictive_value_dev;
alfa_msg::msg::MetricMessage negative_predictive_value_dev;
alfa_msg::msg::MetricMessage f1_dev;
alfa_msg::msg::MetricMessage total_points_mean;
alfa_msg::msg::MetricMessage points_removed_mean;
alfa_msg::msg::MetricMessage TP_mean;
alfa_msg::msg::MetricMessage FP_mean;
alfa_msg::msg::MetricMessage TN_mean;
alfa_msg::msg::MetricMessage FN_mean;

alfa_msg::msg::MetricMessage fr;
alfa_msg::msg::MetricMessage TP;
alfa_msg::msg::MetricMessage FP;
alfa_msg::msg::MetricMessage TN;
alfa_msg::msg::MetricMessage FN;
alfa_msg::msg::MetricMessage total_points;
alfa_msg::msg::MetricMessage points_removed;


AlfaExtensionParameter parameters [30];

boost::shared_ptr<PatchWork<PointTypePatchwork> > PatchworkGroundSeg;

vector<float> tpr_vec, tnr_vec, ppv_vec, npv_vec, f1_vec, total_points_vec, removed_vec, tp_vec, fp_vec, tn_vec, fn_vec, htime_vec, ptime_vec;
int frames;

float mean_vector ( vector<float> v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
float std_dev_vector ( vector<float> v, double vmean) { return std::sqrt(std::accumulate(v.begin(), v.end(), 0.0,[&](double acc, double x){ return acc + std::pow(x - vmean, 2); }) / v.size()); }

void callback_shutdown()
{
    if(frames > 0)
    {
        double _tpr_mean = mean_vector(tpr_vec);
        double _tpr_dev = std_dev_vector(tpr_vec, _tpr_mean);
        double _tnr_mean = mean_vector(tnr_vec);
        double _tnr_dev = std_dev_vector(tnr_vec, _tnr_mean);
        double _ppv_mean = mean_vector(ppv_vec);
        double _ppv_dev = std_dev_vector(ppv_vec, _ppv_mean);
        double _npv_mean = mean_vector(npv_vec);
        double _npv_dev = std_dev_vector(npv_vec, _npv_mean);
        double _f1_mean = mean_vector(f1_vec);
        double _f1_dev = std_dev_vector(f1_vec, _f1_mean);

        std::cout << std::endl << "------ METRICS -------" << std::endl;
        std::cout << "Number of frames: " << frames << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "TP: " << mean_vector(tp_vec) << std::endl;
        std::cout << "FP: " << mean_vector(fp_vec) << std::endl;
        std::cout << "TN: " << mean_vector(tn_vec) << std::endl;
        std::cout << "FN: " << mean_vector(fn_vec) << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout.precision(4);
        std::cout << "True Positive Rate: " << _tpr_mean * 100 << " +/- " << _tpr_dev * 100 << " %" << std::endl;
        std::cout << "True Negative Rate: " << _tnr_mean * 100 << " +/- " << _tnr_dev * 100 << " %" << std::endl;
        std::cout << "Positive Predictive Value: " << _ppv_mean * 100 << " +/- " << _ppv_dev * 100 << " %" << std::endl;
        std::cout << "Negative Predictive Value: " << _npv_mean * 100 << " +/- " << _npv_dev * 100 << " %" << std::endl;
        std::cout << "F1-Score: " << _f1_mean * 100 << " +/- " << _f1_dev * 100 << " %" << std::endl;
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

void metrics (double _TP, double _FP, double _TN, double _FN, int tpoints, int rpoints)
{   
    double _tpr = _TP / ( _TP + _FN );
    double _tnr = _TN / ( _TN + _FP );
    double _ppv = _TP / ( _TP + _FP );
    double _npv = _TN / ( _TN + _FN );
    float _f1 = 2 * ( _ppv * _tpr ) / ( _ppv + _tpr ); 

    tpr_vec.push_back(_tpr);
    tnr_vec.push_back(_tnr);
    ppv_vec.push_back(_ppv);
    npv_vec.push_back(_npv);
    f1_vec.push_back(_f1);
    total_points_vec.push_back(static_cast<float>(tpoints)); 
    removed_vec.push_back(static_cast<float>(rpoints));
    tp_vec.push_back(_TP);
    fp_vec.push_back(_FP);
    tn_vec.push_back(_TN);
    fn_vec.push_back(_FN);

    double _tpr_mean = mean_vector(tpr_vec);
    double _tpr_dev = std_dev_vector(tpr_vec, _tpr_mean);
    double _tnr_mean = mean_vector(tnr_vec);
    double _tnr_dev = std_dev_vector(tnr_vec, _tnr_mean);
    double _ppv_mean = mean_vector(ppv_vec);
    double _ppv_dev = std_dev_vector(ppv_vec, _ppv_mean);
    double _npv_mean = mean_vector(npv_vec);
    double _npv_dev = std_dev_vector(npv_vec, _npv_mean);
    double _f1_mean = mean_vector(f1_vec);
    double _f1_dev = std_dev_vector(f1_vec, _f1_mean);

    double _tpoints_mean = mean_vector(total_points_vec);
    double _rpoints_mean = mean_vector(removed_vec);

    fr.metric = frames; 

    // Metrics of last frame 

    true_positive_rate.metric = _tpr * 100;
    true_negative_rate.metric = _tnr * 100;
    positive_predictive_value.metric = _ppv * 100;
    negative_predictive_value.metric = _npv * 100;
    f1.metric = _f1 * 100;

    TP.metric = _TP;
    FP.metric = _FP;
    TN.metric = _TN;
    FN.metric = _FN;

    TP_mean.metric = mean_vector(tp_vec);
    FP_mean.metric = mean_vector(fp_vec);
    TN_mean.metric = mean_vector(tn_vec);
    FN_mean.metric = mean_vector(fn_vec);

    total_points.metric = tpoints;
    points_removed.metric = rpoints;

    // Metrics mean and std dev of all frames

    true_positive_rate_mean.metric = _tpr_mean * 100;
    true_negative_rate_mean.metric = _tnr_mean * 100;
    positive_predictive_value_mean.metric = _ppv_mean * 100;
    negative_predictive_value_mean.metric = _npv_mean * 100;
    f1_mean.metric = _f1_mean * 100;
    total_points_mean.metric = _tpoints_mean * 100;
    points_removed_mean.metric = _rpoints_mean * 100;
    
    true_positive_rate_dev.metric = _tpr_dev * 100;
    true_negative_rate_dev.metric = _tnr_dev * 100;
    positive_predictive_value_dev.metric = _ppv_dev * 100;
    negative_predictive_value_dev.metric = _npv_dev * 100;
    f1_dev.metric = _f1_dev * 100;
}

void pc2rgb (AlfaNode * node, pcl::PointCloud<PointTypePatchwork> pc_ground, pcl::PointCloud<PointTypePatchwork> pc_non_ground)
{   
    std::uint8_t r, g, b;
    if(node->get_extension_parameter("show_ground_rgb") != 0)
    {
        r = 255;
        g = 0;
        b = 0;
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
    }
    
    if(node->get_extension_parameter("show_non_ground_rgb") != 0)
    {
        r = 0;
        b = 0;
        g = 255;
        uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
        for (const auto& point : pc_non_ground.points) // PUBLISH
        {
            pcl::PointXYZRGB p;

            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            p.rgb = *reinterpret_cast<float*>(&rgb);

            node->push_point_output_cloud(p);
        }
    }
}

void update_var (AlfaNode * node, boost::shared_ptr<PatchWork<PointTypePatchwork> > PatchworkGroundSeg)
{   
    size_t num_parameters = sizeof(parameters) / sizeof(parameters[0]);

    for ( int i = 0; i < num_parameters; i++)
    {
        if ( parameters[i].parameter_name != "" )
        {
            PatchworkGroundSeg->update_parameters(parameters[i].parameter_name,
            node->get_extension_parameter(parameters[i].parameter_name));
        }
    }
}

void handler (AlfaNode * node)
{
    pcl::PointCloud<PointTypePatchwork>::Ptr input_cloud;
    input_cloud.reset(new pcl::PointCloud<PointTypePatchwork>);
    pcl::fromROSMsg(*(node->ros_pointcloud),*input_cloud);

    frames++;
    std::cout << "\r\e[K" << "Point cloud [" << frames << "] received!" << std::flush;

    pcl::PointCloud<PointTypePatchwork> pc_ground;
    pcl::PointCloud<PointTypePatchwork> pc_non_ground;
    double time_taken;
    double TP, FP, TN, FN;

    update_var(node, PatchworkGroundSeg);
    PatchworkGroundSeg->initialize();
    PatchworkGroundSeg->estimate_ground(*input_cloud, pc_ground, pc_non_ground, time_taken);
    
    PatchworkGroundSeg->metrics_wo_vegetation(*input_cloud, pc_ground, pc_non_ground, TP, FP, TN, FN);
    metrics(TP, FP, TN, FN, input_cloud->size(), pc_ground.size());

    pc2rgb(node, pc_ground, pc_non_ground);
}

void post_processing (AlfaNode * node)
{
    node->publish_output_cloud();

    alfa_msg::msg::AlfaMetrics output_metrics;

    output_metrics.metrics.push_back(node->get_handler_time());
    output_metrics.metrics.push_back(node->get_full_processing_time());

    htime_vec.push_back(node->get_handler_time().metric);
    ptime_vec.push_back(node->get_full_processing_time().metric);

    output_metrics.metrics.push_back(fr);
    output_metrics.metrics.push_back(total_points);
    output_metrics.metrics.push_back(points_removed);
    output_metrics.metrics.push_back(true_positive_rate);
    output_metrics.metrics.push_back(true_negative_rate);
    output_metrics.metrics.push_back(positive_predictive_value);
    output_metrics.metrics.push_back(negative_predictive_value);
    output_metrics.metrics.push_back(f1);

    output_metrics.metrics.push_back(TP);
    output_metrics.metrics.push_back(FP);
    output_metrics.metrics.push_back(TN);
    output_metrics.metrics.push_back(FN);   

    output_metrics.metrics.push_back(total_points_mean);
    output_metrics.metrics.push_back(points_removed_mean);
    output_metrics.metrics.push_back(TP_mean);
    output_metrics.metrics.push_back(FP_mean);
    output_metrics.metrics.push_back(TN_mean);
    output_metrics.metrics.push_back(FN_mean);   
    output_metrics.metrics.push_back(true_positive_rate_mean);
    output_metrics.metrics.push_back(true_positive_rate_dev);
    output_metrics.metrics.push_back(true_negative_rate_mean);
    output_metrics.metrics.push_back(true_negative_rate_dev);
    output_metrics.metrics.push_back(positive_predictive_value_mean);    
    output_metrics.metrics.push_back(positive_predictive_value_dev);
    output_metrics.metrics.push_back(negative_predictive_value_mean);
    output_metrics.metrics.push_back(negative_predictive_value_dev);
    output_metrics.metrics.push_back(f1_mean);
    output_metrics.metrics.push_back(f1_dev);

    node->publish_metrics(output_metrics);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    frames = 0;

    //////////////////////////////////////////////////////

    parameters[0].parameter_name = "verbose_";
    parameters[0].parameter_value = 0.0;

    parameters[1].parameter_name = "ATAT_ON_";
    parameters[1].parameter_value = 1.0;

    parameters[2].parameter_name = "max_r_for_ATAT_";
    parameters[2].parameter_value = 5.0;

    parameters[3].parameter_name = "num_sectors_for_ATAT_";
    parameters[3].parameter_value = 20.0;

    parameters[4].parameter_name = "num_iter_";
    parameters[4].parameter_value = 3.0;

    parameters[5].parameter_name = "num_lpr_";
    parameters[5].parameter_value = 20.0;

    parameters[6].parameter_name = "num_min_pts_";
    parameters[6].parameter_value = 10.0;

    parameters[7].parameter_name = "num_zones_";
    parameters[7].parameter_value = 4.0;

    parameters[8].parameter_name = "num_rings_of_interest_";
    parameters[8].parameter_value = 4.0;

    parameters[9].parameter_name = "sensor_height_";
    parameters[9].parameter_value = 1.723;

    parameters[10].parameter_name = "th_seeds_";
    parameters[10].parameter_value = 0.5;   //0.3

    parameters[11].parameter_name = "th_dist_";
    parameters[11].parameter_value = 0.125;

    parameters[12].parameter_name = "noise_bound_";
    parameters[12].parameter_value = 0.2;

    parameters[13].parameter_name = "num_rings_";
    parameters[13].parameter_value = 30.0;

    parameters[14].parameter_name = "max_range_";
    parameters[14].parameter_value = 80.0;

    parameters[15].parameter_name = "min_range_";
    parameters[15].parameter_value = 2.7;

    parameters[16].parameter_name = "uprightness_thr_";
    parameters[16].parameter_value = 0.707;

    parameters[17].parameter_name = "adaptive_seed_selection_margin_";
    parameters[17].parameter_value = -1.1; //-1.2

    parameters[18].parameter_name = "min_range_z2_";
    parameters[18].parameter_value = 12.3625;

    parameters[19].parameter_name = "min_range_z3_";
    parameters[19].parameter_value = 22.025;

    parameters[20].parameter_name = "min_range_z4_";
    parameters[20].parameter_value = 41.35;

    parameters[21].parameter_name = "num_sectors_";
    parameters[21].parameter_value = 108.0;

    parameters[22].parameter_name = "using_global_thr_";
    parameters[22].parameter_value = 0.0;

    parameters[23].parameter_name = "global_elevation_thr_";
    parameters[23].parameter_value = -0.5;

    parameters[25].parameter_name = "show_ground_rgb";
    parameters[25].parameter_value = 1.0;

    parameters[26].parameter_name = "show_non_ground_rgb";
    parameters[26].parameter_value = 1.0;

    //////////////////////////////////////////////////////

    fr.units = "frame(s)";
    fr.metric_name = "Number of frames";
    fr.metric = 0;

    TP.units = "points";
    TP.metric_name = "TP";
    TP.metric = 0;

    TP_mean.units = "points";
    TP_mean.metric_name = "TP Mean";
    TP_mean.metric = 0;

    FP.units = "points";
    FP.metric_name = "FP";
    FP.metric = 0;

    FP_mean.units = "points";
    FP_mean.metric_name = "FP Mean";
    FP_mean.metric = 0;

    TN.units = "points";
    TN.metric_name = "TN";
    TN.metric = 0;

    TN_mean.units = "points";
    TN_mean.metric_name = "TN Mean";
    TN_mean.metric = 0;

    FN.units = "points";
    FN.metric_name = "FN";
    FN.metric = 0;

    FN_mean.units = "points";
    FN_mean.metric_name = "FN Mean";
    FN_mean.metric = 0;

    //////////////////////////////////////////////////////

    total_points.units = "points";
    total_points.metric_name = "Points in cloud";
    total_points.metric = 0;

    total_points_mean.units = "points";
    total_points_mean.metric_name = "Points in cloud Mean";
    total_points_mean.metric = 0;

    //////////////////////////////////////////////////////

    points_removed.units = "points";
    points_removed.metric_name = "Points removed from cloud";
    points_removed.metric = 0;

    points_removed_mean.units = "points";
    points_removed_mean.metric_name = "Points removed from cloud Mean";
    points_removed_mean.metric = 0;

    //////////////////////////////////////////////////////

    true_positive_rate.units = "%";
    true_positive_rate.metric_name = "TPR : True Positive Rate // Sensitivity // Recall";
    true_positive_rate.metric = 0;

    true_positive_rate_mean.units = "%";
    true_positive_rate_mean.metric_name = "TPR Mean";
    true_positive_rate_mean.metric = 0;

    true_positive_rate_dev.units = "%";
    true_positive_rate_dev.metric_name = "TPR Std Dev";
    true_positive_rate_dev.metric = 0;

    //////////////////////////////////////////////////////

    true_negative_rate.units = "%";
    true_negative_rate.metric_name = "TNR : True Negative Rate // Specificity";
    true_negative_rate.metric = 0;

    true_negative_rate_mean.units = "%";
    true_negative_rate_mean.metric_name = "TNR Mean";
    true_negative_rate_mean.metric = 0;

    true_negative_rate_dev.units = "%";
    true_negative_rate_dev.metric_name = "TNR Std Dev";
    true_negative_rate_dev.metric = 0;

    //////////////////////////////////////////////////////

    positive_predictive_value.units = "%";
    positive_predictive_value.metric_name = "PPV : Positive Predictive Value // Precision";
    positive_predictive_value.metric = 0;

    positive_predictive_value_mean.units = "%";
    positive_predictive_value_mean.metric_name = "PPV Mean";
    positive_predictive_value_mean.metric = 0;

    positive_predictive_value_dev.units = "%";
    positive_predictive_value_dev.metric_name = "PPV Std Dev";
    positive_predictive_value_dev.metric = 0;

    //////////////////////////////////////////////////////

    negative_predictive_value.units = "%";
    negative_predictive_value.metric_name = "NPV : Negative Predictive Value";
    negative_predictive_value.metric = 0;

    negative_predictive_value_mean.units = "%";
    negative_predictive_value_mean.metric_name = "NPV Mean";
    negative_predictive_value_mean.metric = 0;

    negative_predictive_value_dev.units = "%";
    negative_predictive_value_dev.metric_name = "NPV Std Dev";
    negative_predictive_value_dev.metric = 0;

    //////////////////////////////////////////////////////

    f1.units = "%";
    f1.metric_name = "F1-Score";
    f1.metric = 0;

    f1_mean.units = "%";
    f1_mean.metric_name = "F1-Score Mean";
    f1_mean.metric = 0;

    f1_dev.units = "%";
    f1_dev.metric_name = "F1-Score Std Dev";
    f1_dev.metric = 0;

    //////////////////////////////////////////////////////

    PatchworkGroundSeg.reset(new PatchWork<PointXYZILID>());

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
    rclcpp::spin(std::make_shared<AlfaNode>("/velodyne_points","patchwork", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/cloud_ply","patchworkpp", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/cloud_pcd","patchworkpp", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/kitti/point_cloud","patchworkpp", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    //rclcpp::spin(std::make_shared<AlfaNode>("/kitti/velo/pointcloud","patchworkpp", parameters, 0, AlfaHardwareSupport{false, false}, 1, 1, &handler, &post_processing));
    rclcpp::shutdown();
    return 0;
}
