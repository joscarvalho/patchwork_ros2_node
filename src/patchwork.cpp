#include "rclcpp/rclcpp.hpp"
#include "alfa_node/alfa_node.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>

#include "patchwork.hpp"

#include <iostream>
#include <fstream>

#define NODE_NAME "ext_patchwork"
#define DEFAULT_TOPIC "/velodyne_points"

#define NODE_ID 0
#define POINTCLOUD_ID 0

using PointType = AlfaPoint;
using namespace std;

boost::shared_ptr<PatchWork<PointType>> PatchworkGroundSeg;
AlfaExtensionParameter parameters[30];
int frames = 0;

void publish_cloud(AlfaNode *node, pcl::PointCloud<PointType> pc_ground, pcl::PointCloud<PointType> pc_non_ground)
{
    if (node->get_extension_parameter("show_ground_rgb") != 0)
    {
        for (const auto &point : pc_ground.points) // PUBLISH
        {
            node->push_point_output_pointcloud(point);
        }
    }

    if (node->get_extension_parameter("show_non_ground_rgb") != 0)
    {
        for (const auto &point : pc_non_ground.points) // PUBLISH
        {
            node->push_point_output_pointcloud(point);
        }
    }
}

void update_var (AlfaNode * node, boost::shared_ptr<PatchWork<PointType> > PatchworkGroundSeg)
{   
    int num_parameters = sizeof(parameters) / sizeof(parameters[0]);

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
    pcl::PointCloud<PointType> pc_ground;
    pcl::PointCloud<PointType> pc_non_ground;
    double time_taken_ATAT;
    double time_taken_ERROR;
    double time_taken_CZM;
    double time_taken_SORT;
    double time_taken_RGPF;
    double time_taken_GLE;
    static double store_atat;
    
    frames++;

    update_var(node, PatchworkGroundSeg);
    PatchworkGroundSeg->initialize();
    PatchworkGroundSeg->estimate_ground(*node->get_input_pointcloud(), pc_ground, pc_non_ground, 
                                        time_taken_ATAT, time_taken_ERROR, time_taken_CZM,
                                        time_taken_SORT,time_taken_RGPF,time_taken_GLE);

    if(frames == 1)
        store_atat = time_taken_ATAT * 1000;
	
    publish_cloud(node, pc_ground, pc_non_ground);
}

void post_processing (AlfaNode * node)
{
    node->publish_pointcloud();
}

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    // Get subscriber topic from command line arguments
    std::string subscriber_topic = DEFAULT_TOPIC;
    if (argc > 1) {
        subscriber_topic = argv[1];
    }

    #ifdef EXT_HARDWARE
        AlfaHardwareSupport hardware_support{false, true};
    #else
        AlfaHardwareSupport hardware_support{false, false};
    #endif
	
    AlfaConfiguration conf;

    conf.subscriber_topic = subscriber_topic;
    conf.node_name = NODE_NAME;
    conf.pointcloud_id = POINTCLOUD_ID;
    conf.extension_id = NODE_ID;
    conf.hardware_support = hardware_support;
    conf.latency = 0;
    conf.number_of_debug_points = 1;
    conf.metrics_publishing_type = ALL_METRICS;
    conf.custom_field_conversion_type = CUSTOM_FIELD_LABEL;

    PatchworkGroundSeg.reset(new PatchWork<PointType>());

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

    // Create an instance of AlfaNode and spin it
    rclcpp::spin(std::make_shared<AlfaNode>(conf, 
                                    parameters,
                                    &handler, 
                                    &post_processing));

    rclcpp::shutdown();
    return 0;
}
