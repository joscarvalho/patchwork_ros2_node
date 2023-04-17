#ifndef PATCHWORK_H
#define PATCHWORK_H

#include <iostream>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <time.h>
#include <cmath>
#include "utils.hpp"

#define MARKER_Z_VALUE -2.2
#define UPRIGHT_ENOUGH 0.55 // cyan
#define FLAT_ENOUGH 0.2 // green
#define TOO_HIGH_ELEVATION 0.0 // blue
#define TOO_TILTED 1.0 // red
#define GLOBALLLY_TOO_HIGH_ELEVATION_THR 0.8

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 3000

using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

using namespace std;

/*
    @brief PathWork ROS Node.
*/
template<typename PointT>
bool point_z_cmp(PointT a, PointT b) {
    return a.z < b.z;
}

template<typename PointT>
class PatchWork {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::vector<pcl::PointCloud<PointT> > Ring;
    typedef std::vector<Ring>                     Zone;

    PatchWork() {
        // Init ROS related
        //ROS_INFO("Inititalizing PatchWork...");

        //sensor_height_ = 1.723;
        sensor_height_ = 0;
        verbose_ = false;

        ATAT_ON_ = false;
        max_r_for_ATAT_ = 5.0;
        num_sectors_for_ATAT_ = 20;
        noise_bound_ = 0.2;

        num_iter_ = 3;
        num_lpr_ = 20;
        num_min_pts_ = 10;
        th_seeds_ = 0.4;
        th_dist_ = 0.3;
        max_range_ = 80.0;
        min_range_ = 2.7; // It indicates bodysize of the car.
        num_rings_ = 30;
        num_sectors_ = 108;
        uprightness_thr_ = 0.5; // The larger, the more strict
        adaptive_seed_selection_margin_ = -1.1; // The more larger, the more soft

        // It is not in the paper
        // It is also not matched our philosophy, but it is employed to reject some FPs easily & intuitively.
        // For patchwork, it is only applied on Z3 and Z4
        using_global_thr_ = false;
        global_elevation_thr_ = 0.0;

        // CZM denotes 'Concentric Zone Model'. Please refer to our paper
        num_zones_ = 4;
        num_sectors_each_zone_ = {16, 32, 54, 32};
        num_rings_each_zone_ = {2, 4, 4, 4};
        min_ranges_ = {2.7, 12.3625, 22.025, 41.35};
        elevation_thr_ = {0.523, 0.746, 0.879, 1.125};
        flatness_thr_ = {0.0005, 0.000725, 0.001, 0.001};

        check_input_parameters_are_correct();

        // It equals to elevation_thr_.size()/flatness_thr_.size();
        num_rings_of_interest_ = elevation_thr_.size();

        revert_pc.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        ground_pc_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        non_ground_pc_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        regionwise_ground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
        regionwise_nonground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

        min_range_z2_ = min_ranges_[1];
        min_range_z3_ = min_ranges_[2];
        min_range_z4_ = min_ranges_[3];

        min_ranges_   = {min_range_, min_range_z2_, min_range_z3_, min_range_z4_};
        ring_sizes_   = {(min_range_z2_ - min_range_) / num_rings_each_zone_.at(0),
                         (min_range_z3_ - min_range_z2_) / num_rings_each_zone_.at(1),
                         (min_range_z4_ - min_range_z3_) / num_rings_each_zone_.at(2),
                         (max_range_ - min_range_z4_) / num_rings_each_zone_.at(3)};
        sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0), 2 * M_PI / num_sectors_each_zone_.at(1),
                         2 * M_PI / num_sectors_each_zone_.at(2),
                         2 * M_PI / num_sectors_each_zone_.at(3)};

        for (int iter = 0; iter < num_zones_; ++iter) {
            Zone z;
            initialize_zone(z, num_sectors_each_zone_.at(iter), num_rings_each_zone_.at(iter));
            ConcentricZoneModel_.push_back(z);
        }
    }

    void estimate_ground(
            const pcl::PointCloud<PointT> &cloudIn,
            pcl::PointCloud<PointT> &cloudOut,
            pcl::PointCloud<PointT> &cloudNonground,
            double &time_taken);

    //geometry_msgs::PolygonStamped set_plane_polygon(const MatrixXf &normal_v, const float &d);

private:
    // For ATAT (All-Terrain Automatic heighT estimator)
    bool ATAT_ON_;
    double noise_bound_;
    double max_r_for_ATAT_;
    int num_sectors_for_ATAT_;

    int num_iter_;
    int num_lpr_;
    int num_min_pts_;
    int num_rings_;
    int num_sectors_;
    int num_zones_;
    int num_rings_of_interest_;

    double sensor_height_;
    double th_seeds_;
    double th_dist_;
    double max_range_;
    double min_range_;
    double uprightness_thr_;
    double adaptive_seed_selection_margin_;
    double min_range_z2_; // 12.3625
    double min_range_z3_; // 22.025
    double min_range_z4_; // 41.35

    bool verbose_;
    bool initialized_ = true;

    // For global threshold
    bool   using_global_thr_;
    double global_elevation_thr_;

    float           d_;
    MatrixXf        normal_;
    VectorXf        singular_values_;
    float           th_dist_d_;
    Eigen::Matrix3f cov_;
    Eigen::Vector4f pc_mean_;
    double          ring_size;
    double          sector_size;
    // For visualization
    bool            visualize_;

    vector<int> num_sectors_each_zone_;
    vector<int> num_rings_each_zone_;

    vector<double> sector_sizes_;
    vector<double> ring_sizes_;
    vector<double> min_ranges_;
    vector<double> elevation_thr_;
    vector<double> flatness_thr_;

    vector<Zone> ConcentricZoneModel_;

    //jsk_recognition_msgs::PolygonArray poly_list_;

    pcl::PointCloud<PointT> revert_pc, reject_pc;
    pcl::PointCloud<PointT> ground_pc_;
    pcl::PointCloud<PointT> non_ground_pc_;

    pcl::PointCloud<PointT> regionwise_ground_;
    pcl::PointCloud<PointT> regionwise_nonground_;

    void check_input_parameters_are_correct();

    void cout_params();

    void initialize_zone(Zone &z, int num_sectors, int num_rings);

    void flush_patches_in_zone(Zone &patches, int num_sectors, int num_rings);

    double calc_principal_variance(const Eigen::Matrix3f &cov, const Eigen::Vector4f &centroid);

    double xy2theta(const double &x, const double &y);

    double xy2radius(const double &x, const double &y);

    void pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm);

    void estimate_plane_(const pcl::PointCloud<PointT> &ground);

    void extract_piecewiseground(
            const int zone_idx, const pcl::PointCloud<PointT> &src,
            pcl::PointCloud<PointT> &dst,
            pcl::PointCloud<PointT> &non_ground_dst,
            bool is_h_available=true);

    void estimate_plane_(const int zone_idx, const pcl::PointCloud<PointT> &ground);

    double consensus_set_based_height_estimation(const Eigen::RowVectorXd& X,
                                                 const Eigen::RowVectorXd& ranges,
                                                 const Eigen::RowVectorXd& weights);

    void estimate_sensor_height(pcl::PointCloud<PointT> cloud_in);

    void extract_initial_seeds_(
            const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
            pcl::PointCloud<PointT> &init_seeds, bool is_h_available=true);
};


template<typename PointT>
inline
void PatchWork<PointT>::initialize_zone(Zone &z, int num_sectors, int num_rings) {
    z.clear();
    pcl::PointCloud<PointT> cloud;
    cloud.reserve(1000);
    Ring     ring;
    for (int i = 0; i < num_sectors; i++) {
        ring.emplace_back(cloud);
    }
    for (int j = 0; j < num_rings; j++) {
        z.emplace_back(ring);
    }
}

template<typename PointT>
inline
void PatchWork<PointT>::flush_patches_in_zone(Zone &patches, int num_sectors, int num_rings) {
    for (int i = 0; i < num_sectors; i++) {
        for (int j = 0; j < num_rings; j++) {
            if (!patches[j][i].points.empty()) patches[j][i].points.clear();
        }
    }
}

template<typename PointT>
inline
void PatchWork<PointT>::estimate_plane_(const pcl::PointCloud<PointT> &ground) {
    pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);
    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();

    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean_.head<3>();

    // according to normal.T*[x,y,z] = -d
    d_         = -(normal_.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
}

template<typename PointT>
inline
void PatchWork<PointT>::extract_initial_seeds_(
        const int zone_idx, const pcl::PointCloud<PointT> &p_sorted,
        pcl::PointCloud<PointT> &init_seeds, bool is_h_available) {
    init_seeds.points.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int    cnt = 0;

    int init_idx = 0;
    // Empirically, adaptive seed selection applying to Z1 is fine
    if (is_h_available) {
        static double lowest_h_margin_in_close_zone = (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ *
                                                                                       sensor_height_;
        if (zone_idx == 0) {
            for (int i = 0; i < p_sorted.points.size(); i++) {
                if (p_sorted.points[i].z < lowest_h_margin_in_close_zone) {
                    ++init_idx;
                } else {
                    break;
                }
            }
        }
    }

    // Calculate the mean height value.
    for (int i          = init_idx; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double   lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seeds_) {
            init_seeds.points.push_back(p_sorted.points[i]);
        }
    }
}
template<typename PointT>
inline
double PatchWork<PointT>::consensus_set_based_height_estimation(const Eigen::RowVectorXd& X,
                                                              const Eigen::RowVectorXd& ranges,
                                                              const Eigen::RowVectorXd& weights) {
    // check input parameters
    bool dimension_inconsistent = (X.rows() != ranges.rows()) || (X.cols() != ranges.cols());

    bool only_one_element = (X.rows() == 1) && (X.cols() == 1);
    assert(!dimension_inconsistent);
    assert(!only_one_element); // TODO: admit a trivial solution

    int N = X.cols();
    std::vector<std::pair<double, int>> h;
    for (size_t i= 0 ;i < N ;++i){
        h.push_back(std::make_pair(X(i) - ranges(i), i+1));
        h.push_back(std::make_pair(X(i) + ranges(i), -i-1));
    }

    // ascending order
    std::sort(h.begin(), h.end(), [](std::pair<double, int> a, std::pair<double, int> b) { return a.first < b.first; });

    int nr_centers = 2 * N;
    Eigen::RowVectorXd x_hat = Eigen::MatrixXd::Zero(1, nr_centers);
    Eigen::RowVectorXd x_cost = Eigen::MatrixXd::Zero(1, nr_centers);

    double ranges_inverse_sum = ranges.sum();
    double dot_X_weights = 0;
    double dot_weights_consensus = 0;
    int consensus_set_cardinal = 0;
    double sum_xi = 0;
    double sum_xi_square = 0;

    for (size_t i = 0 ; i < nr_centers ; ++i){

        int idx = int(std::abs(h.at(i).second)) - 1; // Indices starting at 1
        int epsilon = (h.at(i).second > 0) ? 1 : -1;

        consensus_set_cardinal += epsilon;
        dot_weights_consensus += epsilon * weights(idx);
        dot_X_weights += epsilon * weights(idx) * X(idx);
        ranges_inverse_sum -= epsilon * ranges(idx);
        sum_xi += epsilon * X(idx);
        sum_xi_square += epsilon * X(idx) * X(idx);

        x_hat(i) = dot_X_weights / dot_weights_consensus;

        double residual = consensus_set_cardinal * x_hat(i) * x_hat(i) + sum_xi_square  - 2 * sum_xi * x_hat(i);
        x_cost(i) = residual + ranges_inverse_sum;

    }

    size_t min_idx;
    x_cost.minCoeff(&min_idx);
    double estimate_temp = x_hat(min_idx);
    return estimate_temp;
}

template<typename PointT>
inline
void PatchWork<PointT>::estimate_sensor_height(pcl::PointCloud<PointT> cloud_in) {
    // ATAT: All-Terrain Automatic HeighT estimator
    Ring ring_for_ATAT(num_sectors_for_ATAT_);
    for (auto const &pt : cloud_in.points) {
        int    ring_idx, sector_idx;
        double r = xy2radius(pt.x, pt.y);

        float sector_size_for_ATAT = 2 * M_PI / num_sectors_for_ATAT_;

        if ((r <= max_r_for_ATAT_) && (r > min_range_)) {
            double theta = xy2theta(pt.x, pt.y);

            sector_idx = min(static_cast<int>((theta / sector_size_for_ATAT)), num_sectors_for_ATAT_);
            ring_for_ATAT[sector_idx].points.emplace_back(pt);
        }
    }

    // Assign valid measurements and corresponding linearities/planarities
    vector<double> ground_elevations_wrt_the_origin;
    vector<double> linearities;
    vector<double> planarities;
    for (int i = 0; i < num_sectors_for_ATAT_; ++i) {

        if(ring_for_ATAT[i].size() < num_min_pts_ ){ continue; }

        pcl::PointCloud<PointT> dummy_est_ground;
        pcl::PointCloud<PointT> dummy_est_non_ground;
        extract_piecewiseground(0, ring_for_ATAT[i], dummy_est_ground, dummy_est_non_ground, false);

        const double ground_z_vec       = abs(normal_(2, 0));
        const double ground_z_elevation = pc_mean_(2, 0);
        const double linearity   =
                (singular_values_(0) - singular_values_(1)) / singular_values_(0);
        const double planarity   =
                (singular_values_(1) - singular_values_(2)) / singular_values_(0);

        // Check whether the vector is sufficiently upright and flat
        if (ground_z_vec > uprightness_thr_ && linearity < 0.9) {
            ground_elevations_wrt_the_origin.push_back(ground_z_elevation);
            linearities.push_back(linearity);
            planarities.push_back(planarity);
        }
    }

    // Setting for consensus set-based height estimation
    int N = ground_elevations_wrt_the_origin.size();
    Eigen::Matrix<double, 1, Eigen::Dynamic> values = Eigen::MatrixXd::Ones(1, N);
    Eigen::Matrix<double, 1, Eigen::Dynamic> ranges = noise_bound_ * Eigen::MatrixXd::Ones(1, N);
    Eigen::Matrix<double, 1, Eigen::Dynamic> weights = 1.0 / (noise_bound_ * noise_bound_) * Eigen::MatrixXd::Ones(1, N);
    for (int i = 0; i < N; ++i) {
        values(0, i) = ground_elevations_wrt_the_origin[i];
        ranges(0, i) = ranges(0, i) * linearities[i];
        weights(0, i) = weights(0, i) * planarities[i] * planarities[i];
    }

    double estimated_h = consensus_set_based_height_estimation(values, ranges, weights);
    cout << "\033[1;33m[ATAT] The sensor height is auto-calibrated via the ground points in the vicinity of the vehicle\033[0m" << endl;
    cout << "\033[1;33m[ATAT] Elevation of the ground w.r.t. the origin is " << estimated_h << " m\033[0m" << endl;

    // Note that these are opposites
    sensor_height_ = -estimated_h;
}

template<typename PointT>
inline
void PatchWork<PointT>::estimate_ground(
        const pcl::PointCloud<PointT> &cloud_in,
        pcl::PointCloud<PointT> &cloud_out,
        pcl::PointCloud<PointT> &cloud_nonground,
        double &time_taken) {

    if (initialized_ && ATAT_ON_) {
        estimate_sensor_height(cloud_in);
        initialized_ = false;
    }

    //static double start, t0, t1, t2, end;

    double                  t_total_sort   = 0.0;
    double                  t_total_ground   = 0.0;
    double                  t_total_estimate = 0.0;
    // 1.Msg to pointcloud
    pcl::PointCloud<PointT> laserCloudIn;
    laserCloudIn = cloud_in;

    //start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    //t0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // 2.Error point removal
    // As there are some error mirror reflection under the ground,
    // Sort point according to height, here uses z-axis in default
    // -2.0 is a rough criteria
    for (int i = 0; i < laserCloudIn.points.size(); i++) {
        if (laserCloudIn.points[i].z < -sensor_height_ - 2.0) {
            std::iter_swap(laserCloudIn.points.begin() + i, laserCloudIn.points.end() - 1);
            laserCloudIn.points.pop_back();
        }
    }

    //t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // 4. pointcloud -> regionwise setting
    for (int k = 0; k < num_zones_; ++k) {
        flush_patches_in_zone(ConcentricZoneModel_[k], num_sectors_each_zone_[k], num_rings_each_zone_[k]);
    }
    pc2czm(laserCloudIn, ConcentricZoneModel_);

    //t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    cloud_out.clear();
    cloud_nonground.clear();
    revert_pc.clear();
    reject_pc.clear();

    int      concentric_idx = 0;
    for (int k = 0; k < num_zones_; ++k) {
        auto zone = ConcentricZoneModel_[k];
        for (uint16_t ring_idx = 0; ring_idx < num_rings_each_zone_[k]; ++ring_idx) {
            for (uint16_t sector_idx = 0; sector_idx < num_sectors_each_zone_[k]; ++sector_idx) {
                if (zone[ring_idx][sector_idx].points.size() > num_min_pts_) {
                    //double t_tmp0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    
                                                  /* Region-wise Ground Plane Fitting using PCA */
                    // Region-wise sorting is adopted
                    sort(zone[ring_idx][sector_idx].points.begin(), zone[ring_idx][sector_idx].end(), point_z_cmp<PointT>);
                    //double t_tmp1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    extract_piecewiseground(k, zone[ring_idx][sector_idx], regionwise_ground_, regionwise_nonground_);
                    //double t_tmp2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    //t_total_ground += t_tmp2 - t_tmp1;
                    //t_total_sort += t_tmp1 - t_tmp0;

                                                        /* Ground Likelihood Estimation */
                    // Status of each patch
                    // used in checking uprightness, elevation, and flatness, respectively
                    const double v3 = normal_(2,0);
                    const double ground_z_vec  = abs(normal_(2, 0));
                    const double uprightness, elevation, flatness;
                    uprightness = ( (v3 / abs(v3)) > cos((M_PI_2 - uprightness_thr_)) ) ? 1.0 : 0.0;

                    const double z_mean = pc_mean_(2,0);
                    elevation = 

                    const double ground_z_elevation = pc_mean_(2, 0);
                    const double surface_variable   =
                                         singular_values_.minCoeff() /
                                         (singular_values_(0) + singular_values_(1) + singular_values_(2));

                    //double t_tmp2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (ground_z_vec < uprightness_thr_) {
                        // All points are rejected
                        cloud_nonground += regionwise_ground_;
                    } else { // satisfy uprightness
                        if (concentric_idx < num_rings_of_interest_) {
                            if (ground_z_elevation > -sensor_height_ + elevation_thr_[ring_idx + 2 * k]) {
                                if (flatness_thr_[ring_idx + 2 * k] > surface_variable) {
                                    cloud_out += regionwise_ground_;
                                } else {
                                    cloud_nonground += regionwise_ground_;
                                }
                            } else {
                                cloud_out += regionwise_ground_;
                            }
                        } else {
                            if (using_global_thr_ && (ground_z_elevation > global_elevation_thr_)) {
                                cloud_nonground += regionwise_ground_;
                            } else {
                                cloud_out += regionwise_ground_;
                            }
                        }
                    }

                    // Every regionwise_nonground is considered nonground.
                    cloud_nonground += regionwise_nonground_;

                    //double t_tmp3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    //t_total_estimate += t_tmp3 - t_tmp2;
                }
            }
            ++concentric_idx;
        }
    }
    // end                     = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // time_taken              = end - start;

    // ofstream time_txt("/home/carvalho/Desktop/patchwork_time.txt", std::ios::app);
    // time_txt << "Time taken : " << time_taken << endl;
    // time_txt << "Time taken to sort: " << t0-start << endl;
    // time_txt << "Time taken to remove error points: " << t1 - t0 << endl;
    // time_txt << "Time taken to pc -> czm: " << t2-t1 << endl;
    //std::cout << "Time to sort : " << t_total_sort << endl;
    //std::cout << "Time taken to estimate : " << t_total_ground << endl;
    //std::cout << "Time taken to pca : " << t_total_ground + t_total_sort << endl;
    // time_txt << "Time taken to estimate: " << t_total_estimate << endl;

    // time_txt.close();
}

template<typename PointT>
inline
double PatchWork<PointT>::calc_principal_variance(const Eigen::Matrix3f &cov, const Eigen::Vector4f &centroid) {
    double angle       = atan2(centroid(1, 0), centroid(0, 0)); // y, x
    double c           = cos(angle);
    double s           = sin(angle);
    double var_x_prime = c * c * cov(0, 0) + s * s * cov(1, 1) + 2 * c * s * cov(0, 1);
    double var_y_prime = s * s * cov(0, 0) + c * c * cov(1, 1) - 2 * c * s * cov(0, 1);
    return max(var_x_prime, var_y_prime);
}


template<typename PointT>
inline
double PatchWork<PointT>::xy2theta(const double &x, const double &y) { // 0 ~ 2 * PI
    /*
      if (y >= 0) {
          return atan2(y, x); // 1, 2 quadrant
      } else {
          return 2 * M_PI + atan2(y, x);// 3, 4 quadrant
      }
    */
    auto atan_value = atan2(y,x); // EDITED!
    return atan_value > 0 ? atan_value : atan_value + 2*M_PI; // EDITED!
}

template<typename PointT>
inline
double PatchWork<PointT>::xy2radius(const double &x, const double &y) {
    return sqrt(pow(x, 2) + pow(y, 2));
}

template<typename PointT>
inline
void PatchWork<PointT>::pc2czm(const pcl::PointCloud<PointT> &src, std::vector<Zone> &czm) {

    for (auto const &pt : src.points) {
        int    ring_idx, sector_idx;
        double r = xy2radius(pt.x, pt.y);
        if ((r <= max_range_) && (r > min_range_)) {
            double theta = xy2theta(pt.x, pt.y);

            if (r < min_range_z2_) { // In First rings
                ring_idx   = min(static_cast<int>(((r - min_range_) / ring_sizes_[0])), num_rings_each_zone_[0] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[0])), num_sectors_each_zone_[0] - 1);
                czm[0][ring_idx][sector_idx].points.emplace_back(pt);
            } else if (r < min_range_z3_) {
                ring_idx   = min(static_cast<int>(((r - min_range_z2_) / ring_sizes_[1])), num_rings_each_zone_[1] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[1])), num_sectors_each_zone_[1] - 1);
                czm[1][ring_idx][sector_idx].points.emplace_back(pt);
            } else if (r < min_range_z4_) {
                ring_idx   = min(static_cast<int>(((r - min_range_z3_) / ring_sizes_[2])), num_rings_each_zone_[2] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[2])), num_sectors_each_zone_[2] - 1);
                czm[2][ring_idx][sector_idx].points.emplace_back(pt);
            } else { // Far!
                ring_idx   = min(static_cast<int>(((r - min_range_z4_) / ring_sizes_[3])), num_rings_each_zone_[3] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[3])), num_sectors_each_zone_[3] - 1);
                czm[3][ring_idx][sector_idx].points.emplace_back(pt);
            }
        }

    }
}

// For adaptive
template<typename PointT>
inline
void PatchWork<PointT>::extract_piecewiseground(
        const int zone_idx, const pcl::PointCloud<PointT> &src,
        pcl::PointCloud<PointT> &dst,
        pcl::PointCloud<PointT> &non_ground_dst,
        bool is_h_available) {

    // 0. Initialization
    if (!ground_pc_.empty()) ground_pc_.clear();
    if (!dst.empty()) dst.clear();
    if (!non_ground_dst.empty()) non_ground_dst.clear();

    // 1. set seeds!
    extract_initial_seeds_(zone_idx, src, ground_pc_, is_h_available);

    // 2. Extract ground
    for (int i = 0; i < num_iter_; i++) {
        estimate_plane_(ground_pc_);
        ground_pc_.clear();
        
        //pointcloud to matrix
        Eigen::MatrixXf points(src.points.size(), 3);
        int             j      = 0;
        for (auto       &p:src.points) {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int        r      = 0; r < result.rows(); r++) {
            if (i < num_iter_ - 1) {
                if (result[r] < th_dist_d_) {
                    ground_pc_.points.push_back(src[r]);
                }
            } else { // Final stage
                if (result[r] < th_dist_d_) {
                    dst.points.push_back(src[r]);
                } else {
                    if (i == num_iter_ - 1) {
                        non_ground_dst.push_back(src[r]);
                    }
                }
            }
        }
    }
}

template<typename PointT>
inline
void PatchWork<PointT>::check_input_parameters_are_correct() {
    string SET_SAME_SIZES_OF_PARAMETERS = "Some parameters are wrong! the size of parameters should be same";

    int n_z = num_zones_;
    int n_r = num_rings_each_zone_.size();
    int n_s = num_sectors_each_zone_.size();
    int n_m = min_ranges_.size();

    if ((n_z != n_r) || (n_z != n_s) || (n_z != n_m)) {
        throw invalid_argument(SET_SAME_SIZES_OF_PARAMETERS);
    }

    if ((n_r != n_s) || (n_r != n_m) || (n_s != n_m)) {
        throw invalid_argument(SET_SAME_SIZES_OF_PARAMETERS);
    }

    if (min_range_ != min_ranges_[0]) {
        throw invalid_argument("Setting min. ranges are weired! The first term should be eqaul to min_range_");
    }

    if (elevation_thr_.size() != flatness_thr_.size()) {
        throw invalid_argument("Some parameters are wrong! Check the elevation/flatness_thresholds");
    }

}


template<typename PointT>
inline
void PatchWork<PointT>::cout_params() {
    cout << (boost::format("Num. sectors: %d, %d, %d, %d") % num_sectors_each_zone_[0] % num_sectors_each_zone_[1] %
             num_sectors_each_zone_[2] %
             num_sectors_each_zone_[3]).str() << endl;
    cout << (boost::format("Num. rings: %01d, %01d, %01d, %01d") % num_rings_each_zone_[0] %
             num_rings_each_zone_[1] %
             num_rings_each_zone_[2] %
             num_rings_each_zone_[3]).str() << endl;
    cout << (boost::format("elevation_thr_: %0.4f, %0.4f, %0.4f, %0.4f ") % elevation_thr_[0] % elevation_thr_[1] %
             elevation_thr_[2] %
             elevation_thr_[3]).str() << endl;
    cout << (boost::format("flatness_thr_: %0.4f, %0.4f, %0.4f, %0.4f ") % flatness_thr_[0] % flatness_thr_[1] %
             flatness_thr_[2] %
             flatness_thr_[3]).str() << endl;


}

#endif
