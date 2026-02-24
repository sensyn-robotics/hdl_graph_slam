// SPDX-License-Identifier: BSD-2-Clause

#ifndef BASE_POINT_TRANSFORMER_HPP
#define BASE_POINT_TRANSFORMER_HPP

#include <proj.h>
#include <Eigen/Dense>
#include <ros/ros.h>

namespace hdl_graph_slam {

/**
 * @brief Transforms GPS coordinates (LLA) to ENU coordinates with base_point as origin
 *
 * Uses PROJ library to convert WGS84 (latitude, longitude, altitude) to
 * East-North-Up (ENU) coordinates where the specified base_point is the origin (0,0,0).
 */
class BasePointTransformer {
public:
    /**
     * @brief Constructor - initializes PROJ transformation
     * @param base_lat Base point latitude (degrees)
     * @param base_lon Base point longitude (degrees)
     * @param base_alt Base point altitude (meters)
     */
    BasePointTransformer(double base_lat, double base_lon, double base_alt);

    /**
     * @brief Destructor - cleans up PROJ resources
     */
    ~BasePointTransformer();

    /**
     * @brief Convert LLA to ENU coordinates
     * @param lat Latitude (degrees)
     * @param lon Longitude (degrees)
     * @param alt Altitude (meters)
     * @return ENU coordinates (east, north, up) in meters
     */
    Eigen::Vector3d convertLLAtoENU(double lat, double lon, double alt);

private:
    PJ_CONTEXT* ctx_;
    PJ* proj_;
    double base_lat_;
    double base_lon_;
    double base_alt_;
};

} // namespace hdl_graph_slam

#endif // BASE_POINT_TRANSFORMER_HPP
