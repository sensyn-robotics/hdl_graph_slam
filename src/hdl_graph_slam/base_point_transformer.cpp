// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/base_point_transformer.hpp>
#include <cstdio>

namespace hdl_graph_slam {

BasePointTransformer::BasePointTransformer(double base_lat, double base_lon, double base_alt)
    : base_lat_(base_lat), base_lon_(base_lon), base_alt_(base_alt) {

    ctx_ = proj_context_create();

    // Define ENU coordinate system with base_point as origin
    // +proj=etmerc: Extended Transverse Mercator projection
    // +ellps=GRS80: GRS 1980 ellipsoid
    // +lon_0, +lat_0: Origin of the projection (base_point)
    // +x_0=0, +y_0=0: False easting/northing (no offset)
    // +z_0=-base_alt: Height offset to make base_point altitude = 0
    // +k_0=1: Scale factor at central meridian
    char proj_enu_crs[512];
    std::sprintf(proj_enu_crs,
        "+proj=etmerc +ellps=GRS80 +lon_0=%.10f +lat_0=%.10f "
        "+x_0=0 +y_0=0 +z_0=-%.10f +k_0=1",
        base_lon_, base_lat_, base_alt_);

    // Create coordinate transformation from WGS84 to ENU
    proj_ = proj_create_crs_to_crs(ctx_,
                                   "EPSG:4326",  // WGS84 (latitude, longitude, altitude)
                                   proj_enu_crs, // Custom ENU projection
                                   NULL);

    if (proj_ == nullptr) {
        ROS_FATAL_STREAM("Failed to create PROJ transformation for base_point origin mode. "
                        << "Check PROJ library installation.");
        exit(1);
    }

    ROS_INFO_STREAM("BasePointTransformer initialized with base_point: "
                   << "lat=" << base_lat_ << ", lon=" << base_lon_ << ", alt=" << base_alt_);
    ROS_DEBUG_STREAM("PROJ ENU definition: " << proj_enu_crs);
}

BasePointTransformer::~BasePointTransformer() {
    if (proj_) {
        proj_destroy(proj_);
    }
    if (ctx_) {
        proj_context_destroy(ctx_);
    }
}

Eigen::Vector3d BasePointTransformer::convertLLAtoENU(double lat, double lon, double alt) {
    PJ_COORD input, output;

    // Input coordinates: WGS84 (latitude, longitude, altitude, time)
    input = proj_coord(lat, lon, alt, 0.0);

    // Execute transformation: WGS84 → ENU
    output = proj_trans(proj_, PJ_FWD, input);

    // Return ENU coordinates (east, north, up)
    // Note: If transformation fails, PROJ returns HUGE_VAL
    // We don't check for errors here to match geodesy behavior
    return Eigen::Vector3d(output.enu.e, output.enu.n, output.enu.u);
}

} // namespace hdl_graph_slam
