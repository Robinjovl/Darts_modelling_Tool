#ifndef E0E335E5_CB89_46FA_B098_6A9185E06492
#define E0E335E5_CB89_46FA_B098_6A9185E06492

#include <array>
#include <vector>
#include <pybind11/pybind11.h>
#include "interpolator_base.hpp"
#include "mech/matrix.h"

/**
 * @brief  Interpolator base for static/adaptive piecewise linear interpolator.
 * 
 * @tparam index_t index type used for supporting point indexing
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, int N_DIMS, int N_OPS>
class linear_cpu_interpolator_base : public interpolator_base
{
public:
    /**
     * @brief Construct an interpolator with specified parametrization space
     * 
     * @param[in] supporting_point_evaluator      Object used to compute operators values at supporting points
     * @param[in] axes_points                     Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                        Minimum value for each axis
     * @param[in] axes_max                        Maximum for each axis
     * @param[in] _use_barycentric_interpolation  Flag to turn on barycentric interpolation on Delaunay triangulation
     */
    linear_cpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                 const std::vector<int> &axes_points,
                                 const std::vector<double> &axes_min,
                                 const std::vector<double> &axes_max,
                                 bool _use_barycentric_interpolation);
    /**
     * @brief Get the number of dimensions in interpolation space
     *
     */
    int get_n_dims() const { return N_DIMS; };

    /**
     * @brief Get the number of operators to be interpolated
     *
     */
    int get_n_ops() const { return N_OPS; };
    /**
     * @brief Compute interpolation for all operators at the given point
     *
     * @param[in]   point   Coordinates in parametrization space
     * @param[out]  values  Interpolated values
     * @return 0 if interpolation is successful
     */
    int interpolate(const std::vector<value_t> &point, std::vector<value_t> &values) override;

    /**
     * @brief Compute interpolation and its gradient for all operators at every specified point
     *
     * @param[in]   points        Array of coordinates in parametrization space
     * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
    int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                     std::vector<double> &interp_values, std::vector<double> &derivatives) override;

    /**
     * @brief Structure that stores pre-processed information 
     *        required to perform linear interpolation on Delaunay triangulated of hypercube.
     */
    struct Delaunay
    {
      Delaunay() {};
      /// inverses of transformation matrices to barycentric coordinates
      std::vector<linalg::Matrix<double>> barycentric_matrices;
      /// Delaunay triangulation structure from scipy.spatial
      pybind11::object tri;
    };

    bool use_barycentric_interpolation; ///< flag that enables barycentric interpolation on Delaunay simplices
protected:
    std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> standard_simplex; ///< a standard simplex
    std::array<index_t, N_DIMS> axes_mult;                            ///< multiplication factor used for transferring supporting point to point index
    Delaunay tri_info;                                                ///< contains Delaunay triangulation and barycentric transformations

    int transform_last_axis; ///< apply transformation z'=1-z for the last axis

    /**
     * @brief Given the coordinate of a point, the function computes the hypercube where the point is located and its scaled
     * coordinate inside the hypercube.
     *
     * @param[in] points The array of coordinates of points
     * @param[out] hypercube The lower-left vertex of the hypercube
     * @param[out] scaled_point The scaled coordinate of the given point inside the hypercube
     * @param[in] point_index Index of the point in the std::vector points
     *      The argument point_index is used only when std::vector points consists of multiple points.
     */
    void find_hypercube(const std::vector<double> &points, std::array<index_t, N_DIMS> &hypercube,
                        std::array<double, N_DIMS> &scaled_point, const int point_index = 0);
    /**
     * @brief Compute which simplex the given point is located in using standard triangulation
     *
     * @param[in] hypercube The lower-left vertex of the hypercube
     * @param[in] scaled_point The scaled coordinate of the given point inside the hypercube
     * @param[out] tri_order The order of the scaled coordinate which is used for
     *      1. finding simplex for standard triangulation
     *      2. computing weights of the barycentric interpolation
     * @param[out] simplex An array of vertices which forms simplex in N_DIMS-dimensional space
     */
    void find_simplex(const std::array<index_t, N_DIMS> &hypercube, const std::array<double, N_DIMS> &scaled_point,
                      std::array<int, N_DIMS> &tri_order, std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> &simplex);
    /**
     * @brief Get values of operators at the given supporting point
     * Implementation depends on underlying storage. If static storage is used, the function simply reads
     * operator values of the given supporting point from the the storage.
     * If adaptive storage is used, the function checks whether the values were computed before,
     *      if yes, the value is directly returned;
     *      if not, the function computes the values, stores them and then returns.
     *
     * @param[in] vertex The indexes of coordinates the given supporting point along axes
     * @param[out] values The operator values at the given point
     */
    virtual void get_supporting_point(const std::array<index_t, N_DIMS> &vertex, std::array<double, N_OPS> &values) = 0;
    /**
     * @brief Given a supporting point, compute its index.
     *
     * This function is used as a hash for std::array<int, N_DIMS>.
     *
     * @param[in] vertex The indexes of coordinates the given supporting point along axes
     * @return The index of point among all supporting point
     */
    index_t get_index_from_vertex(const std::array<index_t, N_DIMS> &vertex);
    /**
     * @brief Transfer a vertex to its coordinates.
     *
     * @param[in] vertex The indexes of the given supporting point along axes
     * @param[out] point The coordinates of the supporting point
     */
    void get_point_from_vertex(const std::array<index_t, N_DIMS> &vertex, std::vector<double> &point);
    /**
     * @brief Loads Delaunay triangulation and data for associated barycentric interpolation.
     *
     * @param[in] filename The name of binary datafile
     */
    void load_delaunay_triangulation(const std::string filename);
    /**
     * @brief Calculate Delaunay triangulation and associated barycentric transformations.
     */
    void find_delaunay_and_barycentric();
};

#include "linear_cpu_interpolator_base.tpp"
#endif /* E0E335E5_CB89_46FA_B098_6A9185E06492 */
