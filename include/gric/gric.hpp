/**
 * @file gric.hpp
 * @brief Modern C++ RAII wrapper for the GRIC library.
 */

#ifndef GRIC_HPP
#define GRIC_HPP

#include "gric.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gric {

/**
 * Exception class for errors returned by GRIC operations.
 */
class Error : public std::runtime_error
{
public:
    explicit Error(
        gric_status_t status,
        const std::string &msg = "GRIC operation failed")
        : std::runtime_error(msg + " (code " + std::to_string(status) + ")"),
          status_(status)
    {
    }

    gric_status_t status() const noexcept
    {
        return status_;
    }

private:
    gric_status_t status_;
};

/**
 * High-level C++ configuration helper for GRIC clustering.
 */
struct Config : public gric_cluster_config_t
{
    Config()
    {
        gric_cluster_config_default(this);
    }

    Config &with_rlim(double r)
    {
        rlim = r;
        return *this;
    }

    Config &with_max_clusters(int max_cl)
    {
        maxnbclust = max_cl;
        return *this;
    }

    Config &with_sq16(bool enabled = true)
    {
        use_sq16 = enabled ? 1 : 0;
        return *this;
    }

    Config &with_eq16(bool enabled = true)
    {
        use_eq16 = enabled ? 1 : 0;
        return *this;
    }

    Config &with_threads(int threads)
    {
        ncpu = threads;
        return *this;
    }
};

/**
 * RAII Clusterer class managing the lifecycle of a GRIC clustering session.
 */
class Clusterer
{
public:
    /**
     * Construct a Clusterer with specific dimension and configuration.
     */
    explicit Clusterer(
        size_t ndim,
        const gric_cluster_config_t &cfg = Config())
        : ndim_(ndim),
          handle_(gric_cluster_create(&cfg, ndim), &gric_cluster_destroy)
    {
        if (!handle_)
        {
            throw std::bad_alloc();
        }
    }

    /**
     * Construct a Clusterer with dimension and distance threshold.
     */
    Clusterer(
        size_t ndim,
        double rlim)
        : ndim_(ndim),
          handle_(gric_cluster_create_simple(ndim, rlim), &gric_cluster_destroy)
    {
        if (!handle_)
        {
            throw std::bad_alloc();
        }
    }

    ~Clusterer() = default;

    // Move-only semantics (prevents accidental double destruction)
    Clusterer(Clusterer &&) noexcept = default;
    Clusterer &operator=(Clusterer &&) noexcept = default;
    Clusterer(const Clusterer &) = delete;
    Clusterer &operator=(const Clusterer &) = delete;

    /**
     * Process a single incoming frame of coordinates.
     *
     * @param coords Pointer to array of coordinates of length ndim().
     * @return Assigned cluster ID.
     */
    int64_t feed(const double *coords)
    {
        int64_t cluster_id = -1;
        gric_status_t status = gric_cluster_feed_frame(handle_.get(), coords, &cluster_id);
        if (status != GRIC_SUCCESS)
        {
            throw Error(status, "feed_frame failed");
        }
        return cluster_id;
    }

    /**
     * Process a single incoming frame from a std::vector<double>.
     */
    int64_t feed(const std::vector<double> &frame)
    {
        if (frame.size() != ndim_)
        {
            throw std::invalid_argument("Vector length does not match clusterer ndim");
        }
        return feed(frame.data());
    }

    /**
     * Process a contiguous batch of coordinate frames.
     *
     * @param flat_coords Array of coordinates of length num_frames * ndim().
     * @param num_frames Number of frames in the batch.
     * @return Vector of assigned cluster IDs.
     */
    std::vector<int64_t> feed_batch(
        const double *flat_coords,
        size_t num_frames)
    {
        std::vector<int64_t> assignments(num_frames);
        gric_status_t status = gric_cluster_feed_batch(
            handle_.get(), flat_coords, num_frames, assignments.data()
        );
        if (status != GRIC_SUCCESS)
        {
            throw Error(status, "feed_batch failed");
        }
        return assignments;
    }

    /**
     * Process a batch from a flattened std::vector<double>.
     */
    std::vector<int64_t> feed_batch(
        const std::vector<double> &flat_coords,
        size_t num_frames)
    {
        if (flat_coords.size() != num_frames * ndim_)
        {
            throw std::invalid_argument("Vector length does not match num_frames * ndim");
        }
        return feed_batch(flat_coords.data(), num_frames);
    }

    /**
     * Query current number of discovered clusters.
     */
    size_t num_clusters() const noexcept
    {
        int64_t k = gric_cluster_get_num_clusters(handle_.get());
        return k >= 0 ? static_cast<size_t>(k) : 0;
    }

    /**
     * Dimensionality of feature space.
     */
    size_t ndim() const noexcept
    {
        return ndim_;
    }

    /**
     * Reset the clusterer state.
     */
    void reset()
    {
        gric_status_t status = gric_cluster_reset(handle_.get());
        if (status != GRIC_SUCCESS)
        {
            throw Error(status, "reset failed");
        }
    }

    /**
     * Access the raw underlying C handle.
     */
    gric_cluster_t *c_handle() const noexcept
    {
        return handle_.get();
    }

private:
    size_t ndim_;
    std::unique_ptr<gric_cluster_t, decltype(&gric_cluster_destroy)> handle_;
};

} // namespace gric

#endif // GRIC_HPP
