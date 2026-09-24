/**
 * @file test_libgric_cpp.cpp
 * @brief Unit test verifying C++ RAII wrapper functionality for libgric.
 */

#include "gric/gric.hpp"
#include <iostream>
#include <cassert>
#include <vector>

int main()
{
    std::cout << "Testing libgric C++ wrapper..." << std::endl;

    // Test Config builder
    gric::Config cfg;
    cfg.with_rlim(1.0).with_max_clusters(32);

    {
        // Test RAII scope
        gric::Clusterer clusterer(4, cfg);
        assert(clusterer.ndim() == 4);
        assert(clusterer.num_clusters() == 0);

        // Feed frame 0 (std::vector)
        std::vector<double> f0 = {0.0, 0.0, 0.0, 0.0};
        int64_t c0 = clusterer.feed(f0);
        assert(c0 == 0);
        assert(clusterer.num_clusters() == 1);

        // Feed frame 1 (close to frame 0)
        std::vector<double> f1 = {0.2, 0.0, 0.0, 0.0};
        int64_t c1 = clusterer.feed(f1);
        assert(c1 == 0);
        assert(clusterer.num_clusters() == 1);

        // Feed frame 2 (far from origin)
        std::vector<double> f2 = {10.0, 0.0, 0.0, 0.0};
        int64_t c2 = clusterer.feed(f2);
        assert(c2 == 1);
        assert(clusterer.num_clusters() == 2);

        // Test batch feed
        std::vector<double> batch = {
            0.1, 0.0, 0.0, 0.0,
            10.1, 0.0, 0.0, 0.0
        };
        auto labels = clusterer.feed_batch(batch, 2);
        assert(labels.size() == 2);
        assert(labels[0] == 0);
        assert(labels[1] == 1);

        // Test invalid dimension exception
        try
        {
            std::vector<double> bad_dim = {1.0, 2.0};
            clusterer.feed(bad_dim);
            assert(false && "Should have thrown invalid_argument");
        }
        catch (const std::invalid_argument &)
        {
            // Expected
        }
    } // clusterer goes out of scope and frees its resources cleanly

    std::cout << "libgric C++ wrapper test passed successfully." << std::endl;
    return 0;
}
