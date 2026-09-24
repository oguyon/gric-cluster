# Multi-Language Code Examples

This page provides side-by-side code recipes demonstrating identical tasks implemented across
**C**, **C++**, **Python**, **Rust**, and **Julia**.

---

## 1. Real-Time Streaming Ingestion

Streaming vectors frame-by-frame into a clusterer with an online radius threshold $r_{\text{lim}} = 0.5$:

=== "C"
    ```c
    #include <gric/gric.h>
    #include <stdio.h>

    int main(void)
    {
        gric_cluster_t *cl = gric_cluster_create_simple(3, 0.5);

        double frame[3] = {0.1, 0.2, 0.0};
        int64_t cluster_id = -1;

        gric_cluster_feed_frame(cl, frame, &cluster_id);
        printf("Assigned cluster: %ld\n", cluster_id);

        gric_cluster_destroy(cl);
        return 0;
    }
    ```

=== "C++"
    ```cpp
    #include <gric/gric.hpp>
    #include <iostream>
    #include <vector>

    int main()
    {
        gric::Clusterer cl(3, 0.5);

        std::vector<double> frame = {0.1, 0.2, 0.0};
        int64_t cluster_id = cl.feed(frame);

        std::cout << "Assigned cluster: " << cluster_id << std::endl;
        return 0;
    }
    ```

=== "Python"
    ```python
    import gric
    import numpy as np

    cl = gric.Clusterer(ndim=3, rlim=0.5)

    frame = np.array([0.1, 0.2, 0.0])
    cluster_id = cl.feed(frame)

    print(f"Assigned cluster: {cluster_id}")
    ```

=== "Rust"
    ```rust
    use gric::Clusterer;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut cl = Clusterer::simple(3, 0.5)?;

        let frame = [0.1, 0.2, 0.0];
        let cluster_id = cl.feed(&frame)?;

        println!("Assigned cluster: {}", cluster_id);
        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using GRIC

    cl = Clusterer(3; rlim=0.5)

    frame = [0.1, 0.2, 0.0]
    cluster_id = feed!(cl, frame)

    println("Assigned cluster: ", cluster_id)
    ```

---

## 2. Batch Ingestion (Matrices)

Clustering a matrix of vectors in memory:

=== "C"
    ```c
    #include <gric/gric.h>
    #include <stdio.h>

    int main(void)
    {
        gric_cluster_t *cl = gric_cluster_create_simple(2, 0.5);

        double batch[6] = {
            0.0, 0.0,  // Sample 0
            0.1, 0.1,  // Sample 1
            5.0, 5.0   // Sample 2
        };
        int64_t labels[3];

        gric_cluster_feed_batch(cl, batch, 3, labels);
        printf("Clusters: [%ld, %ld, %ld]\n", labels[0], labels[1], labels[2]);

        gric_cluster_destroy(cl);
        return 0;
    }
    ```

=== "C++"
    ```cpp
    #include <gric/gric.hpp>
    #include <iostream>
    #include <vector>

    int main()
    {
        gric::Clusterer cl(2, 0.5);

        std::vector<double> batch = {
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0
        };
        auto labels = cl.feed_batch(batch, 3);

        for (auto id : labels) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        return 0;
    }
    ```

=== "Python"
    ```python
    import gric
    import numpy as np

    cl = gric.Clusterer(ndim=2, rlim=0.5)

    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0]
    ])
    labels = cl.fit_predict(X)

    print("Clusters:", labels)
    ```

=== "Rust"
    ```rust
    use gric::Clusterer;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut cl = Clusterer::simple(2, 0.5)?;

        let batch = [
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0,
        ];
        let labels = cl.feed_batch(&batch, 3)?;

        println!("Clusters: {:?}", labels);
        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using GRIC

    cl = Clusterer(2; rlim=0.5)

    X = [
        0.0 0.0;
        0.1 0.1;
        5.0 5.0
    ]
    labels = feed_batch!(cl, X)

    println("Clusters: ", labels)
    ```

---

## 3. Retrieving Discovered Cluster Anchors

Accessing centroid vectors discovered by the clustering process:

=== "C"
    ```c
    int64_t K = gric_cluster_get_num_clusters(cl);
    double *anchors = (double *)malloc(K * ndim * sizeof(double));
    int *counts = (int *)malloc(K * sizeof(int));

    gric_cluster_get_anchors(cl, anchors, counts, K);
    ```

=== "C++"
    ```cpp
    size_t K = cl.num_clusters();
    std::vector<double> anchors(K * cl.ndim());
    std::vector<int> counts(K);

    gric_cluster_get_anchors(cl.c_handle(), anchors.data(), counts.data(), K);
    ```

=== "Python"
    ```python
    centroids = cl.anchors          # np.ndarray of shape (K, ndim)
    member_counts = cl.member_counts # np.ndarray of shape (K,)
    ```

=== "Rust"
    ```rust
    let centroids = cl.anchors();   // Vec<Vec<f64>>
    let num_clusters = cl.num_clusters();
    ```

=== "Julia"
    ```julia
    centroids = anchors(cl)         # Matrix{Float64} of size (K, ndim)
    K = num_clusters(cl)
    ```
