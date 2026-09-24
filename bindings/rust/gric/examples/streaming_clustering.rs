use gric::{Clusterer, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Rust GRIC clustering demo (libgric v{})...", gric::version());

    let mut clusterer = Clusterer::new(4, Config::default().rlim(1.0).max_clusters(32))?;

    // Frame 0 at origin -> creates cluster 0
    let f0 = [0.0, 0.0, 0.0, 0.0];
    let c0 = clusterer.feed(&f0)?;
    println!("Frame 0 assigned to cluster: {}", c0);
    assert_eq!(c0, 0);

    // Frame 1 near origin -> matches cluster 0
    let f1 = [0.1, 0.1, 0.0, 0.0];
    let c1 = clusterer.feed(&f1)?;
    println!("Frame 1 assigned to cluster: {}", c1);
    assert_eq!(c1, 0);

    // Frame 2 far from origin -> creates cluster 1
    let f2 = [5.0, 0.0, 0.0, 0.0];
    let c2 = clusterer.feed(&f2)?;
    println!("Frame 2 assigned to cluster: {}", c2);
    assert_eq!(c2, 1);

    println!("Total clusters discovered: {}", clusterer.num_clusters());
    let anchors = clusterer.anchors();
    for (idx, a) in anchors.iter().enumerate() {
        println!("  Cluster {}: {:?}", idx, a);
    }

    Ok(())
}
