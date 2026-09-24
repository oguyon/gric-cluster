# Demo of GRIC in Julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GRIC

println("GRIC Julia Demo (libgric v$(GRIC.version()))")

clusterer = Clusterer(4; rlim=1.0)
println("Initialized Clusterer with ndim=4, rlim=1.0")

# Feed frame 0 at origin
c0 = feed!(clusterer, [0.0, 0.0, 0.0, 0.0])
println("Frame 0 assigned to cluster: ", c0)
@assert c0 == 0

# Feed frame 1 close to origin
c1 = feed!(clusterer, [0.1, 0.1, 0.0, 0.0])
println("Frame 1 assigned to cluster: ", c1)
@assert c1 == 0

# Feed frame 2 far from origin
c2 = feed!(clusterer, [5.0, 0.0, 0.0, 0.0])
println("Frame 2 assigned to cluster: ", c2)
@assert c2 == 1

println("Discovered clusters: ", num_clusters(clusterer))
anc = anchors(clusterer)
println("Cluster Centroids:")
display(anc)
println()
