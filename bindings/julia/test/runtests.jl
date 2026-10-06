using Test
using GRIC

@testset "GRIC Julia Bindings" begin
    @testset "Version" begin
        v = GRIC.version()
        @test v == "1.0.0"
    end

    @testset "Clustering Simple" begin
        c = Clusterer(4; rlim = 1.0)
        @test c.ndim == 4
        @test num_clusters(c) == 0

        # Feed frame 0 at origin
        c0 = feed!(c, [0.0, 0.0, 0.0, 0.0])
        @test c0 == 0
        @test num_clusters(c) == 1

        # Feed frame 1 close to origin (dist < 1.0)
        c1 = feed!(c, [0.1, 0.1, 0.0, 0.0])
        @test c1 == 0
        @test num_clusters(c) == 1

        # Feed frame 2 far from origin (dist = 5.0 > 1.0)
        c2 = feed!(c, [5.0, 0.0, 0.0, 0.0])
        @test c2 == 1
        @test num_clusters(c) == 2

        # Check centroids
        anc = anchors(c)
        @test size(anc) == (2, 4)
        @test anc[1, :] == [0.0, 0.0, 0.0, 0.0]
        @test anc[2, :] == [5.0, 0.0, 0.0, 0.0]
    end

    @testset "Batch Feed" begin
        c = Clusterer(2; rlim = 0.5)
        X = [0.0 0.0; 10.0 10.0]
        ids = feed_batch!(c, X)
        @test ids == [0, 1]
        @test num_clusters(c) == 2
    end
end
