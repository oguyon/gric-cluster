module GRIC

using Libdl

export Clusterer, feed!, feed_batch!, anchors, num_clusters, version

function find_libgric()
    # Search in order: environment variable, build dir, system load path
    if haskey(ENV, "LIBGRIC_PATH") && isfile(ENV["LIBGRIC_PATH"])
        return ENV["LIBGRIC_PATH"]
    end

    pkg_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
    candidates = [
        joinpath(pkg_dir, "build", "libgric.so"),
        joinpath(pkg_dir, "libgric.so"),
        "libgric.so"
    ]
    for path in candidates
        if isfile(path)
            return path
        end
    end
    return "libgric.so"
end

const LIBGRIC = Ref{String}("")

function __init__()
    LIBGRIC[] = find_libgric()
end

"""
    version() -> String

Retrieve the version string of the compiled GRIC C library.
"""
function version()::String
    ptr = ccall((:gric_version, LIBGRIC[]), Cstring, ())
    return unsafe_string(ptr)
end

"""
    Clusterer(ndim; rlim=0.5, max_clusters=256)

Instantiate a high-performance in-memory GRIC clustering session.
Memory is automatically released when the Julia object is garbage-collected.
"""
mutable struct Clusterer
    handle::Ptr{Cvoid}
    ndim::Int

    function Clusterer(ndim::Integer; rlim::Real = 0.5)
        ndim > 0 || throw(ArgumentError("ndim must be positive"))
        h = ccall((:gric_cluster_create_simple, LIBGRIC[]),
                  Ptr{Cvoid}, (Csize_t, Cdouble), Csize_t(ndim), Cdouble(rlim))
        h != C_NULL || error("Failed to allocate GRIC clustering session")

        obj = new(h, Int(ndim))
        finalizer(obj) do c
            if c.handle != C_NULL
                ccall((:gric_cluster_destroy, LIBGRIC[]), Cvoid, (Ptr{Cvoid},), c.handle)
                c.handle = C_NULL
            end
        end
        return obj
    end
end

"""
    feed!(c::Clusterer, frame::AbstractVector{<:Real}) -> Int64

Feed a single incoming coordinate frame into the clusterer.
Returns the assigned cluster index (>= 0).
"""
function feed!(c::Clusterer, frame::AbstractVector{<:Real})::Int64
    length(frame) == c.ndim || throw(DimensionMismatch("Expected vector of length $(c.ndim), got $(length(frame))"))
    c_frame = Float64.(frame)
    out_cid = Ref{Int64}(-1)
    status = ccall((:gric_cluster_feed_frame, LIBGRIC[]),
                   Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Int64}),
                   c.handle, c_frame, out_cid)
    status == 0 || error("gric_cluster_feed_frame failed with status code $status")
    return out_cid[]
end

"""
    feed_batch!(c::Clusterer, X::AbstractMatrix{<:Real}) -> Vector{Int64}

Process a batch of frames (shape: n_samples × ndim).
Returns a vector of assigned cluster indices.
"""
function feed_batch!(c::Clusterer, X::AbstractMatrix{<:Real})::Vector{Int64}
    size(X, 2) == c.ndim || throw(DimensionMismatch("Expected matrix with $(c.ndim) columns, got $(size(X, 2))"))
    num_frames = size(X, 1)

    # Convert to contiguous row-major flat buffer for C
    flat_coords = Vector{Float64}(undef, num_frames * c.ndim)
    for i in 1:num_frames
        for d in 1:c.ndim
            flat_coords[(i - 1) * c.ndim + d] = Float64(X[i, d])
        end
    end

    out_ids = Vector{Int64}(undef, num_frames)
    status = ccall((:gric_cluster_feed_batch, LIBGRIC[]),
                   Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Int64}),
                   c.handle, flat_coords, Csize_t(num_frames), out_ids)
    status == 0 || error("gric_cluster_feed_batch failed with status code $status")
    return out_ids
end

"""
    num_clusters(c::Clusterer) -> Int

Query current number of active clusters discovered.
"""
function num_clusters(c::Clusterer)::Int
    k = ccall((:gric_cluster_get_num_clusters, LIBGRIC[]), Int64, (Ptr{Cvoid},), c.handle)
    return max(0, Int(k))
end

"""
    anchors(c::Clusterer) -> Matrix{Float64}

Retrieve all discovered cluster centroids (matrix of size K × ndim).
"""
function anchors(c::Clusterer)::Matrix{Float64}
    K = num_clusters(c)
    K == 0 && return Matrix{Float64}(undef, 0, c.ndim)

    flat = Vector{Float64}(undef, K * c.ndim)
    n = ccall((:gric_cluster_get_anchors, LIBGRIC[]),
              Int64, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cint}, Csize_t),
              c.handle, flat, C_NULL, Csize_t(K))
    n > 0 || return Matrix{Float64}(undef, 0, c.ndim)

    out = Matrix{Float64}(undef, Int(n), c.ndim)
    for i in 1:Int(n)
        for d in 1:c.ndim
            out[i, d] = flat[(i - 1) * c.ndim + d]
        end
    end
    return out
end

end # module
