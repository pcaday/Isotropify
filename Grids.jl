module Grids


import Base: show, print_range, ==, .+, .-, .*, ./, -, issorted, sort, sum, median, mean, var, ndims, 
             length, size, isempty, step, minimum, maximum, maxabs, start, done, next, getindex, reverse

export RegularGrid, RegularDFTGrid, RegularRFFTGrid, DFTRange
export steps, step!, step, var, vars, dftgrid, rfftgrid, maxfreq, maxfreqs, subgrid,
       fromgrid, togrid



abstract AbstractGrid{T<:Real}
abstract AbstractFourierGrid{T<:Real} <: AbstractGrid{T}

"""
A RegularGrid represents a hypercubical computational grid with
regularly-spaced gridpoints in each direction.

It may have any nonnegative number of dimensions.

An N-dimensional RegularGrid may be constructed using N ranges:

```jldoctest
julia> G = RegularGrid(0:10, -3:0.5:3)
RegularGrid{Float64}
  coordinate 1: 0.0...10.0, spacing 1.0 with 11 grid points
  coordinate 2: -3.0...3.0, spacing 0.5 with 13 grid points
```

or by specifying three vectors, containing the minimum value, maximum value,
and number of grid points (including endpoints) for each coordinate:

```jldoctest
julia> G = RegularGrid([0.; -3.], [10.; 3.], [11; 13])
RegularGrid{Float64}
  coordinate 1: 0.0...10.0, spacing 1.0 with 11 grid points
  coordinate 2: -3.0...3.0, spacing 0.5 with 13 grid points
```

If the minimum and maximum value vectors are integral types, all of the
gridpoints must fall on integer values as well, or an InexactError will
be thrown when the grid is created.
"""
immutable RegularGrid{T<:Real} <: AbstractGrid{T}
    mins::Vector{T}
    maxs::Vector{T}
    ns::Vector{Int}
    
    RegularGrid(mins,maxs,ns) = begin
        @assert length(mins) == length(maxs) == length(ns)
        @assert all(mins .< maxs)
        @assert all(ns .> 1)
        new(mins,maxs,ns)
    end
end

# Min/max/ns constructors
function RegularGrid{T<:Integer}(mins::Vector{T}, maxs::Vector{T}, ns::Vector{Int}) 
    all((maxs .- mins) .% (ns - 1) .== 0) || throw(InexactError())
    RegularGrid{T}(mins,maxs,ns)
end 
RegularGrid{T}(mins::Vector{T}, maxs::Vector{T}, ns::Vector{Int}) = RegularGrid{T}(mins,maxs,ns)
RegularGrid{T,U}(mins::Vector{T}, maxs::Vector{U}, ns::Vector{Int}) = RegularGrid(promote(mins,maxs)..., ns)

# Range constructors
function RegularGrid{T<:Real,N}(ranges::Vararg{Range{T},N})
    RegularGrid{T}(T[first(r) for r=ranges], T[last(r) for r=ranges], Int[length(r) for r=ranges])
end

RegularGrid{N}(ranges::Vararg{Range,N}) = RegularGrid(promote(ranges...)...)

# zero-D constructor
RegularGrid() = RegularGrid{Float64}(Float64[],Float64[],Float64[])

"""
A hypercubical, regularly-spaced grid of DFT frequencies.
"""
immutable RegularDFTGrid{T<:Real} <: AbstractFourierGrid{T}
    ns::Vector{Int}
    steps::Vector{T}
    
    "Construct a RegularDFTGrid from a grid of size ns with given frequency steps"
    RegularDFTGrid(ns,steps) = begin
        @assert length(ns) == length(steps)
        new(ns,steps)
    end
end

"""
A semi-hypercubical, regularly-spaced grid of frequencies for rfft.

Conceptually, this is a RegularDFTGrid, but cut in half along the
first dimension.
"""
immutable RegularRFFTGrid{T<:Real} <: AbstractFourierGrid{T}
    ns::Vector{Int}
    steps::Vector{T}
    
    "Construct a RegularRFFTGrid from a grid of size ns with given frequency steps"
    RegularRFFTGrid(ns,steps) = begin
        @assert length(ns) == length(steps)
        n1 = (ns[1] >> 1) + 1
        new([n1; ns[2:end]],steps)
    end
end


"""
A Range object representing the standard interpretation of DFT frequencies.
"""
immutable DFTRange{T<:Real} <: Range{T}
    n::Int
    step::T
    dim::Int
end

# Shared AbstractGrid methods
vars(G::AbstractGrid) = ntuple(k -> var(G,k), ndims(G))
size(G::AbstractGrid)  = (G.ns...)
size(G::AbstractGrid, k::Integer) = G.ns[k]


# RegularGrid methods

ndims(G::RegularGrid) = length(G.ns)

step{T<:Integer}(G::RegularGrid{T}, k::Integer) = (G.maxs[k] - G.mins[k]) .÷ (G.ns[k] - 1)
step(G::RegularGrid, k::Integer)                = (G.maxs[k] - G.mins[k]) ./ (G.ns[k] - 1)

steps{T<:Integer}(G::RegularGrid{T}) = (G.maxs - G.mins) .÷ (G.ns - 1)
steps(G::RegularGrid)                = (G.maxs - G.mins) ./ (G.ns - 1)

function step!(G::RegularGrid, k, step) 
    G.ns[k] = 1 + max(1, round(Int, (G.maxs[k] - G.mins[k]) / step))
    G
end

var(G::RegularGrid, k::Integer) = linspace(G.mins[k], G.maxs[k], G.ns[k])

function subgrid{T}(G::RegularGrid{T}, dims::Dims)
    for i = 1:length(dims)
        1 <= dims[i] <= ndims(G) || throw(ArgumentError("dims must be in range 1:ndims(G)"))
        for j = 1:i-1
            dims[j] == dims[i] && throw(ArgumentError("dims must be unique"))
        end
    end
    
    if length(dims) > 0
        v = collect(dims);
        RegularGrid{T}(G.mins[v], G.maxs[v], G.ns[v])
    else
        RegularGrid{T}(T[],T[],T[])
    end
end

subgrid(G::RegularGrid, dim::Integer) = subgrid(G, (dim,))


@inline function _gridcheck(G::RegularGrid, x)
    length(x) == ndims(G) || throw(ArgumentError("dimension mismatch"))
    m,M,s = G.mins, G.maxs, steps(G)

    for k = 1:length(x)
        m[k]-0.5s[k] < x[k] < M[k]+0.5s[k] || throw(ArgumentError("value outside grid"))
    end
end


function togrid(G::RegularGrid, x, dim::Integer)
    1 <= dim <= ndims(G) || throw(ArgumentError("invalid dimension requested"))
    m,M,s = G.mins[dim], G.maxs[dim], step(G,dim)
    m-0.5s < x < M+0.5s || throw(ArgumentError("value outside grid"))

    return 1 + (x-m)/s
end

function togrid{T}(G::RegularGrid, x::AbstractArray{T,1})
    _gridcheck(G,x)
    return 1 .+ (x.-G.mins)./steps(G)
end

function togrid(G::RegularGrid, x::Tuple)
    _gridcheck(G,x)
    return ntuple(k -> 1 + (x[k]-G.mins[k])./step(G,k), ndims(G))
end

function fromgrid(G::RegularGrid, xg, dim::Integer)
    1 <= dim <= ndims(G) || throw(ArgumentError("invalid dimension requested"))
    return G.mins[dim] + (xg-1)*step(G,dim)
end

function fromgrid{T}(G::RegularGrid, xg::AbstractArray{T,1})
    length(xg) == ndims(G) || throw(ArgumentError("dimension mismatch"))
    return G.mins .+ (xg.-1).*steps(G)
end

function fromgrid(G::RegularGrid, xg::Tuple)
    length(xg) == ndims(G) || throw(ArgumentError("dimension mismatch"))
    return ntuple(k -> G.mins[k] + (xg[k]-1)*step(G,k), ndims(G))
end

dftgrid{T<:AbstractFloat}(G::RegularGrid{T}) = RegularDFTGrid{T}(G.ns, 2π ./ (G.maxs .- G.mins .+ steps(G)))
dftgrid(G::RegularGrid)                      = RegularDFTGrid{Float64}(G.ns, 2π ./ (G.maxs .- G.mins .+ steps(G)))

rfftgrid{T<:AbstractFloat}(G::RegularGrid{T}) = RegularRFFTGrid{T}(G.ns, 2π ./ (G.maxs .- G.mins .+ steps(G)))
rfftgrid(G::RegularGrid)                      = RegularRFFTGrid{Float64}(G.ns, 2π ./ (G.maxs .- G.mins .+ steps(G)))


# Shared Fourier grid methods
maxfreqs(G::AbstractFourierGrid) = ntuple(k -> maxabs(var(G,k)), ndims(G))

steps(G::Union{RegularDFTGrid,RegularRFFTGrid}) = G.steps
ndims(G::Union{RegularDFTGrid,RegularRFFTGrid}) = length(G.steps)

# RegularDFTGrid methods
var(G::RegularDFTGrid, k::Integer) = DFTRange(G.ns[k], G.steps[k], k)


# RegularRFFTGrid methods
var(G::RegularRFFTGrid, k::Integer) = k>1 ? DFTRange(G.ns[k], G.steps[k], k) : range(0, G.steps[1], G.ns[1])


# DFTRange methods

length(r::DFTRange)  = r.n
isempty(r::DFTRange) = r.n == 0
step(r::DFTRange)    = r.step
minimum(r::DFTRange) = -(r.n >> 1) * r.step
maximum(r::DFTRange) = (r.n - (r.n >> 1) - 1) * r.step
maxabs(r::DFTRange)  = (r.n >> 1) * r.step
maxfreq(r::DFTRange) = maxabs(r)

@inline function unsafe_getindex{T}(r::DFTRange{T}, i::Integer)
    j = i > ((r.n+1)>>1) ? i-r.n-1 : i-1
    convert(T, j*r.step)
end

@inline function getindex{T}(r::DFTRange{T}, i::Integer)
    @boundscheck checkbounds(r,i)
    unsafe_getindex(r,i)
end

function getindex{T}(r::DFTRange{T}, s::OrdinalRange)
    @boundscheck checkbounds(r,s)
    # There are a few possibilities for handling this case.
    # If s doesn't straddle the center of the DFTRange, the result
    #  could be represented as a FloatRange; otherwise it would have
    #  to be either a new type or a plain Array.
    # To avoid a union return type, always return a plain Array.
    T[unsafe_getindex(r,i) for i=s]
end

start(r::DFTRange)        = 0
done(r::DFTRange, i::Int) = length(r) <= i

next{T}(r::DFTRange{T}, i::Int) = (convert(T, unsafe_getindex(r,i+1)), i+1)


==(r::DFTRange, s::DFTRange) = r.n == s.n && r.step == s.step

.+(x::Real, r::DFTRange) = [x+y for y=r]
.-(x::Real, r::DFTRange) = [x-y for y=r]
.-(r::DFTRange, x::Real) = [y-x for y=r]

.*(x::Real, r::DFTRange) = DFTRange(r.n, r.step*x)
./(r::DFTRange, x::Real) = DFTRange(r.n, r.step/x)

-(r::DFTRange) = [-x for x=r]

issorted(r::DFTRange) = false
sort(r::DFTRange) = linspace(minimum(r), maximum(r), r.n)

sum(r::DFTRange)    = isodd(r.n) ? 0 : minimum(r)
mean(r::DFTRange)   = isodd(r.n) ? 0 : -r.step/2
median(r::DFTRange) = mean(r)

reverse(r::DFTRange) = reverse(collect(r))











function show{T}(io::IO, G::RegularGrid{T})
    ds = steps(G)
    N = ndims(G)
    println(io, "$(N)D RegularGrid{$T}")
    for i = 1:N
        println(io, "  coordinate $(i): $(G.mins[i])...$(G.maxs[i]), spacing $(ds[i]) with $(G.ns[i]) grid points")
    end
end

function show(io::IO, G::Union{RegularDFTGrid,RegularRFFTGrid})
    N = ndims(G)
    println(io, "$(N)D $(typeof(G))")
    for i = 1:N
        print(io, "  coordinate $(i): range ")
        print_range(io, var(G,i))
        println(io, " with $(G.ns[i]) grid points")
    end
end

function show{T}(io::IO, r::DFTRange{T})
    print(io, "DFTRange{$T}($(r.n),$(r.step),$(r.dim)}")
end

#function print_range(io::IO, r::DFTRange)
#    print(io, "0,$(r.step),...,$(maximum(r)),$(minimum(r)),...,$(-r.step)")
#end



end
