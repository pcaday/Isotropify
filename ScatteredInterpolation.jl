module ScatteredInterpolation

using VoronoiDelaunay

import GeometricalPredicates: getx, gety


export LabeledPoint2D, scatteredinterpolate, getx, gety



# An Int-labelled Point2D
immutable LabeledPoint2D <: AbstractPoint2D
    _x::Float64
    _y::Float64
    _i::Int
end
LabeledPoint2D(x::Float64,y::Float64) = LabeledPoint2D(x, y, 0)
LabeledPoint2D() = LabeledPoint2D(0., 0.)

getx(p::LabeledPoint2D) = p._x
gety(p::LabeledPoint2D) = p._y
label(p::LabeledPoint2D) = p._i


# Unsafe versions of floor and ceil (do not check for overflow)
unsafe_floor{T<:Integer}(::Type{T}, x::AbstractFloat) = unsafe_trunc(T,floor(x))
unsafe_ceil{ T<:Integer}(::Type{T}, x::AbstractFloat) = unsafe_trunc(T,ceil(x))


"""Interpolate scattered data onto a regular grid."""
function scatteredinterpolate(X::AbstractVector, Y::AbstractVector, V::AbstractVector,
                              x::Range, y::Range)
    # Validate arguments
    # Throughout, capital letters are scattered data.
    @assert length(X) == length(Y) == length(V) > 2
    nx = length(x)
    ny = length(y)
    @assert nx > 1 && ny > 1
    
    # Triangle inside test slop (in barycentric coordinates)
#    const tol = 1200eps(eltype(X))
    
    # Rescale scattered coordinates into VoronoiDelaunay's [1,2]^2 box
    const box_min = 1.25
    const box_max = 1.75
    
    @assert box_min ≥ min_coord && box_max ≤ max_coord    # Protect against API changes in VoronoiDelaunay...
    
    N = length(X)
    Xmin,Xmax = extrema(X)
    Ymin,Ymax = extrema(Y)
    Xscale = (Xmax-Xmin)/(box_max - box_min)
    Yscale = (Ymax-Ymin)/(box_max - box_min)
    invXscale = 1./max(Xscale, eps(typeof(Xscale)))
    invYscale = 1./max(Yscale, eps(typeof(Yscale)))
    
    # Create the Delaunay triangulation
    T = DelaunayTessellation2D{LabeledPoint2D}(N)::VoronoiDelaunay.DelaunayTessellation2D
    push!(T, LabeledPoint2D[LabeledPoint2D(box_min + (X[i]-Xmin)*invXscale,
                                           box_min + (Y[i]-Ymin)*invYscale, i) for i=1:N])
        
    # Rescale grid vectors into the box
    xs = box_min + (x-Xmin)*invXscale
    ys = box_min + (y-Ymin)*invYscale
    
    # Prepare to scale into grid index space
    xsstep = xs[2]-xs[1]
    ysstep = ys[2]-ys[1]
    xs_min = minimum(xs) - xsstep
    ys_min = minimum(ys) - ysstep
    xiscale = 1./xsstep
    yiscale = 1./ysstep
    
    # Initialize output with NaNs
    v = fill(convert(eltype(V),NaN), (nx,ny))
    
    # Read the tessellation.
    for t in T._trigs
        a = geta(t)
        b = getb(t)
        c = getc(t)
        ai = a._i
        bi = b._i
        ci = c._i
        ax = getx(a)
        ay = gety(a)
        bx = getx(b)
        by = gety(b)
        cx = getx(c)
        cy = gety(c)
                
        # Unlabeled points connect to the boundaries of VoronoiDelaunay's box; ignore triangles containing those.
        (ai == 0 || bi == 0 || ci == 0) && continue
        
        # Reciprocal of twice volume of the triangle, up to sign (-t._bx *t._cy + t._by*t._cx)
        w = 1 ./ t._pr2
        
        # Compute index bounds of this triangle, carefully but quickly
        xmin = min(ax,bx,cx)
        xmax = max(ax,bx,cx)
        ymin = min(ay,by,cy)
        ymax = max(ay,by,cy)

        # First, rough estimates
        ximin = unsafe_ceil( Int, (xmin - xs_min) * xiscale)
        ximax = unsafe_floor(Int, (xmax - xs_min) * xiscale)
        yimin = unsafe_ceil( Int, (ymin - ys_min) * yiscale)
        yimax = unsafe_floor(Int, (ymax - ys_min) * yiscale)
        
        # Due to accumulated roundoff error, we may be off by one. Fix if so.
        # At the same time, clamp.
        if ximin ≤ 1
            ximin = 1
        elseif xmin ≤ xs[ximin-1]
            ximin -= 1
        end
        if ximax ≥ nx
            ximax = nx
        elseif xmax ≥ xs[ximax+1]
            ximax += 1
        end
        if yimin ≤ 1
            yimin = 1
        elseif ymin ≤ ys[yimin-1]
            yimin -= 1
        end
        if yimax ≥ ny
            yimax = ny
        elseif ymax ≥ ys[yimax+1]
            yimax += 1
        end
        
        # Loop over a rectangle containing all gridpoints in the triangle
        for j = yimin:yimax
            for i = ximin:ximax
                # Barycentric coordinates for this grid point (computations in the box)
                @inbounds dx = xs[i]
                @inbounds dy = ys[j]            # let the compiler hoist this
                px = dx - ax
                py = dy - ay
                
                λb = -px*t._cy + py*t._cx
                λc = -t._bx*py + t._by*px
                
                # Need to be careful that no points get missed in the gaps. Instead of checking
                #  whether λb+λc≤1, calculate whether we are on the correct side of line BC 
                #  using only the coordinates of b and c. This way all edge tests are the same,
                #  even if we make the wrong decision for points epsilon-near the edge
                ina = (dy-cy)*(bx-cx) ≥ (dx-cx)*(by-cy)

                # Are we in the triangle?
                (ina && λb ≥ 0 && λc ≥ 0) || continue

                # Yes, do linear interpolation
                λb *= w
                λc *= w                
                @inbounds v[i,j] = V[ai] + λb*(V[bi]-V[ai]) + λc*(V[ci]-V[ai])
            end
        end
    end
    
    return v
end




end