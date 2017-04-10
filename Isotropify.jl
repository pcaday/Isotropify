#module Isotropify

include("Grids.jl")
include("ScatteredInterpolation.jl")
using Grids, ScatteredInterpolation


#export firstderivatives, secondderivatives, laplacebeltramidirichlet, scalarcurvature,
#       isotropify

# 2D anisotropic metrics are represented as 3-tuples of arrays
#   g = (g_11,g_12,g_22)


"First derivative operators on a given grid, with zero at the endpoints (second-order centered FDs)"
function firstderivatives(grid::RegularGrid)
    dx,dy = steps(grid)
    nx,ny = size(grid)

    ox = fill(0.5/dx, nx-1)
    Dx = spdiagm((ox,-ox), (1,-1))
    Dx[1,2] = 0
    Dx[end,end-1] = 0

    oy = fill(0.5/dy, ny-1)
    Dy = spdiagm((oy,-oy), (1,-1))
    Dy[1,2] = 0
    Dy[end,end-1] = 0

    return Dx,Dy
end

"Second derivative operators on a given grid, with zero at the endpoints (second-order centered FDs)"
function secondderivatives(grid::RegularGrid)
    dx,dy = steps(grid)
    nx,ny = size(grid)

    ox = fill(1/(dx^2), nx-1)
    Dx2 = spdiagm((ox,-2ox,ox), (1,0,-1))
    Dx2[1,1:2] = 0
    Dx2[end,end-1:end] = 0

    oy = fill(1/(dy^2), ny-1)
    Dy2 = spdiagm((oy,-2oy,oy), (1,0,-1))
    Dy2[1,1:2] = 0
    Dy2[end,end-1:end] = 0

    return Dx2,Dy2
end

"Operators that are identity on interior grid points and zero on the boundary"
function interioridentity(grid::RegularGrid)
    nx,ny = size(grid)

    Ix = speye(nx)
    Iy = speye(ny)

    Ix[1,    1] = 0
    Ix[end,end] = 0
    Iy[1,    1] = 0
    Iy[end,end] = 0

    return Ix,Iy
end

"Operators that are zero on interior grid points and identity on the boundary"
function boundaryrestriction(g::RegularGrid)
    Bx = sparse([1,nx], [1,nx], [1.,1.], nx, nx)      # Boundary operators, for boundary conditions
    By = sparse([1,ny], [1,ny], [1.,1.], ny, ny)
    
    return Bx,By
end

"""
Discrete Laplace-Beltrami operator on a grid, with Dirichlet boundary conditions

Sign convention: negative eigenvalues (agrees with the Laplacian for the Euclidean metric)
"""
function laplacebeltramidirichlet(g, grid::RegularGrid)
    # Check arguments
    nx,ny = size(grid)
    length(g) == 3 || throw(ArgumentError("Metric must have three components (only 2D supported)."))
    all(size(gij) == (nx,ny) for gij in g) || throw(ArgumentError("Dimensions of metric entries do not match grid"))

    # Unpack args
    g11,g12,g22 = (vec(gij) for gij in g)
        
    # Prepare matrices
    Dx,Dy   = firstderivatives(grid)
    Dx2,Dy2 = secondderivatives(grid)
    Ix,Iy   = interioridentity(grid)
    Bx,By   = boundaryrestriction(grid)
    
    invdetg = 1 ./ (g11.*g22 .- g12.^2)
    sinvdetg = sqrt(invdetg)

    FDx = kron(Iy,Dx)
    FDy = kron(Dy,Ix)

    return (Diagonal(invdetg) * (Diagonal(g22) * kron(Iy, Dx2)                            # Second derivatives
                              - 2Diagonal(g12) * kron(Dy, Dx)
                              +  Diagonal(g11) * kron(Dy2,Ix))
            + Diagonal(sinvdetg .* (FDx*(g22.*sinvdetg) - FDy*(g12.*sinvdetg))) * FDx     # First derivatives
            + Diagonal(sinvdetg .* (FDy*(g11.*sinvdetg) - FDx*(g12.*sinvdetg))) * FDy
            + kron(Iy,Bx) + kron(By,Ix) + kron(By,Bx))                                    # Boundary conditions
end

"Compute the scalar curvature of an anisotropic 2D metric"
function scalarcurvature(g, grid::RegularGrid)
    # Check arguments
    nx,ny = size(grid)
    length(g) == 3 || throw(ArgumentError("Metric must have three components (only 2D supported)."))
    all(size(gij) == (nx,ny) for gij in g) || throw(ArgumentError("Dimensions of metric entries do not match grid"))

    # Unpack args
    g11,g12,g22 = (vec(gij) for gij in g)

    # Get derivative matrices. This could be done more efficiently, but it should be fast enough anyway!
    Dx,Dy   = firstderivatives(grid)
    Dx2,Dy2 = secondderivatives(grid)
    Ix,Iy   = interioridentity(grid)
    
    FDx = kron(Iy,Dx)
    FDy = kron(Dy,Ix)
        
    # First derivatives of metric
    g11_1 = FDx*g11
    g11_2 = FDy*g11
    g22_1 = FDx*g22
    g22_2 = FDy*g22
    g12_1 = FDx*g12
    g12_2 = FDy*g12

    # Second derivatives of metric
    g11_22 = kron(Dy2,Ix )*g11
    g12_12 = kron(Dy, Dx )*g12
    g22_11 = kron(Iy, Dx2)*g22
    
    # 1/|g|
    invdetg = 1 ./ (g11.*g22 .- g12.^2)
    
    # The Christoffel symbols which are not simply a first derivative of the metric
    Γ_211 = 2g12_1 .- g11_2
    Γ_122 = 2g12_2 .- g22_1
    
    # One entry of Riemann curvature
    R1212 = (g12_12 .- 0.5*(g11_22.+g22_11)                        # Second derivative contributions
              .+ 0.25invdetg .* (g22 .* (g11_2.^2 - g11_1.*Γ_122)  # First derivative contributions
                              .+ g11 .* (g22_1.^2 - g22_2.*Γ_211)
                              .- g12 .* (2g11_2.*g22_1 - g11_1.*g22_2 - Γ_122.*Γ_211)))
    
    # Finally, scalar curvature
    return reshape(2R1212 .* invdetg, (nx,ny))
end

"""
Compute the conformally Euclidean (isotropic) metric isometric to a given metric.

The given metric g must be isotropic at the grid boundaries.
"""
function isotropify(g, grid::RegularGrid)
    # Compute scalar curvature and the Laplace-Beltrami operator for g
    S = scalarcurvature(g,grid)
    Δ = laplacebeltramidirichlet(g,grid)
    
    # Solve the Yamabe equation for the conformal factor λ = ln c, with Dirichlet
    #  boundary conditions.
    # The capital Λ indicates that we only know its value in the old coordinates
    
    g11 = g[1]
    
    S[:,  1] = -log(g11[:,  1])
    S[:,end] = -log(g11[:,end])
    S[1,  :] = -log(g11[1,  :])
    S[end,:] = -log(g11[end,:])
    
    Λ = Δ \ (0.5vec(S))
    
    # Get the new (flat) metric, G = e^(2λ)g, and its Laplace-Beltrami operator
    G = map(gij -> gij.*exp.(2reshape(Λ,size(grid))), g)
    Δ1 = laplacebeltramidirichlet(G,grid)
    
    # Solve Laplace-Beltrami equations for the isothermal coordinate functions X,Y
    x,y = vars(grid)
    
    # First x:
    z = zeros(S)
    z[1,  :] = x[1]
    z[end,:] = x[end]
    z[:,  1] = x
    z[:,end] = x
    
    X = Δ1 \ vec(z)
    
    # Then y:
    z[1,  :] = y
    z[end,:] = y
    z[:,  1] = y[1]
    z[:,end] = y[end]
    
    Y = Δ1 \ vec(z)

    # Clamp in case of numerical error:
    clamp!(X, x[1], x[end])
    clamp!(Y, y[1], y[end])
    
    # Change coordinates by scattered interpolation
    λ = scatteredinterpolate(X,Y,Λ,x,y)
    return reshape(exp.(λ), size(grid))
end

"""
Compute the conformally Euclidean (isotropic) metric isometric to a given metric.

The given metric g must be isotropic at the grid boundaries.
"""
function isotropify2(g, grid::RegularGrid)
    # Unpack args
    g11,g12,g22 = g
        
    # In 2D, the isothermal coordinate functions X,Y are harmonic.
    # Solve Laplace-Beltrami equations for them.
    Δ = laplacebeltramidirichlet(g,grid)
    x,y = vars(grid)
    
    # First X:
    z = zeros(g11)
    z[1,  :] = x[1]
    z[end,:] = x[end]
    z[:,  1] = x
    z[:,end] = x
    
    X = Δ \ vec(z)
    
    # Then Y:
    z[1,  :] = y
    z[end,:] = y
    z[:,  1] = y[1]
    z[:,end] = y[end]
    
    Y = Δ \ vec(z)

    # Clamp in case of numerical error:
    clamp!(X, x[1], x[end])
    clamp!(Y, y[1], y[end])
    
    # Get derivative matrices... could be done better and/or faster
    Dx,Dy = firstderivatives(grid)
    Ix,Iy = interioridentity(grid)
    
    FDx = kron(Iy,Dx)
    FDy = kron(Dy,Ix)

    # Calculate Jacobian of coordinate transformation
    J = reshape((FDx*X) .* (FDy*Y) .- (FDx*Y) .* (FDy*X), (nx,ny))
    J[1  ,:] = 1
    J[end,:] = 1
    J[:,  1] = 1
    J[:,end] = 1
        
    # Calculate c in original coordinates
    C = vec(sqrt(J ./ sqrt(g11.*g22 - g12.^2)))
    
    # Change coordinates by scattered interpolation
    c = scatteredinterpolate(X,Y,C,x,y)
    return reshape(c, (nx,ny))
end


#end