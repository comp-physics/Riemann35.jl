"""
    halo_exchange_3d!(A::Array{T,4}, decomp, bc::Symbol) where T

Exchange halos for a 3D subdomain with nv variables (no z decomposition).

# Arguments
- `A`: Local array (nx+2h, ny+2h, nz, nv), interior: A[h+1:h+nx, h+1:h+ny, :, :]
- `decomp`: Struct from setup_mpi_cartesian_3d
- `bc`: Boundary condition type (default: :copy)

# Returns
- Modifies `A` in place with exchanged halos

# Notes
- Performs left/right exchange first, then up/down exchange
- Applies physical BC at global boundaries before exchange
- No exchange in z-direction (all ranks have full z extent)
- Corners are filled implicitly after second phase

# Example
```julia
A = zeros(nx+2*halo, ny+2*halo, nz, Nmom)
halo_exchange_3d!(A, decomp, :copy)
```
"""
function halo_exchange_3d!(A::Array{T,4}, decomp, bc) where T
    h = decomp.halo
    nx = decomp.local_size[1]
    ny = decomp.local_size[2]
    nz = decomp.local_size[3]
    
    if h == 0
        return A
    end
    
    # Apply physical BC to halos at global boundaries before exchange
    apply_physical_bc_3d!(A, decomp, bc)
    
    # Single rank - no exchange needed
    if MPI.Comm_size(MPI.COMM_WORLD) == 1
        return A
    end
    
    comm = MPI.COMM_WORLD
    
    # X-direction exchange using non-blocking send/receive
    left_neighbor = decomp.neighbors.left
    right_neighbor = decomp.neighbors.right
    
    # Post all sends first, then do receives
    reqs = MPI.Request[]
    
    # Send interior boundary data to neighbors (includes all z and all variables)
    if left_neighbor != -1
        send_buf = A[h+1:h+h, h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=left_neighbor, tag=0)
        push!(reqs, req)
    end
    
    if right_neighbor != -1
        send_buf = A[h+nx-h+1:h+nx, h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=right_neighbor, tag=1)
        push!(reqs, req)
    end
    
    # Receive halo data from neighbors
    if left_neighbor != -1
        recv_buf = similar(A, h, ny, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=left_neighbor, tag=1)
        MPI.Wait(req)
        A[1:h, h+1:h+ny, :, :] = recv_buf
    end
    
    if right_neighbor != -1
        recv_buf = similar(A, h, ny, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=right_neighbor, tag=0)
        MPI.Wait(req)
        A[h+nx+1:h+nx+h, h+1:h+ny, :, :] = recv_buf
    end
    
    # Wait for all sends to complete
    MPI.Waitall(reqs)
    
    # Y-direction exchange using non-blocking send/receive
    down_neighbor = decomp.neighbors.down
    up_neighbor = decomp.neighbors.up

    reqs = MPI.Request[]

    # Corner-consistent exchange for the order-3 path (halo > 2): the second (y)
    # phase carries the FULL x-extent — including the x-ghost columns just filled by
    # the x-phase — so the diagonal/corner ghosts (x-ghost ∩ y-ghost) are propagated
    # from the diagonal neighbour. Orders 1/2 always call with h == 2 and keep the
    # historical interior-x-only y-exchange (byte-identical). Corner ghosts are read
    # only by the order-3 rank-boundary θ layer, never by the order-1/2 residual.
    xr = (h > 2) ? (1:nx+2h) : (h+1:h+nx)
    nxs = length(xr)

    # Send interior boundary data to neighbors (includes all z and all variables)
    if down_neighbor != -1
        send_buf = A[xr, h+1:h+h, :, :]
        req = MPI.Isend(send_buf, comm; dest=down_neighbor, tag=2)
        push!(reqs, req)
    end

    if up_neighbor != -1
        send_buf = A[xr, h+ny-h+1:h+ny, :, :]
        req = MPI.Isend(send_buf, comm; dest=up_neighbor, tag=3)
        push!(reqs, req)
    end

    # Receive halo data from neighbors
    if down_neighbor != -1
        recv_buf = similar(A, nxs, h, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=down_neighbor, tag=3)
        MPI.Wait(req)
        A[xr, 1:h, :, :] = recv_buf
    end

    if up_neighbor != -1
        recv_buf = similar(A, nxs, h, nz, size(A, 4))
        req = MPI.Irecv!(recv_buf, comm; source=up_neighbor, tag=2)
        MPI.Wait(req)
        A[xr, h+ny+1:h+ny+h, :, :] = recv_buf
    end

    # Wait for all sends to complete
    MPI.Waitall(reqs)

    # Re-apply physical BC after the exchange (order-3 only, h > 2) so that corners
    # at a GLOBAL transverse boundary copy the now-fresh interior edge (the pre-
    # exchange BC copied stale x-ghost values into those corners). This makes every
    # ghost cell — corners included — identical to the value the owning/boundary rank
    # computes, which the rank-boundary θ layer relies on. No-op for orders 1/2.
    if h > 2
        apply_physical_bc_3d!(A, decomp, bc)
    end

    return A
end

"""
    apply_physical_bc_3d!(A::Array{T,4}, decomp, bc::Symbol) where T

Fill halos at global boundaries based on bc type.

# Arguments
- `A`: Local array (nx+2h, ny+2h, nz, nv)
- `decomp`: Domain decomposition struct
- `bc`: Boundary condition type

# Supported bc types
- `:copy` - Neumann-like (copy nearest interior cell)

# Notes
- Also applies copy BC in z-direction at global z boundaries (no decomposition in z)
"""
function apply_physical_bc_3d!(A::Array{T,4}, decomp, bc) where T
    h = decomp.halo
    h == 0 && return A
    faces = expand_bc(bc)                       # canonical six-face spec (see face_bc.jl)
    nv = size(A, 4)

    # inlet Maxwellian, fetched once if any face is an inlet
    inlet = nothing
    if has_inlet(faces)
        inlet = CROSSFLOW_INLET[]
        inlet === nothing &&
            error(":inlet face needs CROSSFLOW_INLET[] set (inlet Maxwellian moment vector)")
        length(inlet) == nv ||
            error(":inlet length $(length(inlet)) != nvar $nv")
    end

    # Direction-agnostic refill over the CPU-haloed spatial axes (x=1, y=2). The z
    # axis (dim 3) carries no halo padding here, so its BC is applied in the flux
    # update, not by this refill. Faces are applied x-then-y so corner ghosts get
    # the same final writer as the legacy hand-written branches (byte-identical for
    # the :copy and :crossflow presets). A :sponge face is :outflow at the halo;
    # its absorbing source lives in the interior (see sponge.jl).
    axinfo = ((1, :xlo, :xhi, :left, :right), (2, :ylo, :yhi, :down, :up))
    for (a, lokey, hikey, lonb_f, hinb_f) in axinfo
        na    = decomp.local_size[a]
        lonb  = getproperty(decomp.neighbors, lonb_f)
        hinb  = getproperty(decomp.neighbors, hinb_f)
        lotyp = faces[lokey]
        hityp = faces[hikey]

        if lotyp === :periodic                  # (hityp === :periodic guaranteed by expand_bc)
            if lonb == -1 && hinb == -1
                for ih in 1:h
                    selectdim(A, a, ih)          .= selectdim(A, a, na + ih)   # lo ghost <- hi interior
                    selectdim(A, a, h + na + ih) .= selectdim(A, a, h + ih)     # hi ghost <- lo interior
                end
            elseif lonb == -1 || hinb == -1
                error("periodic BC on axis $a requires a single rank spanning that axis " *
                      "(multi-rank wrap-around topology is not supported)")
            end
            continue
        end

        # lo face
        if lonb == -1
            _refill_face!(A, a, 1:h, h + 1, halo_face_type(lotyp), inlet, nv)
        end
        # hi face
        if hinb == -1
            _refill_face!(A, a, (h + na + 1):(h + na + h), h + na, halo_face_type(hityp), inlet, nv)
        end
    end

    return A
end

# Fill the ghost planes `dst_idxs` along spatial axis `a` per face type `typ`,
# copying from interior plane `src_idx` for :outflow (zero-gradient). Axis-generic
# via `selectdim`; byte-identical to the explicit `A[ih,:,:,:] .= view(...)` form.
@inline function _refill_face!(A, a::Int, dst_idxs, src_idx::Int, typ::Symbol, inlet, nv::Int)
    if typ === :inlet
        @inbounds for d in dst_idxs
            sd = selectdim(A, a, d)             # view over (other-spatial..., var); var is the last dim
            vd = ndims(sd)
            for m in 1:nv
                selectdim(sd, vd, m) .= inlet[m]
            end
        end
    else                                        # :outflow (zero-gradient)
        for d in dst_idxs
            selectdim(A, a, d) .= selectdim(A, a, src_idx)
        end
    end
    return A
end

