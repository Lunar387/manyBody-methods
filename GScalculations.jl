# Here we perform ground state calculations on symmetrized hamiltonians 
module EDcalc 


include("spinmodels.jl")
using .hamiltonians 

import KrylovKit
import LinearAlgebra 
import SparseArrays 
import DataFrames
import CSV 
using JLD2

export Diagonalize, Szcorrelation, magsector

# use standard libraries for lanczos diagonalization of generated hamiltonian 
# find and store the lowest eigenvalues and ground state 
function Diagonalize(hMatrix::SparseArrays.SparseMatrixCSC{ComplexF64, Int64}, L::Int64, Sz::Int64, m::Int64)

    dim::Int64 = size(hMatrix, 1)
    N::Int64 = 2*L 
    local groundstate = zeros(ComplexF64, dim)
    
    foldername = "kagomedata$(N)"
    if !isdir(foldername)
        mkdir(foldername)   
    end

    nev = 2
    if dim == 1
        df = DataFrames.DataFrame(evals = [real(hMatrix[1,1])])
        groundstate[1] = 1.0 + 0.0im 
    else
        vals, vecs, info = KrylovKit.eigsolve(hMatrix, rand(ComplexF64, dim), nev, :SR, issymmetric=true)
        df = DataFrames.DataFrame(evals = real.(vals))
        groundstate = vecs[1]
    end

    filename = "energies_$(Sz)_$(m)_kagome.csv"
    filepath = joinpath(foldername, filename)
    filename2 = "gs_$(Sz)_$(m)_kagome.jld2"
    filepath2 = joinpath(foldername, filename2)
    
    CSV.write(filepath, df)
    @save filepath2 groundstate 

end


#--------------------------------------------
#            | OBSERVABLES |
#--------------------------------------------

# find the ZZ correlation between x and y 'th spins 
# returns the ground state correlator
function Szcorrelation(L::Int64, x::Int64, y::Int64, state::Vector{ComplexF64}, Sz::Int64, m::Int64)

    local spinX::Int64
    local spinY::Int64
    local correlation::ComplexF64 = 0.0 + 0.0im
    local N::Int64 = 2*L

    # load the basis states 
    @load "Alltranslations_$(N)/label_$(Sz)_$(m).jld2" cpfinal 
    momentabasis = cpfinal[(Sz,m)]

    for i in eachindex(momentabasis)

        label = momentabasis[i]
        spinX = Int((label) & (1 << (N-x)) != 0)
        spinY = Int((label) & (1 << (N-y)) != 0)
        if spinX == spinY
            correlation += conj(state[i])*state[i]
        else
            correlation -= conj(state[i])*state[i]
        end
    end
    return real(correlation)
end

# function for XX correlators
function Sxcorrelation(L::Int, x::Int, y::Int, state::Vector{ComplexF64}, Sz::Int, m::Int)

    local N = 2L
    local corr = 0.0 + 0.0im

    # load momentum basis (Array of bitstrings)
    @load "Alltranslations_$(N)/label_$(Sz)_$(m).jld2" cpfinal
    basis = cpfinal[(Sz,m)]

    # mapping from label → basis index
    indexmap = Dict{Int,Int}()
    for (i,lab) in enumerate(basis)
        indexmap[lab] = i
    end

    # bit positions
    bx = N - x
    by = N - y

    mask = (1 << bx) ⊻ (1 << by)   # flips x and y bits

    for (i,lab) in enumerate(basis)
        flipped = lab ⊻ mask

        if haskey(indexmap, flipped)
            j = indexmap[flipped]
            corr += conj(state[i]) * state[j]
        end
    end

    return real(corr)
end

# returns the magnetization for a selected subset of spins
function magsector(L::Int64, sector::Vector{Int}, state::Vector{ComplexF64}, Sz::Int64, m::Int64)

    local mz::ComplexF64 = 0.0 + 0.0im
    local N::int64 = 2*L

    # load the basis states 
    @load "Alltranslations_$(N)/label_$(Sz)_$(m).jld2" cpfinal 
    momentabasis = cpfinal[(Sz,m)]

    for i in eachindex(momentabasis)
        local element::Float64 = 0.0
        label = momentabasis[i]
        for j = 0:N-1
            if (N-j) in sector
                if (label) & (1 << j) != 0
                    element -= 0.5
                else
                    element += 0.5
                end
            end
        end
        mz += element * conj(state[i])*state[i] / length(sector)
    end
    return mz
end

end 
