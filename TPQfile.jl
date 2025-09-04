# module which contains all the microcanonical and canonical TPQ operations
# all internally optimized

module tpqStates

# dependencies
include("spinmodels.jl") # the Julia file which contains functions to generate the spin models
import SparseArrays
import LinearAlgebra
import CSV
import DataFrames
import KrylovKit
using .hamiltonians # the respective module name

export magnetization, spinZcorrelation, microcanonicalTPQ, canonicalTPQmag, canonicalTPQeng, canonicalTPQf

# returns the operation of total spin-z operator on an input state
function spinZ(N,state::Vector{ComplexF64})
    local Ψ = zeros(ComplexF64, 2^N)
    for i = 1:2^N
        local element::ComplexF64 = 0.0 + 0.0im
        for j = 0:N-1
            if (i-1) & (1 << j) != 0
                element -= 0.5
            else
                element += 0.5
            end
        end
        Ψ[i] = element * state[i] / N
    end
    return Ψ
end

# returns the operation of Sz operator on a selected set of spins
function spinZsector(N,state::Vector{ComplexF64},sector::Vector{Int})
    local Ψ = zeros(ComplexF64, 2^N)
    for i = 1:2^N
        local element::ComplexF64 = 0.0 + 0.0im
        for j = 0:N-1
            if (N-j) in sector
                if (i-1) & (1 << j) != 0
                    element -= 0.5
                else
                    element += 0.5
                end
            end
        end
        Ψ[i] = element * state[i] / length(sector)
    end
    return Ψ
end

# returns the magnetization of a pure state of a system
function magnetization(N,state::Vector{ComplexF64})

    local mz::ComplexF64 = 0.0 + 0.0im

    for i = 1:2^N
        local element::Float64 = 0.0
        for j = 0:N-1
            if (i-1) & (1 << j) != 0
                element -= 0.5
            else
                element += 0.5
            end
        end
        mz += element * conj(state[i])*state[i] / N
    end
    return mz
end

# returns the magnetization for a selected set of spins
function magsector(N,state::Vector{ComplexF64},sector::Vector{Int})

    local mz::ComplexF64 = 0.0 + 0.0im

    for i = 1:2^N
        local element::Float64 = 0.0
        for j = 0:N-1
            if (N-j) in sector
                if (i-1) & (1 << j) != 0
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


# returns the spin-spin correlation between two spins at "x" and "y" positions
function spinZcorrelation(N,x,y,state::Vector{ComplexF64})

    local spinX::Int
    local spinY::Int
    local correlation::ComplexF64 = 0.0 + 0.0im

    for i = 1:2^N
        spinX = Int((i-1) & (1 << (N-x)) != 0)
        spinY = Int((i-1) & (1 << (N-y)) != 0)
        if spinX == spinY
            correlation += conj(state[i])*state[i]
        else
            correlation -= conj(state[i])*state[i]
        end
    end
    return real(correlation)
end


# generate the T vs u data for specific N,J,hz values and store the data in a CSV file
# basically generate microcanonical TPQ states at finite temperatures
function microcanonicalTPQ(N::Int64,J::Float64,hz::Float64)

    local numerator::Float64 = 0.0
    local denominator::Float64 = 0.0
    local a::ComplexF64 = 0.0 + 0.0im
    local b::ComplexF64 = 0.0 + 0.0im
    local uDot::Float64 = 0.0
    local uInfinite::Float64 = 0.0

    hMatrix = (1/N) * hamiltonians.bipartiteEmbedding(Int(N/2),J,hz) # standard case of the Heisenberg hamiltonian
    # λmax, v, info = KrylovKit.eigsolve(hMatrix, 1, :LR)
    # l::Float64 = real(λmax[1])
    l::Float64 = 4*J

    uValues = Float64[] # store uk for the k-th iteration
    tempValues = Float64[] # corresponding temperature
    uThermodynamic = Float64[] # correction for the thermodynamic limit

    iterator = l*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N) .- hMatrix
    local Psi = randn(Float64,2^N) .+ im*randn(Float64,2^N) # generate an initial state, which is a Haar random state

    for k = 1:100
        Psi = iterator * Psi
        Psi /= LinearAlgebra.norm(Psi)
        uk::ComplexF64 = Psi' * (hMatrix * Psi)
        beta = 2*(k/N)/(l - real(uk))
        T = 1/beta
        push!(uValues,real(uk))
        push!(tempValues,T)

        # some process to calculate the error introduced, refer to PhysRevLett.108.240401 for reference
        a = Psi' * (hMatrix - uk*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N))^2 * Psi
        b = Psi' * (hMatrix - uk*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N))^3 * Psi
        uDot = real(uk - b/(2*a))
        numerator = real(b/(N*a^3)) + (4*k/N)/(l - uDot)^3
        denominator = -1/(real(N*a)) + (2*k/N)/(l-uDot)^2
        uInfinite = uDot + numerator/(2*N*denominator^2)
        push!(uThermodynamic,uInfinite)
    end
    df = DataFrames.DataFrame(u=uValues, T = tempValues, β= 1 ./ tempValues, uTherm=uThermodynamic)
    CSV.write("outputfile5.csv",df)
end



# chooses the value of l accurately for good spectral filtering
# this function requires the T vs β vs u data to be generated for all the mTPQ states
# βstate = the relevant value for convergence
function filtering(J::Float64, βstate::Float64, emax::Float64)
    local l::Float64
    if βstate > 1/10
        l = emax
    elseif 1/10 > βstate > 1/20
        l = J/10
    elseif 1/20 > βstate > 1/50
        l = J/2
    else
        l = J
    end
    return l
end

# return the magnetization for range of beta values; use the canonical thermal pure state
# the important thing here is to, choose the spectral filtering carefully => the value of "l"
function canonicalTPQmag(β::Float64,N::Int,J::Float64,hMatrix::SparseArrays.SparseMatrixCSC{ComplexF64, Int64})

    local kTerm::Int = (N+10)^2
    local k::Int = 0
    local Psi0 = randn(Float64,2^N) .+ im*randn(Float64,2^N)
    local Mz = Complex(BigFloat(0.0))
    local normState = Complex(BigFloat(0.0))
    local result = Complex(BigFloat(0.0))
    local resultConverted::ComplexF64
    local expectation::ComplexF64
    local l::Float64
    local logwt::BigFloat

    # spectral filtering
    λmax, v, info = KrylovKit.eigsolve(hMatrix, 1, :LR)  # :LR = Largest Real part
    l = filtering(J,β,λmax[1])
    iterator = l*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N) .- hMatrix

    local Psi1 = iterator * Psi0

    # selectedsites = collect(Int(N/2 + 1):N)

    while k < kTerm
        local norm0::ComplexF64 = LinearAlgebra.norm(Psi0)
        local norm1::ComplexF64 = LinearAlgebra.norm(Psi1)
        Psi0 /= norm0
        Psi1 /= norm1 
        local uk0::ComplexF64 = Psi0' * (hMatrix * Psi0) 
        local uk1::ComplexF64 = Psi1' * (hMatrix * Psi1) 
        local βref0 = (2*k/N)/(l-real(uk0))
        local βref1 = (2*(k+1)/N)/(l-real(uk1))

        expectation = magnetization(N,Psi0)
        logwt = (2*k)*log(N*β) - log(factorial(big(2*k))) + 2*log(big(real(norm0)))
        Mz += exp(logwt) * Complex{BigFloat}(expectation)
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi0) 

        expectation = Psi0' * spinZ(N,Psi1)
        logwt = (2*k+1)*log(N*β) - log(factorial(big(2*k+1))) + log(big(real(norm0))) + log(big(real(norm1)))
        Mz += exp(logwt) * Complex{BigFloat}(expectation)
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi1)

        if βref0 < β < βref1
           println("β0 = ",βref0,"β = ",β,"k = ",k)
            kTerm = k + N + Int(round(0.1*N))
        end
        k += 1
        Psi0 = iterator * Psi0
        Psi1 = iterator * Psi1
    end

    result = Mz / normState
    resultConverted = ComplexF64(Float64(real(result)), Float64(imag(result)))

    if isnan(real(result))
        println("NaN encountered in Mz,k = ",k)
        println(Mz)
        println(normState)
    end

    return real(resultConverted)
end

# return energy values for fixed beta values
function canonicalTPQeng(β::Float64,N::Int,J::Float64,hMatrix::SparseArrays.SparseMatrixCSC{ComplexF64, Int64})

    local kTerm::Int = (N+10)^2
    local k::Int = 0
    local Psi0 = randn(Float64,2^N) .+ im*randn(Float64,2^N)
    local uk = Complex(BigFloat(0.0))
    local normState = Complex(BigFloat(0.0))
    local result = Complex(BigFloat(0.0))
    local resultConverted::ComplexF64
    local expectation::ComplexF64
    local l::Float64
    local logwt::BigFloat 

    # spectral filtering
    λmax, v, info = KrylovKit.eigsolve(hMatrix, 1, :LR)  # :LR = Largest Real part
    l = filtering(J,β,λmax[1])
    iterator = l*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N) .- hMatrix

    local Psi1 = iterator * Psi0

    while k < kTerm
        local norm0::ComplexF64 = LinearAlgebra.norm(Psi0)
        local norm1::ComplexF64 = LinearAlgebra.norm(Psi1)
        Psi0 /= norm0
        Psi1 /= norm1 
        local uk0::ComplexF64 = Psi0' * (hMatrix * Psi0) 
        local uk1::ComplexF64 = Psi1' * (hMatrix * Psi1) 
        βref0 = 2*k/N/(l-real(uk0))
        βref1 = 2*(k+1)/N/(l-real(uk1))

        expectation = Psi0' * hMatrix * Psi0
        logwt = (2*k)*log(N*β) - log(factorial(big(2*k))) + 2*log(big(real(norm0)))
        uk += exp(logwt) * Complex{BigFloat}(expectation)
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi0)

        expectation = Psi0' * hMatrix * Psi1
        logwt = (2*k+1)*log(N*β) - log(factorial(big(2*k+1))) + log(big(real(norm0))) + log(big(real(norm1)))
        uk += exp(logwt) * Complex{BigFloat}(expectation) 
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi1) 

        if βref0 < β < βref1
            println("β0 = ",βref0,"β = ",β,"k = ",k)
            kTerm = k + N + Int(round(0.1*N))
        end
        k += 1
        Psi0 = iterator * Psi0
        Psi1 = iterator * Psi0
    end

    result = uk / normState
    resultConverted = ComplexF64(float(real(result)), float(imag(result)))
    if isnan(real(result))
        println("NaN encountered in u")
        println(uk)
        println(normState)
    end

    return real(resultConverted)
end

# return free energy for given β values
function canonicalTPQf(β::Float64,N::Int,J::Float64,hMatrix::SparseArrays.SparseMatrixCSC{ComplexF64, Int64})
    local kTerm::Int = (N+10)^2
    local k::Int = 0
    local Psi0 = randn(Float64,2^N) .+ im*randn(Float64,2^N)
    local normState = Complex(BigFloat(0.0))
    local resultConverted::ComplexF64
    local l::Float64
    local freeEnergy::Float64
    local logwt::BigFloat 

    # spectral filtering
    λmax, v, info = KrylovKit.eigsolve(hMatrix, 1, :LR)  # :LR = Largest Real part
    l = filtering(J,β,λmax[1])
    iterator = l*SparseArrays.sparse(LinearAlgebra.I,2^N,2^N) .- hMatrix

    local Psi1 = iterator * Psi0

    while k < kTerm
        local norm0::ComplexF64 = LinearAlgebra.norm(Psi0)
        local norm1::ComplexF64 = LinearAlgebra.norm(Psi1)
        Psi0 /= norm0
        Psi1 /= norm1 
        local uk0::ComplexF64 = Psi0' * (hMatrix * Psi0) 
        local uk1::ComplexF64 = Psi1' * (hMatrix * Psi1) 
        βref0 = 2*k/N/(l-real(uk0))
        βref1 = 2*(k+1)/N/(l-real(uk1))

        logwt = (2*k)*log(N*β) - log(factorial(big(2*k))) + 2*log(big(real(norm0)))
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi0) 
        logwt = (2*k+1)*log(N*β) - log(factorial(big(2*k+1))) + log(big(real(norm0))) + log(big(real(norm1)))
        normState += exp(logwt) * Complex{BigFloat}(Psi0'*Psi1)
        if βref0 < β < βref1
            println("β0 = ",βref0,"β = ",β,"k = ",k)
            kTerm = k + N + Int(round(0.1*N))
        end
        k += 1
        Psi0 = iterator * Psi0
        Psi1 = iterator * Psi0
    end

    resultConverted = ComplexF64(float(real(normState)), float(imag(normState)))
    if isnan(real(normState))
        println("NaN encountered in s")
        println(normState)
    end
    freeEnergy = -log(real(resultConverted))/(β*N) - log(2.0)/β
    return freeEnergy
end


# generic function for handling precision.
# input : Complex{BigFloat}()
# output : ComplexF64
function convert(z::Complex{BigFloat})
    try
        return ComplexF64(z)
    catch e
        if isa(e, InexactError)
            @warn "Precision loss during conversion"
            return ComplexF64(Float64(real(z)), Float64(imag(z)))
        else
            rethrow(e)
        end
    end
end

# some specialized function for calculating off chain spin magnetization
# bipartite embedding model


end
