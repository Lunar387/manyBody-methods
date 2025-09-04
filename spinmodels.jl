# A module which contains the functions to generate sparse hamiltonian matrices 
module hamiltonians 

# dependencies
import SparseArrays
import LinearAlgebra

export toBinary, flip, Heisenberg, bipartiteEmbedding

# a function to convert integer to a binary 
function toBinary(num,bits)
    BitVector(digits(num, base=2, pad=bits) |> reverse)
end

# a function to flip two spin-indices and return the new label 
function flip(num,bits,x,y)
    local newbasis::Int64 
    newbasis = (num-1) ⊻ ((1 << (bits-x))|(1 << (bits-y)))
    return newbasis+1
end


# a function to generate Heisenberg model 
function Heisenberg(N::Int,J::Float64,hz::Float64) 

    H = SparseArrays.spzeros(Float64,2^N,2^N)

    for label = 1:2^N
        bitValue = toBinary(label-1,N)
        local field::Float64 = 0.0
        for i = 1:N 
            j = i%N + 1
            
            if bitValue[i] == bitValue[j]
                H[label,label] += J/4
            else
                H[label,label] -= J/4 
                flippedLabel = flip(label,N,i,j)
                H[label,flippedLabel] = J/2
            end

            if bitValue[i] == 0
                field += 0.5 
            else
                field -= 0.5
            end
        end

        H[label,label] += field * hz
    end
    return H 
end

# generate the Bipartite embedding model
# the chain is basically L+L, with bipartite structure, XY on-chain coupling and ZZ off-chain coupling
# J = Jzz/Jxy
function bipartiteEmbedding(L::Int,J::Float64,hz::Float64)
    
    local Jzz::Float64 = -J 
    local Jxy::Float64 = 1.0
    N::Int = 2*L 
    H = SparseArrays.spzeros(ComplexF64,2^N,2^N)
    for label = 1:2^N
        bitValue = toBinary(label-1,N)
        local field::Float64 = 0.0
        for i = 1:L 
            j = i%L + 1
            k = i + L
            if bitValue[i] == bitValue[k]
                H[label,label] += Jzz/4
            else 
                H[label,label] -= Jzz/4
            end
            if bitValue[i] ≠ bitValue[j] 
                flippedLabel = flip(label,N,i,j)
                H[label,flippedLabel] = Jxy/2
            end
        end

        # if we add an external field in the z direction
        for i = 1:N
            if bitValue[i] == 0
                field += 0.5 
            else
                field -= 0.5
            end
        end

        H[label,label] += field * hz
    end
    return H 
end

# generate the Non-bipartite embedding model
# the chain is basically L+L, with triangular motifs, XY on-chain coupling and ZZ off-chain coupling
# J = Jzz/Jxy
function NonbipartiteEmbedding(L::Int,J::Float64,hz::Float64)

    local Jzz::Float64 = -J 
    local Jxy::Float64 = 1.0
    N::Int = 2*L 
    H = SparseArrays.spzeros(ComplexF64,2^N,2^N)
    for label = 1:2^N
        bitValue = toBinary(label-1,N)
        local field::Float64 = 0.0
        for i = 1:N 
            j = i%N + 1
            k = (i+2)%N 
            if bitValue[i] == bitValue[j]
                H[label,label] += Jzz/4
            else 
                H[label,label] -= Jzz/4
            end
            if bitValue[i] ≠ bitValue[k] && i%2==1
                flippedLabel = flip(label,N,i,j)
                H[label,flippedLabel] = Jxy/2
            end
        end

        # if we add an external field in the z direction
        for i = 1:N
            if bitValue[i] == 0
                field += 0.5 
            else
                field -= 0.5
            end
        end

        H[label,label] += field * hz
    end
    return H 
end


end 

