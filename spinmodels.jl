# A module which contains the functions to generate sparse hamiltonian matrices 
module hamiltonians 

# dependencies
import SparseArrays
import LinearAlgebra
using JLD2 

export toBinary, flip, Heisenberg, bipartiteEmbedding, NonbipartiteEmbedding, SpinSector, ShiftTranslate, OCsector, kagomeSymmetrized

# a function to convert integer to a binary 
function toBinary(num,bits)
    BitVector(digits(num, base=2, pad=bits) |> reverse)
end

# a function to flip two spin-indices and return the new label 
function flip(num::Int, bits::Int, x::Int, y::Int)
    mask = (1 << (bits - x)) | (1 << (bits - y))
    return num ⊻ mask
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
    
    local Jzz::Float64 = J 
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

    local Jzz::Float64 = J 
    local Jxy::Float64 = 1.0
    N::Int64 = 2*L 
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
            if i%2==1
                if (bitValue[i] ≠ bitValue[k])
                    flippedLabel = flip(label,N,i,k)
                    H[label,flippedLabel] = Jxy/2
                end
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



#-----------------------------------------------------
#             | APPLYING SYMMETRIES |
#-----------------------------------------------------


#--------------------------------------------
#              | HELPER FUNCTIONS |
#--------------------------------------------

# function to find all the representatives for a given system size, wrt different spin sectors 
function SpinSector(L::Int64)

    local N::Int64 = 2*L
    d = Dict{Int, Vector{Int64}}()
    dir = "SpinBasis_$(N)"
    mkpath(dir)

    for i = 0:(2^N - 1)

        Nup = count(toBinary(i, N))
        Sz = L - Nup 

        if haskey(d, Sz)
            push!(d[Sz], i)
        else
            push!(get!(d, Sz, Int64[]), i)
        end
    end

    for Stot = 0:L 
        cp = Dict(Stot => d[Stot])
        filepath = joinpath(dir, "label_$(Stot).jld2")
        @save filepath cp
    end

end

# function to find different labels fir a given system size, but conserved off-chain spin sectors 
function OCsector(L::Int64)

    local N::Int64 = 2*L
    d = Dict{Int, Vector{Int64}}()
    dir = "SpinBasis_$(N)"
    mkpath(dir)

    finalspin = Int64(L/2)

    for i = 0:(2^N - 1)

        bits = toBinary(i, N)
        Nup = sum(bits[2:2:end])
        Sz = finalspin - Nup 

        if haskey(d, Sz)
            push!(d[Sz], i)
        else
            push!(get!(d, Sz, Int64[]), i)
        end
    end

    for Stot = 0:finalspin 
        cp = Dict(Stot => d[Stot])
        filepath = joinpath(dir, "label_$(Stot).jld2")
        @save filepath cp
    end

end


# generate translations, period, representative label for a set of labels
function ShiftTranslate(L::Int64)

    local N::Int64 = 2*L 

    dir = "Alltranslations_$(N)"
    mkpath(dir)

    for Stot = 0:L 
        
        @load "SpinBasis_$(N)/label_$(Stot).jld2" cp 
        list = cp[Stot]

        cpt = Dict{Int, Tuple{Vector{Int}, Int, Int}}()

        for label in list 
            bitvalue = toBinary(label, N)

            local translate::Vector{Int64} = []
            local period::Int64 = 0
            local irrep::Int64 = -1
            local minrep::Int64 = label

            while irrep ≠ label 

                shiftbits = circshift(bitvalue, 4)
                irrep = foldl((x, y) -> x << 1 | y, shiftbits)
                period += 1

                if irrep < minrep
                    minrep = irrep 
                end

                bitvalue = shiftbits
                push!(translate, irrep)

            end

            cpt[label] = translate, period, minrep
        end
        
        filepath = joinpath(dir, "label_$(Stot).jld2")
        @save filepath cpt 

    end
end

# function to search for a label(if present) in a list with bisection search 
function SearchLabel(label::Int64, list::Vector{Int64})
    
    truthvalue = false
    index = 0
    
    if length(list) == 0 
        
        return truthvalue,0

    elseif length(list) == 1
         
        return isequal(list[1], label),1
    
    else
        
        min = 1
        max = length(list)
        truthvalue = isequal(list[min], label) || isequal(list[max], label)
        index = isequal(list[min], label) == true ? min : isequal(list[max], label) == true ? max : 0
        
        if truthvalue == false
            while min ≠ max-1 
                
                index = floor(Int64,(min+max)/2)
                midval = list[index]
                
                if midval == label
                    
                    truthvalue = true 
                    break 
                
                elseif midval < label 

                    min = floor(Int64,(min+max)/2)

                else

                    max = floor(Int64,(min+max)/2)
                end
            end
        end

        if truthvalue == false
            truthvalue = isequal(list[min], label) || isequal(list[max], label)
            index = isequal(list[min], label) == true ? min : isequal(list[max], label) == true ? max : 0 
        end

    end

    return truthvalue, index 
end

# function to insert the representative labels in ascending order
# while avoiding duplicates 
function insert_sorted!(A::Vector{T}, v::T) where T
    i = searchsortedfirst(A, v)
    if i > length(A) || A[i] != v
        insert!(A, i, v)
    end
    return A
end


#-----------------------------------------------------
#            | BUILDING THE HAMILTONIAN |
#-----------------------------------------------------


# build the actual hamiltonian for the kagome type chain 
function kagomeSymmetrized(L::Int, Jr::Float64, hz::Float64, Sz::Int64, mval::Int64)

    local Jzz::Float64 = -1.0*Jr
    local Jxy::Float64 = -1.0
    local kval::Float64 = 8*3.1415926535897931*mval / L 
    N::Int64 = 2*L 

    dir = "Alltranslations_$(N)" 
    cpfinal = Dict{Tuple{Int,Int}, Vector{Int64}}()

    # now apply the symmetries 
    @load "Alltranslations_$(N)/label_$(Sz).jld2" cpt 

    # build the basis set 
    lbls = collect(keys(cpt))
    local validlabels::Vector{Int64} = []

    for a in lbls 
        Ta, Pa, Ra = cpt[a]

        num = kval * Pa / (4*3.1415926535897931)
        
        if num*100 == round(num)*100
            insert_sorted!(validlabels, Ra)
        end
    end

    cpfinal[(Sz,mval)] = validlabels
    filepath = joinpath(dir, "label_$(Sz)_$(mval).jld2")
    @save filepath cpfinal 
    
    # now build the actual hamiltonian 
    local dims::Int64 = length(validlabels)
    
    H = SparseArrays.spzeros(ComplexF64, dims, dims)

    for sindex in eachindex(validlabels) 

        sbasis = validlabels[sindex]
        Torg, Porg, Rorg = cpt[sbasis]
        bitval = toBinary(sbasis, N)
        
        for i = 1:N 

            local field::Float64 = 0.0
            local lshifts::Int64 = 0
            local P::Int64 = 0
            local posn::Int64 = 0

            j = i%N + 1 
           
            if bitval[i] == bitval[j]
                H[sindex,sindex] += Jzz/4
            else 
                H[sindex,sindex] -= Jzz/4
            end
            
            if i%2==1
                
                k = (i+2)%N 
                
                if (bitval[i] ≠ bitval[k])
                    flippedbasis = flip(sbasis,N,i,k)

                    # now search for the irrep of the newlabel
                    T, P, R = cpt[flippedbasis]

                    lshifts = findfirst(x -> x == R, T) # shifts reqd to obtain a representative 
                    truval, posn = SearchLabel(R, validlabels) # the new index within the subspace 

                    if posn ≠ 0
                        val = exp(-1.0im * kval * lshifts) * sqrt(Porg / P) * Jxy/2
                        H[posn,sindex] += val 
                        H[sindex,posn] += conj(val)
                    end

                end

            end

            # if we add an external field in the z direction
            for i = 1:N
                if bitval[i] == 0
                    field += 0.5 
                else
                    field -= 0.5
                end
            end

            H[sindex,sindex] += field * hz

        end
    end

    return H 

end

end
