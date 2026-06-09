# Dynamics of the XXZ spin chain at infinite temperature
# Using ancilla purification method 

# This method creates an equal number of ancilla spins, which are maximally entangled with the physical spins 
# Then it applies the checkerboard TEBD method for time evolution


# FUNCTION TO DEFINE THE INDICES OF PHYSICAL AND ANCILLA SPINS 

function definebothindices(L::Int64)

    Pindices = siteinds("S=1/2", L; addtags="physical")
    Aindices = siteinds("S=1/2", L; addtags="ancilla")

    return Pindices, Aindices
end

# CREATE A TNS FOR PURIFICATION METHODS 
# (P = A) -- (P = A) -- (P = ...
#  |1         |2         |3  ... n
#  G          G          G   ...
# P is entangled with A in a bell pair, denoted by β

function doublestate(physindices, ancindices, L::Int64)

    tensors = Vector{ITensor}(undef, 2*L)
 
    BondL = nothing 

    v = 1.0/sqrt(2.0)

    for i in 1:2*L 

        n = i%2 + i÷2 # actual ordering of the P=A pair (n) 

        if i%2 == 1 # odd sites of the combined chain contain physical indices

            sp = physindices[n]
            interlink = Index(2,"Link,$i") # create a bell pair with the immediate ancilla
            BondR = interlink
            
            if i == 1

                Tp = ITensor(sp,BondR)
                
                for (k,m) in enumerate(["Up", "Dn"])
                    Tp[sp=>m, BondR=>k] = v
                end
            
            else

                Tp = ITensor(sp, BondR, BondL)

                for (k,m) in enumerate(["Up", "Dn"])
                    Tp[sp=>m, BondR=>k, BondL=>1] = v
                end

            end

            tensors[i] = Tp

        else

            sa = ancindices[n]
            interlink = Index(1,"Link,$i") # trivial connection with next physical qubit
            BondR = interlink  

            if i == 2*L 

                Ta = ITensor(sa, BondL)

                for (k,m) in enumerate(["Up", "Dn"])
                    Ta[sa=>m, BondL=>k] = v 
                end

            else

                Ta = ITensor(sa, BondL, BondR)

                for (k,m) in enumerate(["Up", "Dn"])
                    Ta[sa=>m, BondL=>k, BondR=>1] = v 
                end

            end

            tensors[i] = Ta 
        end

        BondL = BondR 
    end

    psi = MPS(tensors)
    normalize!(psi)
    return psi

end
