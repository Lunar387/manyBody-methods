#  -----------------------------------------------------------------------
# | simple time evolution using Trotter gates in XXZ-checkerboard circuit |
#  -----------------------------------------------------------------------

using ITensors 
using ITensorMPS 

# generate the ITensor indices for a spin 1/2 system, based on odd-even site indices              ---- //
# for a chain of length 2*L

function buildindices(L::Int64)
    
    return siteinds("S=1/2", 2*L)

end
      
# H = σ_x(i)σ_x(i+1) + σ_y(i)σ_y(i+1) + Δ*σ_z(i)σ_z(i+1) 
# G = exp(-iτH)
# break it up into even and odd sites                                                             ---- //

function applyGates(L::Int64, indices, delta::Float64, tau::Float64)

    Oddgates = ITensor[]
    Evengates = ITensor[] 

    for (k,m) in enumerate(indices)
        p = k 
        q = k+1 

        if k < 2*L

            n = indices[q]

            hlocal = delta * op("Sz", m) * op("Sz", n) +      # build the local 2-site hamiltonian 
                        1 / 2 * op("S+", m) * op("S-", n) +
                        1 / 2 * op("S-", m) * op("S+", n)

            if p%2 == 1  # applying the odd gates 

                Godd = exp(-1.0im * tau * hlocal)
                push!(Oddgates, Godd)
            
            else         # applying the even gates

                Geven = exp(-1.0im * tau * hlocal)
                push!(Evengates, Geven)
            
            end
        end

    end

    return Oddgates,Evengates 

end

# apply the gates to the tensor product state and calculate some observable                       ---- //
# here we show : 1. < Sz(t) >    

function executeCircuit(L::Int64, tau::Float64, ttotal::Float64, delta::Float64)

    N = 2*L 

    # create physical site indices for a chain of length = 2*L 

    physicalindex = buildindices(L)
    
    # create an initial product state

    Psit = MPS(physicalindex, n -> isodd(n) ? "Up" : "Dn")

    # create even and odd sets of gates 
    
    Go, Ge = applyGates(L, physicalindex, delta, tau)

    # apply alternately the odd and even gates 
    
    for t in 0.0:tau:ttotal

        orthogonalize!(Psit, L)
        wf = Psit[L] * Psit[2*L]

        Sz1 = op("Sz", physicalindex[L])
        Sz2 = op("Sz", physicalindex[2*L])
        
        wf_op = Sz1 * Sz2 * wf

        val = real(scalar(dag(prime(wf, "Site")) * wf_op))
        println("$t $val")

        t≈ttotal && break

        Psit = apply(Go, Psit, cutoff=1.0E-08, maxdim=200)
        Psit = apply(Ge, Psit, cutoff=1.0E-08, maxdim=200)

        normalize!(Psit)
    end
end

let 
    L=50
    tau=0.1
    Δ = 1.0
    ttotal = 10.0

    executeCircuit(L,tau,ttotal,Δ)
    
end


    

