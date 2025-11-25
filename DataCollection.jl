
# this file is meant for data collection and final calculations 
include("spinmodels.jl")
include("GScalculations.jl")
using .hamiltonians 
using .EDcalc 

import LinearAlgebra 
import SparseArrays 
import DataFrames
import CSV 
import Plots 
using JLD2 

L = 12

# function to generate basis for all translation and Z_2 symmetries 

# hamiltonians.OCsector(L) # generates all basis for corresponding hamming weights for off-chain spins 
# hamiltonians.ShiftTranslate(L) # generates all possible translations of a given set of basis 


# function to find the eigen-energies for each symmetry sector
function findenergies(L::Int64, Jratio::Float64, hzfield::Float64)

    local N::Int64 = 2*L
    quartlength = Int64(L/4)
    halflength = Int64(L/2)

    for totalspin = 0:halflength 
        for translations = 0:(quartlength - 1)
            
            Hmat = hamiltonians.kagomeSymmetrized(L, Jratio, hzfield, totalspin, translations) # create the hamiltonian desired for the subspace

            dimension = size(Hmat, 1)
            
            if dimension ≠ 0
                EDcalc.Diagonalize(Hmat, L, totalspin, translations) #finds and stores lowest energy and state in each unit cell momentum sector 
            end

        end
    end

end

function towerofstates(L::Int64)

    N = 2*L
    quartlength = Int(L/4)
    halflength = Int(L/2)

    plt = Plots.plot(
        title="Energy spectra",
        xlabel="Sz",
        ylabel="E",
        legend=true,
    )

    label0_used = false      # track if label “0” already added
    labelπ_used = false      # track if label “π” already added

    for totalspin = 0:halflength 
        for translations = 0:(quartlength - 1)

            foldername = "kagomedata$(N)_0_5"
            filename = "energies_$(totalspin)_$(translations)_kagome.csv"
            filepath = joinpath(foldername, filename)

            if isfile(filepath)
                df = CSV.read(filepath, DataFrames.DataFrame)
                Egs = df.evals[1]

                spin = totalspin

                if translations == 0
                    Plots.scatter!(
                        plt,
                        [spin], [Egs],
                        markersize=8,
                        marker=:circle,
                        markercolor=:red,
                        label = label0_used ? "" : "0",
                        # ylims = (-7.728,-7.727)
                    )
                    label0_used = true
                else
                    Plots.scatter!(
                        plt,
                        [spin], [Egs],
                        markersize=8,
                        marker=:star5,
                        markercolor=:blue,
                        label = labelπ_used ? "" : "π",
                        # ylims = (-7.728,-7.727)
                    )
                    labelπ_used = true
                end
            end

        end
    end

    Plots.savefig(plt, "TOS_$(N)_kagome_offchain.pdf")
end


# identify the ground state and find the ground state correlations 
function realspacecorrelation(L::Int64)

    local N::Int64 = 2*L

    @load "kagomedata16/gs_4_0_kagome.jld2" groundstate 

    # we find the on-chain ZZ correlations 

    local Zcrdata = zeros(L)
    local Xcrdata = zeros(L)

    for r = 1:L 
        for i in 1:2:(N-1)

            j = (i + 2*r) % N

            Zcrdata[r] += EDcalc.Szcorrelation(L, i, j, groundstate, 4, 0)
            Xcrdata[r] += EDcalc.Sxcorrelation(L, i, j, groundstate, 4, 0)
        end

        Zcrdata[r] /= L 
        Xcrdata[r] /= L 
    end

    df = DataFrames.DataFrame(ZZ = Zcrdata, XX = Xcrdata)
    CSV.write("gscorr.csv",df)
end



# findenergies(L, 0.02, 0.0)
# towerofstates(L)
# println(realspacecorrelation(8))
