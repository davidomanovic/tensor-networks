# DMRG on H2/cc-pVDZ (10 orbitals), get 1-RDM and natural orbitals.
using ITensors, ITensorMPS, NPZ, LinearAlgebra, Printf

const NPZ_FILE = "data/h2_ccpvdz_R0p74.npz"

function build_mpo(h1::Array{Float64,2}, eri_chem::Array{Float64,4}, sites)
    norb = size(h1,1)
    os = OpSum()
    for p=1:norb, q=1:norb
        h = h1[p,q]; if abs(h) > 1e-12
            os += h, "Cdagup",p,"Cup",q
            os += h, "Cdagdn",p,"Cdn",q
        end
    end
    for p=1:norb, q=1:norb, r=1:norb, s=1:norb
        V = eri_chem[p,q,r,s]; if abs(V) > 1e-12
            os += 0.5V, "Cdagup",p,"Cdagup",q, "Cup",s,"Cup",r
            os += 0.5V, "Cdagdn",p,"Cdagdn",q, "Cdn",s,"Cdn",r
            os += 0.5V, "Cdagup",p,"Cdagdn",q, "Cdn",s,"Cup",r
            os += 0.5V, "Cdagdn",p,"Cdagup",q, "Cup",s,"Cdn",r
        end
    end
    return ITensorMPS.MPO(os, sites)
end

function one_rdm_spin_summed(psi::MPS, sites, norb)
    gamma = zeros(norb, norb)
    for p=1:norb, q=1:norb
        os = OpSum()
        os += 1.0,"Cdagup",p,"Cup",q
        os += 1.0,"Cdagdn",p,"Cdn",q
        gamma[p,q] = ITensorMPS.inner(psi, ITensorMPS.MPO(os, sites), psi)
    end
    return (gamma + gamma')/2
end

function main()
    @info "Loading $(NPZ_FILE)"
    d = npzread(NPZ_FILE)
    h1  = Array{Float64,2}(d["h1"])
    eri = Array{Float64,4}(d["eri"])
    norb = size(h1,1)
    @info "norb = $norb"

    sites = ITensorMPS.siteinds("Electron", norb; conserve_qns=true)

    H = build_mpo(h1, eri, sites)

    psi0 = ITensorMPS.productMPS(sites, j -> j==1 ? "UpDn" : "Emp") # HF: doubly occupy the lowest MO (site 1), others empty


    sweeps = ITensorMPS.Sweeps(6)
    ITensorMPS.maxdim!(sweeps, 200, 400, 800, 1200, 1600, 2000)
    ITensorMPS.mindim!(sweeps, 10, 20, 20, 20, 20, 20)
    ITensorMPS.cutoff!(sweeps, 1e-10)
    ITensorMPS.noise!(sweeps, 1e-7, 1e-8, 1e-10, 0, 1e-11, 0)

    @info "Running DMRG…"
    E, psi = ITensorMPS.dmrg(H, psi0, sweeps; outputlevel=0)
    @info @sprintf("DMRG (electronic) = %.12f Ha", E)

    gamma = one_rdm_spin_summed(psi, sites, norb)
    vals, vecs = eigen(gamma); perm = sortperm(vals; rev=true)
    no_occ = vals[perm]; U_no = vecs[:,perm]
    @info "Top NO occ = $(no_occ[1:min(end,8)])"
    npzwrite("data/h2_ccpvdz_no_from_dmrg.npz",
             Dict("gamma"=>gamma, "no_occ"=>no_occ, "U_no"=>U_no, "E_dmrg"=>E))
    @info "Wrote data/h2_ccpvdz_no_from_dmrg.npz"
    @info @sprintf("Tr(gamma) = %.6f (should be ≈ 2)", sum(no_occ))
end

main()
