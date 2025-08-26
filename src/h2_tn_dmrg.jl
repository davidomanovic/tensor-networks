using ITensors, ITensorMPS, NPZ, LinearAlgebra, Printf

const NPZ_FILE = "data/h2_sto3g_R0p74.npz"

function main()
    @info "Loading integrals from $(NPZ_FILE)"
    data = npzread(NPZ_FILE)
    h1  = Array{Float64,2}(data["h1"])
    eri = Array{Float64,4}(data["eri"])
    eri = permutedims(eri, (1,3,2,4)) #physicist to chemist order

    norb = size(h1,1)
    sites = ITensorMPS.siteinds("Electron", norb; conserve_qns=true)

    # Build Hamiltonian
    os = OpSum()
    for p=1:norb, q=1:norb
        h = h1[p,q]
        if abs(h) > 1e-12
            os += h, "Cdagup", p, "Cup", q
            os += h, "Cdagdn", p, "Cdn", q
        end
    end
    for p=1:norb, q=1:norb, r=1:norb, s=1:norb
        V = eri[p,q,r,s]
        if abs(V) > 1e-12
            os += 0.5*V, "Cdagup", p, "Cdagup", q, "Cup", s, "Cup", r   # same spin
            os += 0.5*V, "Cdagdn", p, "Cdagdn", q, "Cdn", s, "Cdn", r
            os += 0.5*V, "Cdagup", p, "Cdagdn", q, "Cdn", s, "Cup", r   # opposite spin
            os += 0.5*V, "Cdagdn", p, "Cdagup", q, "Cup", s, "Cdn", r
        end
    end
    H = ITensorMPS.MPO(os, sites)

    # Closed-shell HF determinant: double occupancy of the lowest MO
    psiHF = ITensorMPS.productMPS(sites, j -> j==1 ? "UpDn" : "Emp")

    sweeps = ITensorMPS.Sweeps(6)
    ITensorMPS.maxdim!(sweeps, 50, 100, 200, 400, 800, 1000)
    ITensorMPS.mindim!(sweeps, 10, 20, 20, 20, 20, 20)
    ITensorMPS.cutoff!(sweeps, 1e-12)
    ITensorMPS.noise!(sweeps, 1e-7, 1e-8, 1e-10, 0, 1e-11, 0)

    @info "Running DMRG..."
    E, psi = ITensorMPS.dmrg(H, psiHF, sweeps; outputlevel=0)
    @info @sprintf("DMRG (electronic) = %.12f Ha", E) 

    # 1-RDM and NOs
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
    gamma = one_rdm_spin_summed(psi, sites, norb)
    vals, vecs = eigen(gamma); perm = sortperm(vals; rev=true)
    no_occ = vals[perm]; U_no = vecs[:,perm]
    npzwrite("data/h2_no_from_dmrg.npz", Dict("gamma"=>gamma, "no_occ"=>no_occ, "U_no"=>U_no, "E_dmrg"=>E))
end

main()
