# TN -> QC helper: DMRG on ab initio H, 1-RDM & Natural Orbitals
using ITensors
using ITensorMPS
using NPZ
using LinearAlgebra
using Printf

const NPZ_FILE = "data/h6_sto3g_R1p2.npz"  

function main()
    @info "Loading integrals from $(NPZ_FILE)"
    data = npzread(NPZ_FILE)
    h1  = Array{Float64,2}(data["h1"])
    eri = Array{Float64,4}(data["eri"])

    norb = size(h1, 1)
    @info "norb = $norb"

    # Build Electron sites
    sites = ITensorMPS.siteinds("Electron", norb; conserve_qns=true)

    # Build ab initio Hamiltonian
    # Using OpSum (modern API). Add terms separately per spin channel.
    @info "Building Hamiltonian OpSum..."
    os = OpSum()

    # one-electron terms
    for p = 1:norb, q = 1:norb
        h = h1[p, q]
        if abs(h) > 1e-12
            os += h, "Cdagup", p, "Cup", q
            os += h, "Cdagdn", p, "Cdn", q
        end
    end

    # two-electron terms (pq|rs)
    @info "Adding two-electron terms..."
    for p = 1:norb, q = 1:norb, r = 1:norb, s = 1:norb
        V = eri[p, q, r, s]
        if abs(V) > 1e-12
            # Same-spin contributions (0.5 factor)
            os += 0.5 * V, "Cdagup", p, "Cdagup", r, "Cup", s, "Cup", q
            os += 0.5 * V, "Cdagdn", p, "Cdagdn", r, "Cdn", s, "Cdn", q
            # Opposite-spin contributions
            os += V, "Cdagup", p, "Cdagdn", r, "Cdn", s, "Cup", q
            os += V, "Cdagdn", p, "Cdagup", r, "Cup", s, "Cdn", q
        end
    end

    H = ITensorMPS.MPO(os, sites)
    @info "Hamiltonian MPO built."

    # Initial state: half-filling product
    init_state = ITensorMPS.productMPS(sites, j -> isodd(j) ? "Up" : "Dn")
    # DMRG setup
    sweeps = ITensorMPS.Sweeps(6)
    ITensorMPS.maxdim!(sweeps, 200, 400, 800, 1200, 1600, 2000)
    ITensorMPS.cutoff!(sweeps, 1e-9)
    ITensorMPS.noise!(sweeps, 1e-6, 1e-6, 1e-7, 0.0, 0.0, 0.0)

    @info "Starting DMRG..."
    E, psi = ITensorMPS.dmrg(H, init_state, sweeps; outputlevel=0)
    @info @sprintf("DMRG finished. Energy = %.12f Ha", E)

    # Spin-summed 1-RDM
    function one_rdm_spin_summed(psi::MPS, sites, norb::Int)
        gamma = zeros(Float64, norb, norb)
        for p = 1:norb, q = 1:norb
            os = OpSum()
            os += 1.0, "Cdagup", p, "Cup", q
            os += 1.0, "Cdagdn", p, "Cdn", q
            O = ITensorMPS.MPO(os, sites)
            gamma[p, q] = ITensorMPS.inner(psi, O, psi)
        end
        return gamma
    end

    @info "Computing spin-summed 1-RDM..."
    gamma = one_rdm_spin_summed(psi, sites, norb)

    # Ensure Hermitian
    gamma = (gamma + gamma') / 2

    # Natural orbitals: eigen-decomposition of gamma
    vals, vecs = eigen(gamma) 

    # Sort by occupation descending
    perm = sortperm(vals; rev=true)
    no_occ = vals[perm]
    U_no  = vecs[:, perm]  # columns are NOs

    @info "Top natural occupations:"
    for i in 1:min(norb, 8)
        @info @sprintf("  NO %2d: occ = %.8f", i, no_occ[i])
    end

    # Save outputs for Qiskit step
    out_npz = "data/tn_no_from_dmrg.npz"
    npzwrite(out_npz, Dict("gamma" => gamma, "no_occ" => no_occ, "U_no" => U_no, "E_dmrg" => E))
    @info "Wrote TN outputs to $out_npz"

    # quick sanity checks
    N_e = sum(no_occ)           # should be close to total electrons
    @info @sprintf("Tr(gamma) ~ total electrons = %.8f", N_e)
end

main()