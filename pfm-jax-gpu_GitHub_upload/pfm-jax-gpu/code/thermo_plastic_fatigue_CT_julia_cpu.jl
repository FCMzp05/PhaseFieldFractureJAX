# ============================================================================
# Standalone canonical ΔT=200 K thermo-elasto-plastic CT fatigue benchmark.
#
# Generates the h=l0/2 Gmsh mesh internally and requires only registered Julia
# packages (Gridap, GridapGmsh, Gmsh). It writes cycle data, timing, and
# default-on milestone/final VTK snapshots without importing project files.
#
# Run:
#   julia --project=. code/thermo_plastic_fatigue_CT_julia_cpu.jl
# Quick coupled check:
#   julia --project=. code/thermo_plastic_fatigue_CT_julia_cpu.jl --smoke
# ============================================================================

using Gmsh: gmsh
using Gridap
using Gridap.Geometry
using Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using Gridap.TensorValues
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Algebra
using GridapGmsh
using LinearAlgebra
using Statistics
using Printf

parse_env_int(key::String, default::Int) = parse(Int, get(ENV, key, string(default)))
parse_env_float(key::String, default::Float64) = parse(Float64, get(ENV, key, string(default)))

function script_root()
    d = abspath(@__DIR__)
    return basename(d) == "code" ? dirname(d) : d
end

function parse_cli(args)
    output_dir = joinpath(script_root(), "outputs", "results_cpu",
                          "thermo_plastic_fatigue_CT", "julia")
    visualization = true
    snapshots = true
    smoke = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--output-dir"
            i == length(args) && error("--output-dir requires a path")
            i += 1
            output_dir = args[i]
        elseif arg == "--no-visualization"
            visualization = false
        elseif arg == "--no-snapshots"
            snapshots = false
        elseif arg == "--smoke"
            smoke = true
        else
            error("unknown argument: $arg")
        end
        i += 1
    end
    return (output_dir=abspath(output_dir), visualization=visualization,
            snapshots=snapshots, smoke=smoke)
end
const CLI = parse_cli(ARGS)

# ============================================================================
# MESH-CONVERGENCE CONFIGURATION  (edit here)
# ============================================================================
const l0          = 0.3          # phase-field length scale (FIXED across the study)
const CHI_EFF     = 0.85         # Taylor--Quinney coefficient (fixed)
const T_RIGHT_MAX = 500.0
const H_FINE = l0 / 2.0

# ============================================================================
# CT geometry (ASTM E647, mm)
# ============================================================================
const W_CT   = 50.0
const H_CT   = 48.0
const Y_MID  = 24.0

const PIN_RADIUS = 5.0
const PIN1_X     = 8.0
const PIN1_Y     = 12.0
const PIN2_X     = 8.0
const PIN2_Y     = 36.0

const NOTCH_LEN   = 12.0
const NOTCH_HALF  = 1.0

const PRECRACK_LEN       = 2.0
const PRECRACK_TRANS_LEN = 0.8
const PRECRACK_HALF      = 0.12
const PRECRACK_RECT_LEN  = PRECRACK_LEN - PRECRACK_TRANS_LEN
PRECRACK_RECT_LEN > 0.0 || error("PRECRACK_TRANS_LEN must be smaller than PRECRACK_LEN")

const a0_total     = NOTCH_LEN + PRECRACK_LEN
const CRACK_TIP_X0 = a0_total

# ============================================================================
# Mechanical material (Chaboche)
# ============================================================================
struct MaterialData
    E  ::Float64
    ν  ::Float64
    G  ::Float64
    λ  ::Float64
    σy0::Float64
    Q∞ ::Float64
    b  ::Float64
    C  ::Float64
    γ  ::Float64
end

function create_material()
    E   = 210000.0
    ν   = 0.3
    G   = E / (2.0 * (1.0 + ν))
    λ   = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))
    σy0 = 440.0
    Q∞  = 55.0
    b   = 2.38
    C   = 23_554.0
    γ   = 139.0
    return MaterialData(E, ν, G, λ, σy0, Q∞, b, C, γ)
end

const mat = create_material()

# ============================================================================
# Thermal material
# ============================================================================
const k0_heat = 0.05
const k1_heat = 1e-6 * k0_heat
const Cv_vol  = 3.925e-3
const α_th    = 1.2e-5
const T0      = 300.0

# ============================================================================
# Cycle control
# ============================================================================
const N_HEAT_CYCLES     = 30
const N_CYCLES          = 2000
const N_STEPS_PER_CYCLE = 20
const phase_every       = 1
const max_stagger       = 20
const stagger_tol       = 1e-6

const Δt_mech_cycle = 1.0

# ============================================================================
# Mechanical loading (pin-hole force control)
# ============================================================================
const THICKNESS  = 25.0
const F_MAX_3D   = 20000.0
const PIN_CIRC   = 2.0 * π * PIN_RADIUS
const F_MAX      = F_MAX_3D / THICKNESS
const R_LOAD     = 0.0
const F_MIN_3D   = R_LOAD * F_MAX_3D
const F_MIN      = F_MIN_3D / THICKNESS
const LOAD_SUBSTEP_SIZE = 50.0

@inline pin_traction(force_per_thickness::Float64) = force_per_thickness / PIN_CIRC

# ============================================================================
# Phase-field and thermal softening
# ============================================================================
const Gc0             = 2.7
const αT_fat0         = 20.0
const η_stab          = 1e-3

const Tref_soft = 300.0
const Tmax_soft = 1000.0
const b1_soft   = 1.80
const b2_soft   = 1.10

function _temp_softening(T::Float64, val0::Float64)
    ξ = clamp((T - Tref_soft) / Tmax_soft, 0.0, 1.0)
    return val0 * max(1.0 - b1_soft * ξ + b2_soft * ξ^2, 0.05)
end

Gc_T(T::Float64)     = _temp_softening(T, Gc0)
αT_fat_T(T::Float64) = _temp_softening(T, αT_fat0)

@inline function fatigue_degradation_T(ᾱ::Real, T_loc::Float64)
    αT = αT_fat_T(T_loc)
    return (αT / (ᾱ + αT + 1e-12))^2
end

@inline function phase_field_degradation(s::Real)
    s̄ = clamp(s, 0.0, 1.0)
    return (1.0 - s̄)^2 + η_stab
end

# ============================================================================
# Output control (CSV only)
# ============================================================================
const PRINT_CYCLE_EVERY = 20
const FORCE_GC_EVERY    = 20
const RESULT_ROOT       = CLI.output_dir
const SNAPSHOT_FRACTIONS = (0.10, 0.30, 0.60, 0.90)

# ============================================================================
# Coarse mesh sizes (fixed); fine size is the swept quantity
# ============================================================================
const h_coarse = 2.0
const h_pin    = 0.8

# ============================================================================
# Elastic stiffness
# ============================================================================
function elastic_stiffness()
    C1111 = mat.E * (1.0 - mat.ν) / ((1.0 + mat.ν) * (1.0 - 2.0 * mat.ν))
    C1122 = mat.E * mat.ν         / ((1.0 + mat.ν) * (1.0 - 2.0 * mat.ν))
    C1212 = mat.E / (2.0 * (1.0 + mat.ν))
    return SymFourthOrderTensorValue(
        C1111, 0.0,   C1122,
        0.0,   C1212, 0.0,
        C1122, 0.0,   C1111
    )
end

const C_tensor = elastic_stiffness()
const ε_sym = symmetric_gradient

# ============================================================================
# Plastic state
# ============================================================================
mutable struct PlasticStateData
    εp   ::SymTensorValue{2,Float64,3}
    εp33 ::Float64
    εp_eq::Float64
    α    ::SymTensorValue{2,Float64,3}
    α33  ::Float64
end

PlasticStateData() = PlasticStateData(
    SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0, 0.0,
    SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0)

function copy_plastic_state(ps::PlasticStateData)
    return PlasticStateData(ps.εp, ps.εp33, ps.εp_eq, ps.α, ps.α33)
end

# ============================================================================
# Chaboche combined hardening
# ============================================================================
yield_stress(εp_eq::Float64) = mat.σy0 + mat.Q∞ * (1.0 - exp(-mat.b * εp_eq))
dσY_dεp(εp_eq::Float64)      = mat.Q∞ * mat.b * exp(-mat.b * εp_eq)

function plastic_strain_energy(εp_eq::Float64)
    abs(εp_eq) < 1e-12 && return 0.0
    return mat.σy0 * εp_eq + mat.Q∞ * (εp_eq + (exp(-mat.b * εp_eq) - 1.0) / mat.b)
end

@inline function svolos_damage_measure(c::Real)
    c̄ = clamp(c, 0.0, 1.0)
    return (c̄ * (2.0 - c̄))^(3 / 2)
end

@inline function thermal_conductivity_degradation_svolos(c::Real)
    d̄ = svolos_damage_measure(c)
    mT = (1.0 - d̄) / (1.0 + 0.5 * d̄)
    return clamp(mT, 0.0, 1.0)
end

@inline function plastic_heat_source_density(εp_eq_old::Float64, εp_eq_new::Float64,
                                             dt::Float64, χ_eff::Float64)
    dt <= 0.0 && return 0.0
    Δεp_eq = max(εp_eq_new - εp_eq_old, 0.0)
    Δεp_eq < 1e-14 && return 0.0
    σflow_bar = 0.5 * (yield_stress(εp_eq_old) + yield_stress(εp_eq_new))
    return χ_eff * σflow_bar * Δεp_eq / dt
end

# ============================================================================
# Thermo-elasto-plastic stress and energy
# ============================================================================
@inline function elastic_strain_components_thermo(ε_total::SymTensorValue{2,Float64,3},
                                                  plastic::PlasticStateData, T_local::Float64)
    εth  = α_th * (T_local - T0)
    εe11 = ε_total[1,1] - plastic.εp[1,1] - εth
    εe22 = ε_total[2,2] - plastic.εp[2,2] - εth
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33 - εth
    return εe11, εe22, εe33, εe12
end

@inline function elastic_stress_thermo_ep(ε_total::SymTensorValue{2,Float64,3},
                                          plastic::PlasticStateData, T_local::Float64)
    εe11, εe22, εe33, εe12 = elastic_strain_components_thermo(ε_total, plastic, T_local)
    trεe = εe11 + εe22 + εe33
    σ11  = 2.0 * mat.G * εe11 + mat.λ * trεe
    σ22  = 2.0 * mat.G * εe22 + mat.λ * trεe
    σ33  = 2.0 * mat.G * εe33 + mat.λ * trεe
    σ12  = 2.0 * mat.G * εe12
    return σ11, σ22, σ33, σ12, εe11, εe22, εe33, εe12
end

@inline function von_mises_eq_3d(σ11, σ22, σ33, σ12)
    p   = (σ11 + σ22 + σ33) / 3.0
    s11 = σ11 - p; s22 = σ22 - p; s33 = σ33 - p; s12 = σ12
    σeq = sqrt(max(0.0, 1.5 * (s11^2 + s22^2 + s33^2 + 2.0 * s12^2)))
    return σeq, s11, s22, s33, s12
end

@inline macaulay(x::Float64) = x > 0.0 ? x : 0.0

function mode_I_elastic_energy_thermo(ε_total::SymTensorValue{2,Float64,3},
                                      plastic::PlasticStateData, T_local::Float64)
    εe11, εe22, εe33, εe12 = elastic_strain_components_thermo(ε_total, plastic, T_local)
    avg  = 0.5 * (εe11 + εe22)
    diff = 0.5 * (εe11 - εe22)
    rad  = sqrt(diff^2 + εe12^2)
    ε1   = avg + rad; ε2 = avg - rad; ε3 = εe33
    tr_pos = macaulay(ε1) + macaulay(ε2) + macaulay(ε3)
    return 0.5 * mat.λ * tr_pos^2
end

function mode_II_elastic_energy_thermo(ε_total::SymTensorValue{2,Float64,3},
                                       plastic::PlasticStateData, T_local::Float64)
    εe11, εe22, εe33, εe12 = elastic_strain_components_thermo(ε_total, plastic, T_local)
    avg  = 0.5 * (εe11 + εe22)
    diff = 0.5 * (εe11 - εe22)
    rad  = sqrt(diff^2 + εe12^2)
    ε1   = avg + rad; ε2 = avg - rad; ε3 = εe33
    ε1p  = macaulay(ε1); ε2p = macaulay(ε2); ε3p = macaulay(ε3)
    return mat.G * (ε1p^2 + ε2p^2 + ε3p^2)
end

# ============================================================================
# Radial return mapping
# ============================================================================
function return_mapping_thermo!(ε_total::SymTensorValue{2,Float64,3},
                                plastic::PlasticStateData, s_pf::Float64,
                                T_local::Float64; update_state::Bool = true)
    σ11_tr, σ22_tr, σ33_tr, σ12_tr, _, _, _, _ =
        elastic_stress_thermo_ep(ε_total, plastic, T_local)
    _, s11, s22, s33, s12 = von_mises_eq_3d(σ11_tr, σ22_tr, σ33_tr, σ12_tr)

    g_s = phase_field_degradation(s_pf)

    α11_n = plastic.α[1,1]; α22_n = plastic.α[2,2]; α12_n = plastic.α[1,2]; α33_n = plastic.α33

    ξ11 = s11 - α11_n; ξ22 = s22 - α22_n; ξ12 = s12 - α12_n; ξ33 = s33 - α33_n
    ξ_eq = sqrt(max(0.0, 1.5 * (ξ11^2 + ξ22^2 + ξ33^2 + 2.0 * ξ12^2)))

    σy = yield_stress(plastic.εp_eq)
    if g_s * ξ_eq - σy <= 0.0 || ξ_eq < 1e-12
        return g_s * SymTensorValue{2,Float64}(σ11_tr, σ12_tr, σ22_tr)
    end

    inv_ξ = 1.0 / ξ_eq
    n11 = 1.5 * ξ11 * inv_ξ; n22 = 1.5 * ξ22 * inv_ξ
    n33 = 1.5 * ξ33 * inv_ξ; n12 = 1.5 * ξ12 * inv_ξ

    f_trial   = g_s * ξ_eq - σy
    denom0    = 3.0 * g_s * mat.G + mat.C + dσY_dεp(plastic.εp_eq)
    Δγ        = max(f_trial / max(denom0, 1e-12), 0.0)
    kin_coeff = (2.0 / 3.0) * mat.C
    tol       = 1e-10 * (ξ_eq + mat.σy0)
    converged = false

    function eval_consistency(Δγ_loc::Float64)
        denom_γ = 1.0 + mat.γ * Δγ_loc
        α11_loc = (α11_n + kin_coeff * n11 * Δγ_loc) / denom_γ
        α22_loc = (α22_n + kin_coeff * n22 * Δγ_loc) / denom_γ
        α33_loc = (α33_n + kin_coeff * n33 * Δγ_loc) / denom_γ
        α12_loc = (α12_n + kin_coeff * n12 * Δγ_loc) / denom_γ
        s11_loc = s11 - 2.0 * mat.G * Δγ_loc * n11
        s22_loc = s22 - 2.0 * mat.G * Δγ_loc * n22
        s33_loc = s33 - 2.0 * mat.G * Δγ_loc * n33
        s12_loc = s12 - 2.0 * mat.G * Δγ_loc * n12
        ξ11_loc = s11_loc - α11_loc; ξ22_loc = s22_loc - α22_loc
        ξ33_loc = s33_loc - α33_loc; ξ12_loc = s12_loc - α12_loc
        ξ_eq_loc = sqrt(max(0.0, 1.5 * (ξ11_loc^2 + ξ22_loc^2 + ξ33_loc^2 + 2.0 * ξ12_loc^2)))
        if ξ_eq_loc < 1e-18
            return -yield_stress(plastic.εp_eq + Δγ_loc), 0.0
        end
        dα11_dΔ = (kin_coeff * n11 - mat.γ * α11_n) / denom_γ^2
        dα22_dΔ = (kin_coeff * n22 - mat.γ * α22_n) / denom_γ^2
        dα33_dΔ = (kin_coeff * n33 - mat.γ * α33_n) / denom_γ^2
        dα12_dΔ = (kin_coeff * n12 - mat.γ * α12_n) / denom_γ^2
        dξ11_dΔ = -2.0 * mat.G * n11 - dα11_dΔ
        dξ22_dΔ = -2.0 * mat.G * n22 - dα22_dΔ
        dξ33_dΔ = -2.0 * mat.G * n33 - dα33_dΔ
        dξ12_dΔ = -2.0 * mat.G * n12 - dα12_dΔ
        ξ_dξ    = ξ11_loc * dξ11_dΔ + ξ22_loc * dξ22_dΔ +
                  ξ33_loc * dξ33_dΔ + 2.0 * ξ12_loc * dξ12_dΔ
        dξeq_dΔ = 1.5 * ξ_dξ / ξ_eq_loc
        gval = g_s * ξ_eq_loc - yield_stress(plastic.εp_eq + Δγ_loc)
        dgdΔ = g_s * dξeq_dΔ - dσY_dεp(plastic.εp_eq + Δγ_loc)
        return gval, dgdΔ
    end

    Δγ_lo = 0.0
    Δγ_hi = max(2.0 * Δγ, f_trial / max(denom0, 1e-12), 1e-12)
    g_hi, _ = eval_consistency(Δγ_hi)
    for _ in 1:30
        g_hi <= 0.0 && break
        Δγ_hi *= 2.0
        g_hi, _ = eval_consistency(Δγ_hi)
    end
    has_bracket = g_hi <= 0.0

    for _ in 1:200
        gval, dgdΔ = eval_consistency(Δγ)
        if abs(gval) < tol
            converged = true
            break
        end
        if has_bracket && gval > 0.0
            Δγ_lo = Δγ
        elseif has_bracket
            Δγ_hi = Δγ
        end
        if has_bracket
            Δγ_new = abs(dgdΔ) < 1e-30 ? 0.5 * (Δγ_lo + Δγ_hi) : (Δγ - gval / dgdΔ)
            if !(Δγ_lo < Δγ_new < Δγ_hi)
                Δγ_new = 0.5 * (Δγ_lo + Δγ_hi)
            end
        else
            Δγ_new = abs(dgdΔ) < 1e-30 ? (1.5 * Δγ) : (Δγ - gval / dgdΔ)
            Δγ_new <= 0.0 && (Δγ_new = max(0.5 * Δγ, 1e-14))
        end
        Δγ = Δγ_new
    end

    converged || @warn "Return mapping did not converge" ξ_eq Δγ

    denom_γ   = 1.0 + mat.γ * Δγ
    εp_eq_new = plastic.εp_eq + Δγ
    Δεp11 = Δγ * n11; Δεp22 = Δγ * n22; Δεp33 = Δγ * n33; Δεp12 = Δγ * n12
    α11_new = (α11_n + kin_coeff * n11 * Δγ) / denom_γ
    α22_new = (α22_n + kin_coeff * n22 * Δγ) / denom_γ
    α33_new = (α33_n + kin_coeff * n33 * Δγ) / denom_γ
    α12_new = (α12_n + kin_coeff * n12 * Δγ) / denom_γ

    if update_state
        plastic.εp    += SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22)
        plastic.εp33  += Δεp33
        plastic.εp_eq  = εp_eq_new
        plastic.α      = SymTensorValue{2,Float64}(α11_new, α12_new, α22_new)
        plastic.α33    = α33_new
        σ11, σ22, _, σ12, _, _, _, _ = elastic_stress_thermo_ep(ε_total, plastic, T_local)
    else
        plastic_tmp = PlasticStateData(
            plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22),
            plastic.εp33 + Δεp33, εp_eq_new,
            SymTensorValue{2,Float64}(α11_new, α12_new, α22_new), α33_new)
        σ11, σ22, _, σ12, _, _, _, _ = elastic_stress_thermo_ep(ε_total, plastic_tmp, T_local)
    end
    return g_s * SymTensorValue{2,Float64}(σ11, σ12, σ22)
end

# ============================================================================
# CT mesh (h_fine is the swept argument)
# ============================================================================
function create_CT_mesh(h_fine::Float64; output_dir::String, tag::String)
    W  = W_CT; H  = H_CT; pin_r = PIN_RADIUS
    p1x, p1y = PIN1_X, PIN1_Y; p2x, p2y = PIN2_X, PIN2_Y
    tip_x = a0_total

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CT_ThermoFatigue_$(tag)")

    pt = 1
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_coarse, pt); pt_LB = pt; pt += 1
    gmsh.model.geo.addPoint(W,   0.0, 0.0, h_coarse, pt); pt_RB = pt; pt += 1
    gmsh.model.geo.addPoint(W,   H,   0.0, h_coarse, pt); pt_RT = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, H,   0.0, h_coarse, pt); pt_LT = pt; pt += 1

    gmsh.model.geo.addPoint(0.0,      Y_MID + NOTCH_HALF, 0.0, h_fine, pt); pt_notch_UL = pt; pt += 1
    gmsh.model.geo.addPoint(NOTCH_LEN, Y_MID + NOTCH_HALF, 0.0, h_fine, pt); pt_notch_UR = pt; pt += 1
    gmsh.model.geo.addPoint(NOTCH_LEN, Y_MID - NOTCH_HALF, 0.0, h_fine, pt); pt_notch_LR = pt; pt += 1
    gmsh.model.geo.addPoint(0.0,      Y_MID - NOTCH_HALF, 0.0, h_fine, pt); pt_notch_LL = pt; pt += 1

    crack_start_x = tip_x - PRECRACK_RECT_LEN
    gmsh.model.geo.addPoint(crack_start_x, Y_MID + PRECRACK_HALF, 0.0, h_fine, pt); pt_crack_UL = pt; pt += 1
    gmsh.model.geo.addPoint(tip_x,         Y_MID + PRECRACK_HALF, 0.0, h_fine, pt); pt_crack_UR = pt; pt += 1
    gmsh.model.geo.addPoint(tip_x,         Y_MID - PRECRACK_HALF, 0.0, h_fine, pt); pt_crack_LR = pt; pt += 1
    gmsh.model.geo.addPoint(crack_start_x, Y_MID - PRECRACK_HALF, 0.0, h_fine, pt); pt_crack_LL = pt; pt += 1

    gmsh.model.geo.addPoint(p1x,       p1y,       0.0, h_coarse, pt); pt_pin1_C = pt; pt += 1
    gmsh.model.geo.addPoint(p1x+pin_r, p1y,       0.0, h_coarse, pt); pt_pin1_E = pt; pt += 1
    gmsh.model.geo.addPoint(p1x,       p1y+pin_r, 0.0, h_coarse, pt); pt_pin1_N = pt; pt += 1
    gmsh.model.geo.addPoint(p1x-pin_r, p1y,       0.0, h_coarse, pt); pt_pin1_W = pt; pt += 1
    gmsh.model.geo.addPoint(p1x,       p1y-pin_r, 0.0, h_coarse, pt); pt_pin1_S = pt; pt += 1

    gmsh.model.geo.addPoint(p2x,       p2y,       0.0, h_coarse, pt); pt_pin2_C = pt; pt += 1
    gmsh.model.geo.addPoint(p2x+pin_r, p2y,       0.0, h_coarse, pt); pt_pin2_E = pt; pt += 1
    gmsh.model.geo.addPoint(p2x,       p2y+pin_r, 0.0, h_coarse, pt); pt_pin2_N = pt; pt += 1
    gmsh.model.geo.addPoint(p2x-pin_r, p2y,       0.0, h_coarse, pt); pt_pin2_W = pt; pt += 1
    gmsh.model.geo.addPoint(p2x,       p2y-pin_r, 0.0, h_coarse, pt); pt_pin2_S = pt; pt += 1

    ln = 1
    gmsh.model.geo.addLine(pt_LB, pt_RB, ln); line_bottom = ln; ln += 1
    gmsh.model.geo.addLine(pt_RB, pt_RT, ln); line_right = ln; ln += 1
    gmsh.model.geo.addLine(pt_RT, pt_LT, ln); line_top = ln; ln += 1
    gmsh.model.geo.addLine(pt_LT, pt_notch_UL, ln); line_left_upper = ln; ln += 1
    gmsh.model.geo.addLine(pt_notch_LL, pt_LB, ln); line_left_lower = ln; ln += 1

    gmsh.model.geo.addLine(pt_notch_UL, pt_notch_UR, ln); line_notch_top = ln; ln += 1
    gmsh.model.geo.addLine(pt_notch_UR, pt_crack_UL, ln); line_notch_to_crack_U = ln; ln += 1
    gmsh.model.geo.addLine(pt_crack_LL, pt_notch_LR, ln); line_crack_to_notch_L = ln; ln += 1
    gmsh.model.geo.addLine(pt_notch_LR, pt_notch_LL, ln); line_notch_bottom = ln; ln += 1

    gmsh.model.geo.addLine(pt_crack_UL, pt_crack_UR, ln); line_crack_top = ln; ln += 1
    gmsh.model.geo.addLine(pt_crack_UR, pt_crack_LR, ln); line_crack_tip = ln; ln += 1
    gmsh.model.geo.addLine(pt_crack_LR, pt_crack_LL, ln); line_crack_bottom = ln; ln += 1

    gmsh.model.geo.addCircleArc(pt_pin1_E, pt_pin1_C, pt_pin1_N, ln); line_p1_1 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin1_N, pt_pin1_C, pt_pin1_W, ln); line_p1_2 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin1_W, pt_pin1_C, pt_pin1_S, ln); line_p1_3 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin1_S, pt_pin1_C, pt_pin1_E, ln); line_p1_4 = ln; ln += 1
    lines_pin1 = [line_p1_1, line_p1_2, line_p1_3, line_p1_4]

    gmsh.model.geo.addCircleArc(pt_pin2_E, pt_pin2_C, pt_pin2_N, ln); line_p2_1 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin2_N, pt_pin2_C, pt_pin2_W, ln); line_p2_2 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin2_W, pt_pin2_C, pt_pin2_S, ln); line_p2_3 = ln; ln += 1
    gmsh.model.geo.addCircleArc(pt_pin2_S, pt_pin2_C, pt_pin2_E, ln); line_p2_4 = ln; ln += 1
    lines_pin2 = [line_p2_1, line_p2_2, line_p2_3, line_p2_4]

    outer_loop = [line_bottom, line_right, line_top, line_left_upper,
                  line_notch_top, line_notch_to_crack_U, line_crack_top, line_crack_tip,
                  line_crack_bottom, line_crack_to_notch_L, line_notch_bottom, line_left_lower]

    gmsh.model.geo.addCurveLoop(outer_loop, 1)
    gmsh.model.geo.addCurveLoop(lines_pin1, 2)
    gmsh.model.geo.addCurveLoop(lines_pin2, 3)
    gmsh.model.geo.addPlaneSurface([1, 2, 3], 1)

    gmsh.model.addPhysicalGroup(2, [1], 1); gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, lines_pin2, 1); gmsh.model.setPhysicalName(1, 1, "pin_upper")
    gmsh.model.addPhysicalGroup(1, lines_pin1, 2); gmsh.model.setPhysicalName(1, 2, "pin_lower")
    gmsh.model.addPhysicalGroup(0, [pt_RB], 3); gmsh.model.setPhysicalName(0, 3, "fixed_point")
    gmsh.model.addPhysicalGroup(0, [pt_LB], 8); gmsh.model.setPhysicalName(0, 8, "rot_lock_y")
    gmsh.model.addPhysicalGroup(1, [line_left_upper, line_left_lower], 4); gmsh.model.setPhysicalName(1, 4, "left")
    gmsh.model.addPhysicalGroup(1, [line_right], 5); gmsh.model.setPhysicalName(1, 5, "right")
    gmsh.model.addPhysicalGroup(1, [line_top], 6); gmsh.model.setPhysicalName(1, 6, "top")
    gmsh.model.addPhysicalGroup(1, [line_bottom], 7); gmsh.model.setPhysicalName(1, 7, "bottom_edge")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn",  h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", a0_total - 2.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", W_CT)
    gmsh.model.mesh.field.setNumber(1, "YMin", Y_MID - 1.25)
    gmsh.model.mesh.field.setNumber(1, "YMax", Y_MID + 1.25)

    gmsh.model.mesh.field.add("Ball", 2)
    gmsh.model.mesh.field.setNumber(2, "Radius", 7.5)
    gmsh.model.mesh.field.setNumber(2, "VIn", h_pin)
    gmsh.model.mesh.field.setNumber(2, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(2, "XCenter", p1x)
    gmsh.model.mesh.field.setNumber(2, "YCenter", p1y)
    gmsh.model.mesh.field.setNumber(2, "ZCenter", 0.0)

    gmsh.model.mesh.field.add("Ball", 3)
    gmsh.model.mesh.field.setNumber(3, "Radius", 7.5)
    gmsh.model.mesh.field.setNumber(3, "VIn", h_pin)
    gmsh.model.mesh.field.setNumber(3, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(3, "XCenter", p2x)
    gmsh.model.mesh.field.setNumber(3, "YCenter", p2y)
    gmsh.model.mesh.field.setNumber(3, "ZCenter", 0.0)

    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1, 2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    gmsh.model.mesh.generate(2)
    n_nodes = length(gmsh.model.mesh.getNodes()[1])
    n_elems = sum(length.(gmsh.model.mesh.getElements(2)[2]))
    mesh_file = joinpath(output_dir, "CT_mesh_$(tag).msh")
    gmsh.write(mesh_file)
    gmsh.finalize()
    println("  mesh[$(tag)]: h_fine=$(h_fine) mm, $(n_nodes) nodes, $(n_elems) elements")
    return mesh_file, n_elems
end

# ============================================================================
# Heat conduction / displacement / phase-field solves
# ============================================================================
function solve_heat_conduction_CT(Th_old::FEFunction, sh::FEFunction, V_T,
                                  T_right_val::Float64, Qp_cell, dt::Float64, dΩ_)
    U_T_cur = TrialFESpace(V_T, [x -> T0, x -> T_right_val])
    k_field = ((c -> k1_heat +
                     thermal_conductivity_degradation_svolos(c) * (k0_heat - k1_heat)) ∘ sh)
    coeff = Cv_vol / dt
    a_T(T, ψ) = ∫(coeff * T * ψ + k_field * ∇(T) ⋅ ∇(ψ)) * dΩ_
    l_T(ψ)    = ∫((coeff * Th_old + Qp_cell) * ψ) * dΩ_
    op = AffineFEOperator(a_T, l_T, U_T_cur, V_T)
    Th_new = solve(op)
    T_vals = get_free_dof_values(Th_new)
    for i in eachindex(T_vals)
        (isnan(T_vals[i]) || isinf(T_vals[i])) && (T_vals[i] = T0)
    end
    return FEFunction(U_T_cur, T_vals)
end

function solve_displacement_CT(sh::FEFunction, plastic_states::Vector{PlasticStateData},
                               f_val::Float64, V_u, dΩ_, Ω_, uh_prev::FEFunction, cell_pts,
                               Th::FEFunction, dΓ_upper, dΓ_lower, f_prev::Float64 = 0.0)
    g_s_fn(s) = phase_field_degradation(s)
    n_c = num_cells(Ω_)
    pw  = [copy_plastic_state(ps) for ps in plastic_states]
    T_at_qp = evaluate(Th, cell_pts)

    uh = uh_prev
    Δf = f_val - f_prev
    n_load_substeps = max(1, ceil(Int, abs(Δf) / LOAD_SUBSTEP_SIZE))

    for load_substep in 1:n_load_substeps
        f_cur = f_prev + Δf * load_substep / n_load_substeps
        ps_substep_start = [copy_plastic_state(p) for p in pw]
        for _ in 1:10
            u_old = copy(get_free_dof_values(uh))
            σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, n_c)
            for i in 1:n_c
                T_loc = mean(T_at_qp[i][q] for q in 1:length(T_at_qp[i]))
                εth   = α_th * (T_loc - T0)
                εp11 = pw[i].εp[1,1]; εp22 = pw[i].εp[2,2]; εp12 = pw[i].εp[1,2]; εp33 = pw[i].εp33
                σeig11 = (mat.λ + 2.0 * mat.G) * (εp11 + εth) + mat.λ * (εp22 + εth) + mat.λ * (εp33 + εth)
                σeig22 = mat.λ * (εp11 + εth) + (mat.λ + 2.0 * mat.G) * (εp22 + εth) + mat.λ * (εp33 + εth)
                σeig12 = 2.0 * mat.G * εp12
                σp_vals[i] = SymTensorValue{2,Float64}(σeig11, σeig12, σeig22)
            end
            σp_field = CellField(σp_vals, Ω_)
            t_upper = VectorValue(0.0,  pin_traction(f_cur))
            t_lower = VectorValue(0.0, -pin_traction(f_cur))
            a_u(u, v) = ∫((g_s_fn ∘ sh) * (C_tensor ⊙ ε_sym(u)) ⊙ ε_sym(v)) * dΩ_
            l_u(v)    = ∫((g_s_fn ∘ sh) * σp_field ⊙ ε_sym(v)) * dΩ_ +
                        ∫(t_upper ⋅ v) * dΓ_upper + ∫(t_lower ⋅ v) * dΓ_lower
            uh = solve(AffineFEOperator(a_u, l_u, V_u, V_u))
            for i in 1:n_c
                pw[i].εp = ps_substep_start[i].εp; pw[i].εp33 = ps_substep_start[i].εp33
                pw[i].εp_eq = ps_substep_start[i].εp_eq
                pw[i].α = ps_substep_start[i].α; pw[i].α33 = ps_substep_start[i].α33
            end
            ε_qp = evaluate(ε_sym(uh), cell_pts)
            s_qp = evaluate(sh, cell_pts)
            for i in 1:n_c
                T_loc = mean(T_at_qp[i][q] for q in 1:length(T_at_qp[i]))
                return_mapping_thermo!(ε_qp[i][1], pw[i], clamp(s_qp[i][1], 0.0, 1.0), T_loc; update_state = true)
            end
            Δrel = norm(get_free_dof_values(uh) - u_old) / (norm(get_free_dof_values(uh)) + 1e-10)
            Δrel < 1e-4 && break
        end
    end

    for i in 1:n_c
        plastic_states[i].εp = pw[i].εp; plastic_states[i].εp33 = pw[i].εp33
        plastic_states[i].εp_eq = pw[i].εp_eq
        plastic_states[i].α = pw[i].α; plastic_states[i].α33 = pw[i].α33
    end
    return uh
end

function solve_phase_field_xie(sh_prev::FEFunction, D_field, f_cell, V_s, U_s, dΩ_)
    a_s(s, ϕ) = ∫((f_cell / l0) * s * ϕ + f_cell * l0 * ∇(s) ⋅ ∇(ϕ) + 2.0 * D_field * s * ϕ) * dΩ_
    b_s(ϕ) = ∫(2.0 * D_field * ϕ) * dΩ_
    sh_new = solve(AffineFEOperator(a_s, b_s, U_s, V_s))
    s_new  = get_free_dof_values(sh_new)
    s_old  = get_free_dof_values(sh_prev)
    for i in eachindex(s_new)
        (isnan(s_new[i]) || isinf(s_new[i])) && (s_new[i] = s_old[i])
        s_new[i] = clamp(max(s_new[i], s_old[i]), 0.0, 1.0)
    end
    return FEFunction(U_s, s_new)
end

function track_crack_tip_CT(sh, Ω_; tol=0.5)
    coords = get_node_coordinates(Ω_)
    s_vals = get_free_dof_values(sh)
    x_max  = CRACK_TIP_X0
    for i in eachindex(s_vals)
        x = coords[i][1]; y = coords[i][2]
        if x > CRACK_TIP_X0 - l0 && abs(y - Y_MID) < 3.0 * l0 && s_vals[i] > tol
            x_max = max(x_max, x)
        end
    end
    return x_max
end

@inline function ct_geometry_factor(x::Float64)
    return ((2.0 + x) / (1.0 - x)^(3 / 2)) *
           (0.886 + 4.64 * x - 13.32 * x^2 + 14.72 * x^3 - 5.6 * x^4)
end

function stress_intensity_CT(P_total_3d::Float64, a::Float64)
    x = a / W_CT
    (0.0 < x < 1.0) || return NaN
    return P_total_3d / (THICKNESS * sqrt(W_CT)) * ct_geometry_factor(x)
end

function stress_intensity_range_CT(Pmax_3d::Float64, Pmin_3d::Float64, a::Float64)
    Kmax = stress_intensity_CT(Pmax_3d, a)
    Kmin = stress_intensity_CT(Pmin_3d, a)
    return Kmax, Kmin, Kmax - Kmin
end

function T_right_at_cycle(cycle::Int, T_right_max::Float64)
    return cycle <= N_HEAT_CYCLES ?
        T0 + (T_right_max - T0) * (cycle / N_HEAT_CYCLES) : T_right_max
end

# ============================================================================
# Single mesh-level run (CSV only)
# ============================================================================
function run_one_mesh_case(tag::String, mesh_file::String, h_fine::Float64,
                           n_elems::Int, T_right_max::Float64)
    out_dir = RESULT_ROOT
    mkpath(out_dir)
    println("\n" * "="^70)
    println("mesh case: $(tag)  (h_fine=$(h_fine) mm, l0=$(l0) mm, elements=$(n_elems))")
    println("="^70)

    model = GmshDiscreteModel(mesh_file)
    Ω_  = Triangulation(model)
    dΩ_ = Measure(Ω_, 2)
    order = 1

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["fixed_point", "rot_lock_y"],
                      dirichlet_masks=[(true, true), (false, true)])
    U_u0 = TrialFESpace(V_u, [x -> VectorValue(0.0, 0.0), x -> VectorValue(0.0, 0.0)])

    Γ_upper  = BoundaryTriangulation(model; tags=["pin_upper"]); dΓ_upper = Measure(Γ_upper, 2)
    Γ_lower  = BoundaryTriangulation(model; tags=["pin_lower"]); dΓ_lower = Measure(Γ_lower, 2)

    reffe_s = ReferenceFE(lagrangian, Float64, order)
    V_s = TestFESpace(model, reffe_s, conformity=:H1); U_s = TrialFESpace(V_s)

    reffe_T = ReferenceFE(lagrangian, Float64, order)
    V_T = TestFESpace(model, reffe_T, conformity=:H1, dirichlet_tags=["left", "right"])
    U_T_init = TrialFESpace(V_T, [x -> T0, x -> T0])
    Th = interpolate_everywhere(x -> T0, U_T_init)

    quad_cache = CellQuadrature(Ω_, 2)
    cell_pts   = get_cell_points(quad_cache)
    n_c        = num_cells(Ω_)

    plastic_states = [PlasticStateData() for _ in 1:n_c]
    HI_hist  = zeros(Float64, n_c); HII_hist = zeros(Float64, n_c)
    ᾱ_acc    = zeros(Float64, n_c); ψd_prev  = zeros(Float64, n_c)
    q_plastic = zeros(Float64, n_c)

    sh = interpolate_everywhere(x -> 0.0, U_s)
    uh = zero(U_u0)

    csv_path = joinpath(out_dir, "data.csv")
    open(csv_path, "w") do io
        println(io, "cycle,a_mm,delta_a_mm,da_dN_mm_per_cycle," *
                    "T_right_K,T_mean_K,T_max_K," *
                    "Kmax_MPa_sqrt_mm,Kmin_MPa_sqrt_mm,deltaK_MPa_sqrt_mm," *
                    "max_epspeq,max_damage")
    end

    n_half          = N_STEPS_PER_CYCLE ÷ 2
    load_steps      = range(F_MIN, F_MAX, length=n_half + 1)
    unload_steps    = range(F_MAX, F_MIN, length=n_half + 1)[2:end]
    steps_per_cycle = length(load_steps) + length(unload_steps)
    Δt_step_cycle   = Δt_mech_cycle / steps_per_cycle

    current_cycle = 0
    f_prev = F_MIN
    a_prev = CRACK_TIP_X0

    stall_count    = 0
    crack_initiated = false
    stall_limit = parse_env_int("STALL_CYCLES", 40)
    stall_tol   = parse_env_float("STALL_TOL", 1e-5)
    init_tol    = parse_env_float("INIT_DA", 5e-3)
    saved_snapshots = falses(length(SNAPSHOT_FRACTIONS))
    stop_extension = 30.0
    run_t0 = time()

    cycle_limit = CLI.smoke ? 1 : N_CYCLES
    while current_cycle < cycle_limit
        current_cycle += 1
        T_right_cur = T_right_at_cycle(current_cycle, T_right_max)
        substep = 0
        for f_range in [load_steps, unload_steps]
            for f_val in f_range
                substep += 1
                try
                    Qp_cell = CellField(q_plastic, Ω_)
                    Th = solve_heat_conduction_CT(Th, sh, V_T, T_right_cur, Qp_cell, Δt_step_cycle, dΩ_)
                catch e
                    @warn "heat solve failed" exception=(e, catch_backtrace())
                end
                try
                    εp_eq_before = [ps.εp_eq for ps in plastic_states]
                    uh = solve_displacement_CT(sh, plastic_states, f_val, V_u, dΩ_, Ω_, uh, cell_pts, Th, dΓ_upper, dΓ_lower, f_prev)
                    for i in 1:n_c
                        q_plastic[i] = plastic_heat_source_density(εp_eq_before[i], plastic_states[i].εp_eq, Δt_step_cycle, CHI_EFF)
                    end
                    f_prev = f_val
                catch e
                    isa(e, SingularException) && (println("  singular matrix, stop"); @goto done)
                    rethrow(e)
                end

                ε_qp = evaluate(ε_sym(uh), cell_pts)
                s_qp = evaluate(sh, cell_pts)
                T_qp = evaluate(Th, cell_pts)
                T_cell_mean = zeros(Float64, n_c)
                ψp_vals = zeros(Float64, n_c)
                D_vals  = zeros(Float64, n_c)
                for i in 1:n_c
                    T_cell_mean[i] = mean(T_qp[i][q] for q in 1:length(T_qp[i]))
                    ψI_max = 0.0; ψII_max = 0.0; ψd_max = 0.0
                    for q in 1:length(ε_qp[i])
                        T_loc = T_qp[i][q]; s_loc = clamp(s_qp[i][q], 0.0, 1.0)
                        ψI_q  = mode_I_elastic_energy_thermo(ε_qp[i][q], plastic_states[i], T_loc)
                        ψII_q = mode_II_elastic_energy_thermo(ε_qp[i][q], plastic_states[i], T_loc)
                        ψplus_q = ψI_q + ψII_q
                        ψI_max  = max(ψI_max, ψI_q); ψII_max = max(ψII_max, ψII_q)
                        ψd_max  = max(ψd_max, phase_field_degradation(s_loc) * ψplus_q)
                    end
                    HI_hist[i]  = max(HI_hist[i], ψI_max)
                    HII_hist[i] = max(HII_hist[i], ψII_max)
                    ψp_vals[i]  = (1.0 - CHI_EFF) * plastic_strain_energy(plastic_states[i].εp_eq)
                    ᾱ_acc[i]   += max(ψd_max - ψd_prev[i], 0.0)
                    ψd_prev[i]  = ψd_max
                end
                if phase_every > 0 && substep % phase_every == 0
                    Gc_vals = [Gc_T(T_cell_mean[i]) for i in 1:n_c]
                    f_vals  = [fatigue_degradation_T(ᾱ_acc[i], T_cell_mean[i]) for i in 1:n_c]
                    for i in 1:n_c
                        D_vals[i] = (HI_hist[i] + HII_hist[i] + ψp_vals[i]) / max(Gc_vals[i], 1e-12)
                    end
                    D_field = CellField(D_vals, Ω_); f_cell = CellField(f_vals, Ω_)
                    for _ in 1:max_stagger
                        s_old = copy(get_free_dof_values(sh))
                        sh = solve_phase_field_xie(sh, D_field, f_cell, V_s, U_s, dΩ_)
                        s_new = get_free_dof_values(sh)
                        norm(s_new - s_old) / (norm(s_new) + 1e-10) < stagger_tol && break
                    end
                end
            end
        end

        a_cur = track_crack_tip_CT(sh, Ω_)
        Δa = a_cur - CRACK_TIP_X0
        da_dN = a_cur - a_prev; a_prev = a_cur
        Δa > init_tol && (crack_initiated = true)
        crack_initiated && (stall_count = da_dN < stall_tol ? stall_count + 1 : 0)
        Kmax_cur, Kmin_cur, ΔK_cur = stress_intensity_range_CT(F_MAX_3D, F_MIN_3D, a_cur)
        T_dofs = get_free_dof_values(Th)
        T_mean = mean(T_dofs); T_max = maximum(T_dofs)
        max_ep = maximum(ps.εp_eq for ps in plastic_states)
        max_s  = maximum(get_free_dof_values(sh))

        open(csv_path, "a") do io
            @printf(io, "%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    current_cycle, a_cur, Δa, da_dN, T_right_cur, T_mean, T_max,
                    Kmax_cur, Kmin_cur, ΔK_cur, max_ep, max_s)
        end

        if CLI.snapshots
            for i in eachindex(SNAPSHOT_FRACTIONS)
                if !saved_snapshots[i] && Δa >= SNAPSHOT_FRACTIONS[i] * stop_extension
                    frac = round(Int, 100 * SNAPSHOT_FRACTIONS[i])
                    writevtk(Ω_, joinpath(out_dir, @sprintf("milestone_da%02dpct", frac)),
                             cellfields=[
                                 "u" => uh, "d" => sh, "T" => Th,
                                 "epspeq" => CellField([ps.εp_eq for ps in plastic_states], Ω_),
                                 "plastic_strain" => CellField([ps.εp for ps in plastic_states], Ω_),
                                 "backstress" => CellField([ps.α for ps in plastic_states], Ω_),
                                 "HI_hist" => CellField(HI_hist, Ω_),
                                 "HII_hist" => CellField(HII_hist, Ω_),
                                 "alphabar" => CellField(ᾱ_acc, Ω_),
                                 "psi_d" => CellField(ψd_prev, Ω_)], nsubcells=4)
                    saved_snapshots[i] = true
                end
            end
        end

        if current_cycle <= 5 || current_cycle % PRINT_CYCLE_EVERY == 0
            @printf("  N=%d Tmax=%.0fK ep=%.5f s=%.4f da=%.4fmm dK=%.2f\n",
                    current_cycle, T_max, max_ep, max_s, Δa, ΔK_cur)
        end

        current_cycle % FORCE_GC_EVERY == 0 && GC.gc(false)
        (Δa > 30.0 || a_cur > W_CT - 5.0) && (println("  crack near right edge, stop"); break)
        if crack_initiated && stall_count >= stall_limit
            println("  crack growth stalled for $(stall_limit) cycles (Δa=$(round(Δa, digits=4)) mm), stop")
            break
        end
    end

    @label done
    if CLI.snapshots
        writevtk(Ω_, joinpath(out_dir, "final"),
                 cellfields=[
                     "u" => uh, "d" => sh, "T" => Th,
                     "epspeq" => CellField([ps.εp_eq for ps in plastic_states], Ω_),
                     "plastic_strain" => CellField([ps.εp for ps in plastic_states], Ω_),
                     "backstress" => CellField([ps.α for ps in plastic_states], Ω_),
                     "HI_hist" => CellField(HI_hist, Ω_),
                     "HII_hist" => CellField(HII_hist, Ω_),
                     "alphabar" => CellField(ᾱ_acc, Ω_),
                     "psi_d" => CellField(ψd_prev, Ω_)], nsubcells=4)
    end
    open(joinpath(out_dir, "timing.csv"), "w") do io
        println(io, "elapsed_s,n_cycles,n_nodes,n_elements")
        @printf(io, "%.6f,%d,%d,%d\n", time() - run_t0, current_cycle,
                length(get_node_coordinates(Ω_)), n_elems)
    end
    final_a  = a_prev
    final_da = final_a - CRACK_TIP_X0
    println("  done: N=$(current_cycle), Δa_final=$(round(final_da, digits=4)) mm")
    return (tag = tag, h_fine = h_fine, n_elems = n_elems,
            N_end = current_cycle, a_final = final_a, da_final = final_da)
end

# ============================================================================
# Main: sweep over crack-tip mesh size
# ============================================================================
function main()
    println("\n" * "="^70)
    println("Canonical ΔT=200 K thermo-elasto-plastic CT fatigue benchmark")
    @printf("h_tip=l0/2=%.4f mm\n", H_FINE)
    @printf("N_cycles=%d, steps/cycle=%d, chi_eff=%.2f, dT=%dK\n",
            N_CYCLES, N_STEPS_PER_CYCLE, CHI_EFF, round(Int, T_RIGHT_MAX - T0))
    println("="^70)

    mkpath(RESULT_ROOT)
    mesh_file, n_elems = create_CT_mesh(H_FINE; output_dir=RESULT_ROOT, tag="canonical")
    run_one_mesh_case("canonical", mesh_file, H_FINE, n_elems, T_RIGHT_MAX)
    println("\nCanonical run complete: $(joinpath(RESULT_ROOT, "data.csv"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
