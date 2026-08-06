# ============================================================================
# CT试样 Chaboche 弹塑性疲劳裂纹扩展（等温，对标 Xie et al. 2023 Fig.9）
#
# 设计目标：
#   - 几何、加载、相场、疲劳形式与 Xie_fig9.jl 完全一致（等温，无热传导）
#   - 塑性本构改为 Chaboche（Voce 等向 + Armstrong-Frederick 随动），屈服在有效应力空间（Jia 风格：不乘损伤）
#   - 可调参数：σy0, Q∞, b, C, γ（默认给出一组值，便于后续逼近 Xie 曲线）
#
# 论文: Xie et al. (2023) EFM 282, 109176 — Section 4.2 CT 模式I疲劳
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
using DelimitedFiles
ENV["GKSwstype"] = "100"
using Plots
gr(show = false)

function parse_cli(args)
    output_dir = normpath(joinpath(
        @__DIR__, "..", "outputs", "results_cpu", "plastic_fatigue_CT", "julia"
    ))
    benchmark = "cycle_jump_100_defaults"
    visualization = true
    snapshots = true
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--output-dir"
            i == length(args) && error("--output-dir requires a path")
            i += 1
            output_dir = args[i]
        elseif arg == "--benchmark"
            i == length(args) && error("--benchmark requires a value")
            i += 1
            benchmark = args[i]
        elseif arg == "--no-visualization"
            visualization = false
        elseif arg == "--no-snapshots"
            snapshots = false
        else
            error("Unknown argument: $(arg)")
        end
        i += 1
    end
    benchmark == "cycle_jump_100_defaults" ||
        error("Supported benchmark: cycle_jump_100_defaults")
    return (output_dir=abspath(output_dir), benchmark=benchmark,
            visualization=visualization, snapshots=snapshots)
end

const CLI = parse_cli(ARGS)

# ============================================================================
# 几何参数（与 Xie_fig9.jl 一致）
# ============================================================================
const W_CT = 50.0
const H_CT = 48.0
const THICKNESS = 25.0

const PIN_RADIUS = 5.0
const PIN1_X = 8.0
const PIN1_Y = 13.0
const PIN2_X = 8.0
const PIN2_Y = 35.0

const Y_CENTER = 24.0
const V_TIP_X = 14.0   # 初始裂纹长度 a0 = 14 mm
const V_TIP_Y = Y_CENTER
const V_ANGLE = 30.0
const V_BASE_WIDTH = 2.0

const V_HALF_WIDTH = V_BASE_WIDTH / 2.0
const V_HALF_ANGLE_RAD = deg2rad(V_ANGLE / 2.0)
const V_DEPTH = V_HALF_WIDTH / tan(V_HALF_ANGLE_RAD)
const RECT_LENGTH = V_TIP_X - V_DEPTH

# ============================================================================
# 材料参数 — Chaboche（GH4169 弹性基数 + 可调塑性）
# ============================================================================
struct MaterialData
    E::Float64
    ν::Float64
    G::Float64
    λ::Float64
    σy0::Float64   # 初始屈服 [MPa]
    Q∞::Float64    # Voce 等向硬化饱和 [MPa]
    b::Float64     # Voce 速率
    C::Float64     # AF 随动模量 [MPa]
    γ::Float64     # AF 动态回复
end

function create_material()
    E = 1.61e5
    ν = 0.3
    G = E / (2.0 * (1.0 + ν))
    λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))
    σy0 = 900.0
    Q∞  = 80.0
    b   = 5.0
    C   = 12_000.0
    γ   = 80.0
    return MaterialData(E, ν, G, λ, σy0, Q∞, b, C, γ)
end

const mat = create_material()

# ============================================================================
# 相场与疲劳参数（与 Xie 一致）
# ============================================================================
const l0 = 0.4
const GcI = 47.5
const GcII = 47.5
const η_stab = 1e-4
const αT = 5000.0

const F_MAX_3D = 25000.0
const F_MAX = F_MAX_3D / THICKNESS
const R_LOAD = 0.1
const F_MIN = R_LOAD * F_MAX

const N_CYCLES_TOTAL = 60000
const N_STEPS_PER_CYCLE = 20
const CYCLE_JUMP = 100
const phase_every = 1
const max_stagger_default = 20
const stagger_tol_default = 1e-6
const verbose_substep = false

const h_coarse = 2.0
const h_fine = 0.2
const OUTPUT_DIR = CLI.output_dir
const STOP_CRACK_X = 36.0
const VTK_CRACK_FRACTIONS = [0.10, 0.30, 0.60, 0.90]

# ============================================================================
# 弹性刚度
# ============================================================================
function elastic_stiffness_2D(m::MaterialData)
    E, ν = m.E, m.ν
    C1111 = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    C1122 = E * ν / ((1 + ν) * (1 - 2ν))
    C1212 = E / (2 * (1 + ν))
    return SymFourthOrderTensorValue(
        C1111, 0.0, C1122,
        0.0, C1212, 0.0,
        C1122, 0.0, C1111
    )
end

const C_tensor = elastic_stiffness_2D(mat)
const ε = symmetric_gradient

# ============================================================================
# Chaboche 塑性状态与本构
# ============================================================================
mutable struct PlasticStateData
    εp::SymTensorValue{2,Float64,3}
    εp33::Float64
    εp_eq::Float64
    α::SymTensorValue{2,Float64,3}
    α33::Float64
end

PlasticStateData() = PlasticStateData(
    SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0, 0.0,
    SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0
)

function copy_plastic_state(ps::PlasticStateData)
    return PlasticStateData(ps.εp, ps.εp33, ps.εp_eq, ps.α, ps.α33)
end

function yield_stress(εp_eq::Float64)
    return mat.σy0 + mat.Q∞ * (1.0 - exp(-mat.b * εp_eq))
end

function dσY_dεp(εp_eq::Float64)
    return mat.Q∞ * mat.b * exp(-mat.b * εp_eq)
end

function plastic_strain_energy(εp_eq::Float64)
    abs(εp_eq) < 1e-12 && return 0.0
    return mat.σy0 * εp_eq +
           mat.Q∞ * (εp_eq + (exp(-mat.b * εp_eq) - 1.0) / mat.b)
end

@inline function elastic_stress_plane_strain(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData
)
    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33
    trεe = εe11 + εe22 + εe33
    σ11 = 2.0 * mat.G * εe11 + mat.λ * trεe
    σ22 = 2.0 * mat.G * εe22 + mat.λ * trεe
    σ33 = 2.0 * mat.G * εe33 + mat.λ * trεe
    σ12 = 2.0 * mat.G * εe12
    return σ11, σ22, σ33, σ12
end

@inline function von_mises_eq_3d(σ11, σ22, σ33, σ12)
    p = (σ11 + σ22 + σ33) / 3.0
    s11 = σ11 - p
    s22 = σ22 - p
    s33 = σ33 - p
    s12 = σ12
    σeq = sqrt(max(0.0, 1.5 * (s11^2 + s22^2 + s33^2 + 2.0 * s12^2)))
    return σeq, s11, s22, s33, s12
end

# Chaboche 径向返回：有效应力空间屈服 F = ξ_eq - σy，输出名义应力 g(s)*σ_eff（Jia 风格）
function return_mapping_chaboche!(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    s_phasefield::Float64;
    update_state::Bool = true
)
    σ11_tr, σ22_tr, σ33_tr, σ12_tr = elastic_stress_plane_strain(ε_total, plastic)
    σeq_tr, s11, s22, s33, s12 = von_mises_eq_3d(σ11_tr, σ22_tr, σ33_tr, σ12_tr)
    degradation = (1.0 - s_phasefield)^2 + η_stab

    α11_n = plastic.α[1,1]
    α22_n = plastic.α[2,2]
    α12_n = plastic.α[1,2]
    α33_n = plastic.α33
    ξ11 = s11 - α11_n
    ξ22 = s22 - α22_n
    ξ12 = s12 - α12_n
    ξ33 = s33 - α33_n
    ξ_eq = sqrt(max(0.0, 1.5 * (ξ11^2 + ξ22^2 + ξ33^2 + 2.0 * ξ12^2)))
    inv_ξ = ξ_eq > 1e-12 ? (1.0 / ξ_eq) : 0.0
    n11 = 1.5 * ξ11 * inv_ξ
    n22 = 1.5 * ξ22 * inv_ξ
    n33 = 1.5 * ξ33 * inv_ξ
    n12 = 1.5 * ξ12 * inv_ξ

    σy = yield_stress(plastic.εp_eq)
    f_trial = ξ_eq - σy
    if f_trial <= 0.0 || ξ_eq < 1e-12
        return degradation * SymTensorValue{2,Float64}(σ11_tr, σ12_tr, σ22_tr)
    end

    kin_coeff = (2.0 / 3.0) * mat.C
    denom0 = 3.0 * mat.G + mat.C + dσY_dεp(plastic.εp_eq)
    Δγ = max(f_trial / max(denom0, 1e-12), 0.0)
    tol = 1e-10 * (ξ_eq + mat.σy0)
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
        ξ11_loc = s11_loc - α11_loc
        ξ22_loc = s22_loc - α22_loc
        ξ33_loc = s33_loc - α33_loc
        ξ12_loc = s12_loc - α12_loc
        ξ_eq_loc = sqrt(max(0.0, 1.5 * (ξ11_loc^2 + ξ22_loc^2 + ξ33_loc^2 + 2.0 * ξ12_loc^2)))
        if ξ_eq_loc < 1e-18
            return -yield_stress(plastic.εp_eq + Δγ_loc), 0.0
        end
        gval = ξ_eq_loc - yield_stress(plastic.εp_eq + Δγ_loc)
        dα11_dΔ = (kin_coeff * n11 - mat.γ * α11_n) / denom_γ^2
        dα22_dΔ = (kin_coeff * n22 - mat.γ * α22_n) / denom_γ^2
        dα33_dΔ = (kin_coeff * n33 - mat.γ * α33_n) / denom_γ^2
        dα12_dΔ = (kin_coeff * n12 - mat.γ * α12_n) / denom_γ^2
        dξ11_dΔ = -2.0 * mat.G * n11 - dα11_dΔ
        dξ22_dΔ = -2.0 * mat.G * n22 - dα22_dΔ
        dξ33_dΔ = -2.0 * mat.G * n33 - dα33_dΔ
        dξ12_dΔ = -2.0 * mat.G * n12 - dα12_dΔ
        ξ_dξ = ξ11_loc * dξ11_dΔ + ξ22_loc * dξ22_dΔ + ξ33_loc * dξ33_dΔ + 2.0 * ξ12_loc * dξ12_dΔ
        dξeq_dΔ = 1.5 * ξ_dξ / ξ_eq_loc
        dgdΔ = dξeq_dΔ - dσY_dεp(plastic.εp_eq + Δγ_loc)
        return gval, dgdΔ
    end

    Δγ_lo, Δγ_hi = 0.0, max(2.0 * Δγ, 1e-12)
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
            Δγ_new = clamp(Δγ_new, Δγ_lo + 1e-14, Δγ_hi - 1e-14)
        else
            Δγ_new = abs(dgdΔ) < 1e-30 ? 1.5 * Δγ : (Δγ - gval / dgdΔ)
            Δγ_new = max(Δγ_new, 1e-14)
        end
        Δγ = Δγ_new
    end

    converged || @warn "Chaboche return mapping did not converge" maxlog = 5

    denom_γ = 1.0 + mat.γ * Δγ
    εp_eq_new = plastic.εp_eq + Δγ
    Δεp11 = Δγ * n11
    Δεp22 = Δγ * n22
    Δεp33 = Δγ * n33
    Δεp12 = Δγ * n12
    α11_new = (α11_n + kin_coeff * n11 * Δγ) / denom_γ
    α22_new = (α22_n + kin_coeff * n22 * Δγ) / denom_γ
    α33_new = (α33_n + kin_coeff * n33 * Δγ) / denom_γ
    α12_new = (α12_n + kin_coeff * n12 * Δγ) / denom_γ

    if update_state
        plastic.εp = plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22)
        plastic.εp33 += Δεp33
        plastic.εp_eq = εp_eq_new
        plastic.α = SymTensorValue{2,Float64}(α11_new, α12_new, α22_new)
        plastic.α33 = α33_new
        σ11, σ22, σ33, σ12 = elastic_stress_plane_strain(ε_total, plastic)
    else
        plastic_tmp = PlasticStateData(
            plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22),
            plastic.εp33 + Δεp33, εp_eq_new,
            SymTensorValue{2,Float64}(α11_new, α12_new, α22_new), α33_new
        )
        σ11, σ22, σ33, σ12 = elastic_stress_plane_strain(ε_total, plastic_tmp)
    end
    return degradation * SymTensorValue{2,Float64}(σ11, σ12, σ22)
end

# ============================================================================
# 能量分解（与 Xie 一致，用弹性应变 ε_e = ε - εp）
# ============================================================================
function principal_strains_2D(ε::SymTensorValue{2,Float64,3})
    ε11, ε12, ε22 = ε[1,1], ε[1,2], ε[2,2]
    trace_ε = ε11 + ε22
    det_ε = ε11 * ε22 - ε12^2
    discriminant = sqrt(max(0.0, (trace_ε/2)^2 - det_ε))
    ε1 = trace_ε/2 + discriminant
    ε2 = trace_ε/2 - discriminant
    return ε1, ε2
end

macaulay_bracket(x::Real) = max(x, 0.0)

function tensile_strain_energy(ε::SymTensorValue{2,Float64,3})
    ε1, ε2 = principal_strains_2D(ε)
    ε1_pos = macaulay_bracket(ε1)
    ε2_pos = macaulay_bracket(ε2)
    tr_ε_pos = macaulay_bracket(ε1 + ε2)
    return 0.5 * mat.λ * tr_ε_pos^2 + mat.G * (ε1_pos^2 + ε2_pos^2)
end

function mode_I_energy(ε::SymTensorValue{2,Float64,3})
    ε1, ε2 = principal_strains_2D(ε)
    tr_ε_pos = macaulay_bracket(ε1 + ε2)
    return 0.5 * mat.λ * tr_ε_pos^2
end

function mode_II_energy(ε::SymTensorValue{2,Float64,3})
    ε1, ε2 = principal_strains_2D(ε)
    ε1_pos = macaulay_bracket(ε1)
    ε2_pos = macaulay_bracket(ε2)
    return mat.G * (ε1_pos^2 + ε2_pos^2)
end

function update_history_variable(ψ_new::Real, H_old::Real)
    return (true, max(H_old, ψ_new))
end

function fatigue_degradation(ᾱ::Real)
    return (αT / (ᾱ + αT + 1e-12))^2
end

function phase_field_degradation(s::Real)
    return (1.0 - s)^2 + η_stab
end

function project_to_fe_space(q, V_proj, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * q) * dΩ
    op = AffineFEOperator(a, b, V_proj, V_proj)
    return solve(op)
end

# ============================================================================
# 网格（与 Xie 完全一致：V 型缺口 CT）
# ============================================================================
function create_CT_mesh(h_fine_local::Float64, h_coarse_local::Float64; output_dir::String = OUTPUT_DIR)
    println("\n创建CT试样网格（Xie 几何）...")
    W, H = W_CT, H_CT
    pin_r = PIN_RADIUS
    pin1_x, pin1_y = PIN1_X, PIN1_Y
    pin2_x, pin2_y = PIN2_X, PIN2_Y
    v_tip_x, v_tip_y = V_TIP_X, V_TIP_Y
    rect_len, rect_half_w = RECT_LENGTH, V_HALF_WIDTH

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CT_Xie_Chaboche")

    pt = 1
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_coarse_local, pt); pt_LB = pt; pt += 1
    gmsh.model.geo.addPoint(W, 0.0, 0.0, h_coarse_local, pt); pt_RB = pt; pt += 1
    gmsh.model.geo.addPoint(W, H, 0.0, h_coarse_local, pt); pt_RT = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, H, 0.0, h_coarse_local, pt); pt_LT = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, v_tip_y + rect_half_w, 0.0, h_fine_local, pt); pt_rect_UL = pt; pt += 1
    gmsh.model.geo.addPoint(rect_len, v_tip_y + rect_half_w, 0.0, h_fine_local, pt); pt_rect_UR = pt; pt += 1
    gmsh.model.geo.addPoint(rect_len, v_tip_y - rect_half_w, 0.0, h_fine_local, pt); pt_rect_LR = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, v_tip_y - rect_half_w, 0.0, h_fine_local, pt); pt_rect_LL = pt; pt += 1
    gmsh.model.geo.addPoint(v_tip_x, v_tip_y, 0.0, h_fine_local, pt); pt_v_tip = pt; pt += 1
    gmsh.model.geo.addPoint(pin1_x, pin1_y, 0.0, h_coarse_local, pt); pt_pin1_C = pt; pt += 1
    gmsh.model.geo.addPoint(pin1_x + pin_r, pin1_y, 0.0, h_coarse_local, pt); pt_pin1_E = pt; pt += 1
    gmsh.model.geo.addPoint(pin1_x, pin1_y + pin_r, 0.0, h_coarse_local, pt); pt_pin1_N = pt; pt += 1
    gmsh.model.geo.addPoint(pin1_x - pin_r, pin1_y, 0.0, h_coarse_local, pt); pt_pin1_W = pt; pt += 1
    gmsh.model.geo.addPoint(pin1_x, pin1_y - pin_r, 0.0, h_coarse_local, pt); pt_pin1_S = pt; pt += 1
    gmsh.model.geo.addPoint(pin2_x, pin2_y, 0.0, h_coarse_local, pt); pt_pin2_C = pt; pt += 1
    gmsh.model.geo.addPoint(pin2_x + pin_r, pin2_y, 0.0, h_coarse_local, pt); pt_pin2_E = pt; pt += 1
    gmsh.model.geo.addPoint(pin2_x, pin2_y + pin_r, 0.0, h_coarse_local, pt); pt_pin2_N = pt; pt += 1
    gmsh.model.geo.addPoint(pin2_x - pin_r, pin2_y, 0.0, h_coarse_local, pt); pt_pin2_W = pt; pt += 1
    gmsh.model.geo.addPoint(pin2_x, pin2_y - pin_r, 0.0, h_coarse_local, pt); pt_pin2_S = pt; pt += 1

    line = 1
    gmsh.model.geo.addLine(pt_LB, pt_RB, line); line_bottom = line; line += 1
    gmsh.model.geo.addLine(pt_RB, pt_RT, line); line_right = line; line += 1
    gmsh.model.geo.addLine(pt_RT, pt_LT, line); line_top = line; line += 1
    gmsh.model.geo.addLine(pt_LT, pt_rect_UL, line); line_left_upper = line; line += 1
    gmsh.model.geo.addLine(pt_rect_UL, pt_rect_UR, line); line_notch_top = line; line += 1
    gmsh.model.geo.addLine(pt_rect_UR, pt_v_tip, line); line_v_upper = line; line += 1
    gmsh.model.geo.addLine(pt_v_tip, pt_rect_LR, line); line_v_lower = line; line += 1
    gmsh.model.geo.addLine(pt_rect_LR, pt_rect_LL, line); line_notch_bottom = line; line += 1
    gmsh.model.geo.addLine(pt_rect_LL, pt_LB, line); line_left_lower = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin1_E, pt_pin1_C, pt_pin1_N, line); line_p1_1 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin1_N, pt_pin1_C, pt_pin1_W, line); line_p1_2 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin1_W, pt_pin1_C, pt_pin1_S, line); line_p1_3 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin1_S, pt_pin1_C, pt_pin1_E, line); line_p1_4 = line; line += 1
    line_pin1 = [line_p1_1, line_p1_2, line_p1_3, line_p1_4]
    gmsh.model.geo.addCircleArc(pt_pin2_E, pt_pin2_C, pt_pin2_N, line); line_p2_1 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin2_N, pt_pin2_C, pt_pin2_W, line); line_p2_2 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin2_W, pt_pin2_C, pt_pin2_S, line); line_p2_3 = line; line += 1
    gmsh.model.geo.addCircleArc(pt_pin2_S, pt_pin2_C, pt_pin2_E, line); line_p2_4 = line; line += 1
    line_pin2 = [line_p2_1, line_p2_2, line_p2_3, line_p2_4]

    outer_loop = [line_bottom, line_right, line_top, line_left_upper,
                  line_notch_top, line_v_upper, line_v_lower, line_notch_bottom, line_left_lower]
    gmsh.model.geo.addCurveLoop(outer_loop, 1)
    gmsh.model.geo.addCurveLoop(line_pin1, 2)
    gmsh.model.geo.addCurveLoop(line_pin2, 3)
    gmsh.model.geo.addPlaneSurface([1, 2, 3], 1)
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, line_pin2, 1)
    gmsh.model.setPhysicalName(1, 1, "pin_upper")
    gmsh.model.addPhysicalGroup(1, line_pin1, 2)
    gmsh.model.setPhysicalName(1, 2, "pin_lower")
    gmsh.model.addPhysicalGroup(0, [pt_RB], 3)
    gmsh.model.setPhysicalName(0, 3, "fixed_point")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine_local)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse_local)
    gmsh.model.mesh.field.setNumber(1, "XMin", v_tip_x - 2.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", W)
    gmsh.model.mesh.field.setNumber(1, "YMin", v_tip_y - 5.0)
    gmsh.model.mesh.field.setNumber(1, "YMax", v_tip_y + 5.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)

    mesh_file = joinpath(output_dir, "CT_mesh.msh")
    gmsh.write(mesh_file)
    gmsh.finalize()
    println("  网格已写: $(mesh_file)")
    return mesh_file
end

function plot_mesh(mesh_file::String; output_dir::String = OUTPUT_DIR)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(mesh_file)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = Dict{Int,Tuple{Float64,Float64}}()
    for i in eachindex(node_tags)
        j = 3 * (i - 1)
        node_map[Int(node_tags[i])] = (node_coords[j+1], node_coords[j+2])
    end
    tri_nodes = gmsh.model.mesh.getElementsByType(2)[2]
    p = plot(aspect_ratio = :equal, size = (900, 700), dpi = 220,
             xlabel = "x (mm)", ylabel = "y (mm)", legend = false, grid = false,
             title = "CT mesh")
    for i in 1:(length(tri_nodes) ÷ 3)
        ids = tri_nodes[3*(i-1)+1:3*i]
        pts = [node_map[Int(id)] for id in ids]
        xs = [pt[1] for pt in pts]
        ys = [pt[2] for pt in pts]
        plot!(p, Shape(xs, ys), fillcolor = :white, linecolor = :gray55, linewidth = 0.12)
    end
    savefig(p, joinpath(output_dir, "mesh_CT.png"))
    gmsh.finalize()
end

function save_cell_scalar_png(cell_coords, vals::Vector{Float64};
                              output_file::String,
                              title_str::String,
                              clims = (0.0, 1.0),
                              cbar_title = "")
    grad = cgrad(:jet)
    vmin, vmax = clims
    p = plot(aspect_ratio = :equal, size = (900, 700), dpi = 220,
             xlabel = "x (mm)", ylabel = "y (mm)", title = title_str,
             legend = false, grid = false, framestyle = :box)
    denom = max(vmax - vmin, 1e-12)
    for i in eachindex(cell_coords)
        xs = [v[1] for v in cell_coords[i]]
        ys = [v[2] for v in cell_coords[i]]
        ξ = clamp((vals[i] - vmin) / denom, 0.0, 1.0)
        plot!(p, Shape(xs, ys), fillcolor = grad[ξ], linecolor = :transparent, linewidth = 0.0, label = "")
    end
    heatmap!(p, [NaN NaN; NaN NaN], c = :jet, clims = clims, colorbar_title = cbar_title)
    savefig(p, output_file)
end

function save_vtu_state(
    Ω, tag::String, uh, sh, HI_field, HII_field, ᾱ_field,
    plastic_states, ψd_prev_array;
    output_dir::String = OUTPUT_DIR,
    cycle::Int, load::Float64, crack_extension::Float64,
    crack_tip_x::Float64, crack_tip_y::Float64, milestone::Float64
)
    snapshot_dir = joinpath(output_dir, "snapshots")
    mkpath(snapshot_dir)
    nc = length(plastic_states)
    writevtk(Ω, joinpath(snapshot_dir, tag),
             cellfields = [
                 "u" => uh, "s" => sh,
                 "plastic_strain" => CellField([ps.εp for ps in plastic_states], Ω),
                 "plastic_strain_33" => CellField([ps.εp33 for ps in plastic_states], Ω),
                 "equivalent_plastic_strain" => CellField([ps.εp_eq for ps in plastic_states], Ω),
                 "backstress" => CellField([ps.α for ps in plastic_states], Ω),
                 "backstress_33" => CellField([ps.α33 for ps in plastic_states], Ω),
                 "HI" => HI_field, "HII" => HII_field, "alphabar" => ᾱ_field,
                 "psi_d" => CellField(ψd_prev_array, Ω),
                 "cycle" => CellField(fill(Float64(cycle), nc), Ω),
                 "step" => CellField(fill(Float64(cycle), nc), Ω),
                 "load" => CellField(fill(load, nc), Ω),
                 "qoi_crack_extension_mm" => CellField(fill(crack_extension, nc), Ω),
                 "qoi_crack_tip_x_mm" => CellField(fill(crack_tip_x, nc), Ω),
                 "qoi_crack_tip_y_mm" => CellField(fill(crack_tip_y, nc), Ω),
                 "milestone" => CellField(fill(milestone, nc), Ω)
             ],
             nsubcells = 4)
end

# ============================================================================
# 位移求解（等温，无热应变）
# ============================================================================
function compute_elastoplastic_stress!(uh, sh, plastic_states::Vector{PlasticStateData}, Ω; cell_points = nothing, update_state::Bool = true)
    n_cells = num_cells(Ω)
    cell_points === nothing && (cell_points = get_cell_points(CellQuadrature(Ω, 2)))
    εh = ε(uh)
    ε_at_qp = evaluate(εh, cell_points)
    s_at_qp = evaluate(sh, cell_points)
    σ_values = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)
    for i in 1:n_cells
        ε_cell = ε_at_qp[i][1]
        s_cell = clamp(s_at_qp[i][1], 0.0, 1.0)
        σ_values[i] = return_mapping_chaboche!(ε_cell, plastic_states[i], s_cell; update_state = update_state)
    end
    return σ_values
end

function solve_displacement_step_picard(
    sh, plastic_states::Vector{PlasticStateData},
    t_val::Float64, V_u, U_u, dΩ, dΓ_upper, dΓ_lower, Ω,
    uh_prev, t_prev::Float64 = 0.0;
    cell_points_cache = nothing
)
    L_contact = π * PIN_RADIUS
    g_s(s) = (1.0 - s)^2 + η_stab
    n_cells = num_cells(Ω)
    plastic_work = [copy_plastic_state(ps) for ps in plastic_states]
    Δt = t_val - t_prev
    n_substeps = max(1, ceil(Int, abs(Δt) / 50.0))
    uh = uh_prev

    for substep in 1:n_substeps
        t_current = t_prev + Δt * substep / n_substeps
        t_line = t_current / L_contact
        t_upper_func(x) = VectorValue(0.0, +t_line)
        t_lower_func(x) = VectorValue(0.0, -t_line)
        plastic_substep_start = [copy_plastic_state(ps) for ps in plastic_work]

        for _ in 1:10
            u_old = copy(get_free_dof_values(uh))
            σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)
            for i in 1:n_cells
                σp_in = C_tensor ⊙ plastic_work[i].εp
                add = mat.λ * plastic_work[i].εp33
                σp_vals[i] = σp_in + SymTensorValue{2,Float64}(add, 0.0, add)
            end
            σp_field = CellField(σp_vals, Ω)
            a(u, v) = ∫((g_s ∘ sh) * (C_tensor ⊙ ε(u)) ⊙ ε(v)) * dΩ
            l(v) = ∫(t_upper_func ⋅ v) * dΓ_upper + ∫(t_lower_func ⋅ v) * dΓ_lower +
                   ∫((g_s ∘ sh) * σp_field ⊙ ε(v)) * dΩ
            op = AffineFEOperator(a, l, U_u, V_u)
            uh = solve(op)

            for i in 1:n_cells
                plastic_work[i].εp   = plastic_substep_start[i].εp
                plastic_work[i].εp33  = plastic_substep_start[i].εp33
                plastic_work[i].εp_eq = plastic_substep_start[i].εp_eq
                plastic_work[i].α    = plastic_substep_start[i].α
                plastic_work[i].α33  = plastic_substep_start[i].α33
            end
            compute_elastoplastic_stress!(uh, sh, plastic_work, Ω; cell_points = cell_points_cache, update_state = true)
            Δrel = norm(get_free_dof_values(uh) - u_old) / (norm(get_free_dof_values(uh)) + 1e-10)
            Δrel < 1e-3 && break
        end
    end
    for i in 1:n_cells
        plastic_states[i].εp   = plastic_work[i].εp
        plastic_states[i].εp33 = plastic_work[i].εp33
        plastic_states[i].εp_eq = plastic_work[i].εp_eq
        plastic_states[i].α    = plastic_work[i].α
        plastic_states[i].α33  = plastic_work[i].α33
    end
    return uh
end

function array_to_fe_field(arr, V_dg0, V_proj, dΩ)
    f_dg = FEFunction(V_dg0, arr)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * f_dg) * dΩ
    return solve(AffineFEOperator(a, b, V_proj, V_proj))
end

# ============================================================================
# 相场求解（与 Xie 一致）
# ============================================================================
function solve_phase_field(sh_prev, HI_field, HII_field, ψp_field, ᾱ_field, V_s, U_s, dΩ)
    f_dg = (x -> (αT / (x + αT + 1e-12))^2) ∘ ᾱ_field
    D = HI_field / GcI + HII_field / GcII + ψp_field / GcII
    a(s, ϕ) = ∫((f_dg / l0) * s * ϕ + f_dg * l0 * ∇(s) ⋅ ∇(ϕ) + 2.0 * D * s * ϕ) * dΩ
    b(ϕ) = ∫(2.0 * D * ϕ) * dΩ
    op = AffineFEOperator(a, b, U_s, V_s)
    sh_solved = solve(op)
    s_vals_new = get_free_dof_values(sh_solved)
    s_vals_old = get_free_dof_values(sh_prev)
    for i in 1:length(s_vals_new)
        (isnan(s_vals_new[i]) || isinf(s_vals_new[i])) && (s_vals_new[i] = s_vals_old[i])
        s_vals_new[i] = max(s_vals_new[i], s_vals_old[i])
        s_vals_new[i] = clamp(s_vals_new[i], 0.0, 1.0)
    end
    return FEFunction(U_s, s_vals_new)
end

function track_crack_tip(sh, Ω, tol = 0.5)
    coords = get_node_coordinates(Ω)
    s_vals = get_free_dof_values(sh)
    damaged_indices = Int[]
    for i in 1:length(s_vals)
        x, y, s = coords[i][1], coords[i][2], s_vals[i]
        if x > V_TIP_X - l0 && abs(y - Y_CENTER) < 3.0 * l0 && s > tol
            push!(damaged_indices, i)
        end
    end
    if isempty(damaged_indices)
        return (V_TIP_X, V_TIP_Y)
    end
    x_max, y_tip = V_TIP_X, Y_CENTER
    for idx in damaged_indices
        if coords[idx][1] > x_max
            x_max = coords[idx][1]
            y_tip = coords[idx][2]
        end
    end
    return (x_max, y_tip)
end

# ============================================================================
# 主模拟
# ============================================================================
function run_simulation(;
    output_dir::String = OUTPUT_DIR,
    h_fine_case::Float64 = h_fine,
    h_coarse_case::Float64 = h_coarse,
    stop_crack_x::Float64 = STOP_CRACK_X
)
    mkpath(output_dir)
    println("\n" * "="^80)
    println("CT 等温 Chaboche 疲劳裂纹扩展（对标 Xie Fig.9 几何与加载）")
    println("="^80)
    @printf("材料: E=%.0f GPa, ν=%.2f, σy0=%.0f MPa\n", mat.E/1e3, mat.ν, mat.σy0)
    @printf("Chaboche: Q∞=%.0f, b=%.2f, C=%.0f, γ=%.0f\n", mat.Q∞, mat.b, mat.C, mat.γ)
    @printf("相场: l0=%.2f, GcI=GcII=%.1f, αT=%.0f\n", l0, GcI, αT)
    @printf("加载: F_max=%.0f kN, R=%.2f\n", F_MAX_3D/1e3, R_LOAD)
    println("="^80)

    mesh_file = create_CT_mesh(h_fine_case, h_coarse_case; output_dir = output_dir)
    CLI.visualization && plot_mesh(mesh_file; output_dir = output_dir)
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    Γ_upper = BoundaryTriangulation(model, tags = "pin_upper")
    dΓ_upper = Measure(Γ_upper, 2)
    Γ_lower = BoundaryTriangulation(model, tags = "pin_lower")
    dΓ_lower = Measure(Γ_lower, 2)

    order = 1
    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V_u = TestFESpace(model, reffe_u, conformity = :H1, dirichlet_tags = ["fixed_point"], dirichlet_masks = [(true, true)])
    U_u = TrialFESpace(V_u, [x -> VectorValue(0.0, 0.0)])
    reffe_s = ReferenceFE(lagrangian, Float64, order)
    V_s = TestFESpace(model, reffe_s, conformity = :H1)
    U_s = TrialFESpace(V_s)
    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = FESpace(model, reffe_dg0, conformity = :L2)
    reffe_L2 = ReferenceFE(lagrangian, Float64, order)
    V_L2 = FESpace(model, reffe_L2, conformity = :L2)
    quad_cache = CellQuadrature(Ω, 2)
    cell_points_cache = get_cell_points(quad_cache)
    n_cells = num_cells(Ω)
    println("  单元数: $(n_cells)")

    plastic_states = [PlasticStateData() for _ in 1:n_cells]
    HI_prev = CellState(0.0, dΩ)
    HII_prev = CellState(0.0, dΩ)
    ψd_prev_array = zeros(Float64, n_cells)
    ᾱ_array = zeros(Float64, n_cells)
    sh = interpolate_everywhere(x -> 0.0, U_s)
    uh = zero(U_u)

    cycle_history = Int[]
    time_history = Float64[]
    crack_length_history = Float64[]
    tip_x_history = Float64[]
    smax_history = Float64[]
    alphap_history = Float64[]
    alphabar_history = Float64[]
    fatigue_history = Float64[]
    run_t0 = time()
    target_da = stop_crack_x - V_TIP_X
    saved_vtu = falses(length(VTK_CRACK_FRACTIONS))

    t_max, t_min = F_MAX, F_MIN
    n_half = N_STEPS_PER_CYCLE ÷ 2
    load_steps = range(t_min, t_max, length = n_half + 1)
    unload_steps = range(t_max, t_min, length = n_half + 1)[2:end]
    current_cycle = 0
    t_prev = t_min
    stop_due_to_singular = false

    println("\n开始循环加载（等温）...")
    while current_cycle < N_CYCLES_TOTAL
        block_end = min(current_cycle + CYCLE_JUMP, N_CYCLES_TOTAL)
        actual_jump = min(CYCLE_JUMP, N_CYCLES_TOTAL - current_cycle)
        substep_counter = 0

        for (_, t_range) in [("加载", load_steps), ("卸载", unload_steps)]
            for t_val in t_range
                substep_counter += 1
                try
                    uh = solve_displacement_step_picard(
                        sh, plastic_states, t_val, V_u, U_u, dΩ, dΓ_upper, dΓ_lower, Ω,
                        uh, t_prev; cell_points_cache = cell_points_cache
                    )
                catch e
                    if isa(e, SingularException)
                        println("\n  奇异矩阵，停止 N=$(current_cycle)")
                        stop_due_to_singular = true
                        break
                    else
                        rethrow(e)
                    end
                end

                ε_at_qp = evaluate(ε(uh), cell_points_cache)
                s_at_qp = evaluate(sh, cell_points_cache)
                ψI_per_cell = zeros(n_cells)
                ψII_per_cell = zeros(n_cells)
                ψd_plus_per_cell = zeros(n_cells)
                for i in 1:n_cells
                    ψI_max = ψII_max = ψd_max = 0.0
                    for q in 1:length(ε_at_qp[i])
                        ε_e_qp = ε_at_qp[i][q] - plastic_states[i].εp
                        s_qp = clamp(s_at_qp[i][q], 0.0, 1.0)
                        g_qp = phase_field_degradation(s_qp)
                        ψd_qp = g_qp * tensile_strain_energy(ε_e_qp)
                        ψI_max = max(ψI_max, mode_I_energy(ε_e_qp))
                        ψII_max = max(ψII_max, mode_II_energy(ε_e_qp))
                        ψd_max = max(ψd_max, ψd_qp)
                    end
                    ψI_per_cell[i] = ψI_max
                    ψII_per_cell[i] = ψII_max
                    ψd_plus_per_cell[i] = ψd_max
                end
                ψI_dg = FEFunction(V_dg0, ψI_per_cell)
                ψII_dg = FEFunction(V_dg0, ψII_per_cell)
                update_state!(update_history_variable, HI_prev, ψI_dg)
                update_state!(update_history_variable, HII_prev, ψII_dg)
                for i in 1:n_cells
                    Δψd = max(0.0, ψd_plus_per_cell[i] - ψd_prev_array[i])
                    ᾱ_array[i] += actual_jump * Δψd
                    ψd_prev_array[i] = ψd_plus_per_cell[i]
                end

                if phase_every > 0 && substep_counter % phase_every == 0
                    for _ in 1:max_stagger_default
                        s_vals_old = copy(get_free_dof_values(sh))
                        HI_field = project_to_fe_space(HI_prev, V_L2, dΩ)
                        HII_field = project_to_fe_space(HII_prev, V_L2, dΩ)
                        ψp_array = [plastic_strain_energy(ps.εp_eq) for ps in plastic_states]
                        ψp_field = array_to_fe_field(ψp_array, V_dg0, V_L2, dΩ)
                        ᾱ_field = array_to_fe_field(ᾱ_array, V_dg0, V_L2, dΩ)
                        sh = solve_phase_field(sh, HI_field, HII_field, ψp_field, ᾱ_field, V_s, U_s, dΩ)
                        if norm(get_free_dof_values(sh) - s_vals_old) / (norm(get_free_dof_values(sh)) + 1e-10) < stagger_tol_default
                            break
                        end
                    end
                end
                t_prev = t_val
            end
            stop_due_to_singular && break
        end
        stop_due_to_singular && break

        if phase_every > 0 && substep_counter % phase_every != 0
            for _ in 1:max_stagger_default
                s_vals_old = copy(get_free_dof_values(sh))
                HI_field = project_to_fe_space(HI_prev, V_L2, dΩ)
                HII_field = project_to_fe_space(HII_prev, V_L2, dΩ)
                ψp_array = [plastic_strain_energy(ps.εp_eq) for ps in plastic_states]
                ψp_field = array_to_fe_field(ψp_array, V_dg0, V_L2, dΩ)
                ᾱ_field = array_to_fe_field(ᾱ_array, V_dg0, V_L2, dΩ)
                sh = solve_phase_field(sh, HI_field, HII_field, ψp_field, ᾱ_field, V_s, U_s, dΩ)
                if norm(get_free_dof_values(sh) - s_vals_old) / (norm(get_free_dof_values(sh)) + 1e-10) < stagger_tol_default
                    break
                end
            end
        end

        current_cycle += actual_jump
        ᾱ_max = maximum(ᾱ_array)
        max_α = maximum(ps.εp_eq for ps in plastic_states)
        s_max = maximum(get_free_dof_values(sh))
        f_min = fatigue_degradation(ᾱ_max)
        @printf("  N=%d  αp_max=%.4f  s_max=%.4f  ᾱ=%.1f  f=%.4f\n",
                current_cycle, max_α, s_max, ᾱ_max, f_min)

        (x_tip, y_tip) = track_crack_tip(sh, Ω)
        crack_length = x_tip - V_TIP_X
        push!(cycle_history, current_cycle)
        push!(time_history, time() - run_t0)
        push!(crack_length_history, crack_length)
        push!(tip_x_history, x_tip)
        push!(smax_history, s_max)
        push!(alphap_history, max_α)
        push!(alphabar_history, ᾱ_max)
        push!(fatigue_history, f_min)
        @printf("  裂纹 a=%.3f mm  Δa=%.3f mm\n", x_tip, crack_length)

        HI_field = project_to_fe_space(HI_prev, V_L2, dΩ)
        HII_field = project_to_fe_space(HII_prev, V_L2, dΩ)
        ᾱ_field = array_to_fe_field(ᾱ_array, V_dg0, V_L2, dΩ)
        for i in eachindex(VTK_CRACK_FRACTIONS)
            if CLI.snapshots && !saved_vtu[i] && crack_length >= VTK_CRACK_FRACTIONS[i] * target_da
                fraction = VTK_CRACK_FRACTIONS[i]
                tag = @sprintf("milestone_da%02dpct", round(Int, 100 * fraction))
                save_vtu_state(
                    Ω, tag, uh, sh, HI_field, HII_field, ᾱ_field,
                    plastic_states, ψd_prev_array; output_dir=output_dir,
                    cycle=current_cycle, load=t_prev, crack_extension=crack_length,
                    crack_tip_x=x_tip, crack_tip_y=y_tip, milestone=fraction
                )
                saved_vtu[i] = true
            end
        end

        if x_tip >= stop_crack_x
            println("\n  裂纹达停止阈值，结束")
            break
        end
    end

    HI_field = project_to_fe_space(HI_prev, V_L2, dΩ)
    HII_field = project_to_fe_space(HII_prev, V_L2, dΩ)
    ᾱ_field = array_to_fe_field(ᾱ_array, V_dg0, V_L2, dΩ)
    if CLI.snapshots
        final_x, final_y = track_crack_tip(sh, Ω)
        save_vtu_state(
            Ω, "final", uh, sh, HI_field, HII_field, ᾱ_field,
            plastic_states, ψd_prev_array; output_dir=output_dir,
            cycle=current_cycle, load=t_prev, crack_extension=final_x - V_TIP_X,
            crack_tip_x=final_x, crack_tip_y=final_y, milestone=-1.0
        )
    end
    if CLI.visualization
        sh_qp = evaluate(sh, cell_points_cache)
        s_cell = [mean(sh_qp[i]) for i in eachindex(sh_qp)]
        save_cell_scalar_png(
            get_cell_coordinates(Ω), s_cell;
            output_file = joinpath(output_dir, "damage_final.png"),
            title_str = "CT Chaboche isothermal final damage",
            clims = (0.0, 1.0),
            cbar_title = "s"
        )
    end

    println("\n" * "="^80)
    println("模拟完成  N=$(current_cycle)")
    if !isempty(crack_length_history)
        println("  最终 Δa=$(round(crack_length_history[end], digits=3)) mm")
    end
    println("="^80)
    return cycle_history, time_history, crack_length_history, tip_x_history,
           smax_history, alphap_history, alphabar_history, fatigue_history
end

function plot_results(cycles, times, lengths, tip_x, smax_hist, alphap_hist, alphabar_hist, fatigue_hist; output_dir = OUTPUT_DIR)
    isempty(cycles) && (println("无数据可绘图"); return)
    if CLI.visualization
        p1 = plot(cycles, lengths, xlabel = "N (cycles)", ylabel = "Crack extension Δa (mm)",
                  title = "CT Chaboche isothermal", legend = false, linewidth = 2.5,
                  marker = :circle, markersize = 3, grid = true, size = (800, 600), dpi = 300)
        png_file = joinpath(output_dir, "crack_growth.png")
        savefig(p1, png_file)
        println("  图已保存: $(png_file)")
    end
    data_file = joinpath(output_dir, "cycle_summary.csv")
    open(data_file, "w") do io
        println(io, "cycle,time_s,delta_a_mm,tip_x_mm,s_max,alpha_p_max,alphabar_max,fatigue_f")
        writedlm(io, hcat(cycles, times, lengths, tip_x, smax_hist, alphap_hist, alphabar_hist, fatigue_hist), ',')
    end
    println("  数据已保存: $(data_file)")
end

function main()
    mkpath(OUTPUT_DIR)
    cycles, times, lengths, tip_x, smax_hist, alphap_hist, alphabar_hist, fatigue_hist = run_simulation()
    plot_results(cycles, times, lengths, tip_x, smax_hist, alphap_hist, alphabar_hist, fatigue_hist)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
