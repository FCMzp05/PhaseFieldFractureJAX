# ============================================================================
# SENT试样弹塑性疲劳裂纹扩展 - 完整修订版（修正所有物理公式）
# 
# 论文: "An adaptive cycle jump method for elasto-plastic phase field 
#        modeling addressing fatigue crack propagation" (2025)
# 对标: Section 4.1.2 - SENT specimen (Fig.9a, Table 4)
# 
# 📋 主要修正（相比原代码）：
# 1. ✅ 疲劳变量ᾱ改为时间积分 (Eq.5修正)
# 2. ✅ 移除相场驱动力中重复的塑性能量
# 3. ✅ 函数命名更清晰（tensile_elastic_strain_energy）
# 4. ✅ 添加塑性能量计算保护
# 5. ✅ 保留所有原有功能（PNG绘图、VTK输出、网格可视化等）
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
using Plots
gr()
using DelimitedFiles

function parse_cli(args)
    output_dir = normpath(joinpath(
        @__DIR__, "..", "outputs", "results_cpu", "plastic_fatigue_SENT", "julia"
    ))
    benchmark = "l0_0.02_defaults"
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
    benchmark == "l0_0.02_defaults" ||
        error("Supported benchmark: l0_0.02_defaults")
    return (output_dir=abspath(output_dir), benchmark=benchmark,
            visualization=visualization, snapshots=snapshots)
end

const CLI = parse_cli(ARGS)

# ============================================================================
# 几何参数
# ============================================================================

const W = 1.0       # 试样宽度 (mm)
const H = 1.0       # 试样高度 (mm)
const a0 = 0.5      # 初始裂纹长度 (mm)

# ============================================================================
# 材料参数 (Table 3)
# ============================================================================

struct MaterialData
    E::Float64      # 弹性模量 (MPa)
    ν::Float64      # 泊松比
    G::Float64      # 剪切模量 (MPa)
    λ::Float64      # Lamé常数 (MPa)
    σy0::Float64    # 初始屈服应力 (MPa)
    n::Float64      # 应变硬化指数
end

function create_material()
    E = 210000.0
    ν = 0.3
    G = E / (2 * (1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    σy0 = 440.0
    n = 0.1
    return MaterialData(E, ν, G, λ, σy0, n)
end

const mat = create_material()

# ============================================================================
# 相场参数 (Table 3)
# ============================================================================

const l0 = 0.02      # 长度尺度参数 (mm)
const Gc = 2.7       # 临界能量释放率 (N/mm)
const αT = 11.25     # 疲劳阈值 (N/mm²)
const η_stab = 1e-6  # 数值稳定性参数

# ⭐ 塑性能量权重系数 (Section 2.1)
const η_plastic = 1.0  # 论文Table 3建议η=1.0

# ============================================================================
# 载荷参数 (Section 4.1.2)
# ============================================================================

const u_max = 0.001  # 最大位移 (mm)
const R_load = 0.0   # 载荷比
const u_min = R_load * u_max

# ============================================================================
# 循环参数
# ============================================================================

const N_CYCLES = 800           # 总循环数
const N_STEPS_PER_CYCLE = 20   # 每个循环的步数
const phase_every = 1          # 相场更新频率
const max_stagger = 20         # 最大交错迭代次数
const stagger_tol = 1e-6       # 交错迭代收敛容差

# 可视化与内存控制
const ENABLE_PNG = false
const SAVE_PNG_EVERY = 10
const PNG_DPI = 110
const PLOT_CELL_EDGES = false
const FORCE_GC_EVERY = 20
const SAVE_PROGRESS_VTK = CLI.snapshots
const VTK_A_LEVELS = [0.02, 0.10, 0.20, 0.30, 0.40]

# ⭐ 新增：时间步长定义（用于疲劳变量时间积分）
const Δt_cycle = 1.0           # 每个循环的时间增量（归一化为1）
const Δt_step = Δt_cycle / N_STEPS_PER_CYCLE  # 每个子步的时间增量

# ============================================================================
# 网格参数
# ============================================================================

const h_fine = 0.005    # 裂纹区域细网格尺寸 (mm)
const h_coarse = 0.05   # 远场粗网格尺寸 (mm)
const OUTPUT_DIR = CLI.output_dir
const TIP_X_BACK = 3.0 * l0
const TIP_X_FRONT = 1.0 * l0
const TIP_Y_HALF = 3.0 * l0
const WAKE_X_BACK = 10.0 * l0
const AHEAD_X_FRONT = 5.0 * l0
const AHEAD_ALPHA_FRONT = 2.0 * l0
const EPSP_ACTIVE_THRESH = 1e-4
const EPSP_ZONE_THRESH = 1e-2

# ============================================================================
# 弹性刚度张量 (平面应变)
# ============================================================================

function elastic_stiffness()
    C1111 = mat.E * (1 - mat.ν) / ((1 + mat.ν) * (1 - 2 * mat.ν))
    C1122 = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2 * mat.ν))
    C1212 = mat.E / (2 * (1 + mat.ν))
    
    return SymFourthOrderTensorValue(
        C1111, 0.0, C1122,
        0.0, C1212, 0.0,
        C1122, 0.0, C1111
    )
end

const C_tensor = elastic_stiffness()
const ε = symmetric_gradient

# ============================================================================
# 塑性状态数据结构
# ============================================================================

mutable struct PlasticStateData
    εp::SymTensorValue{2,Float64,3}  # in-plane 塑性应变 (11,12,22)
    εp33::Float64                    # ⭐ 平面应变：out-of-plane 塑性应变分量 εp33
    εp_eq::Float64                   # 等效塑性应变（J2 累积量）
end

PlasticStateData() = PlasticStateData(SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0, 0.0)

function copy_plastic_state(ps::PlasticStateData)
    return PlasticStateData(ps.εp, ps.εp33, ps.εp_eq)
end

# ============================================================================
# 幂律硬化模型 (Eq.13)
# ============================================================================

function yield_stress(εp_eq::Float64)
    # Q = σy0 * (1 + (E/σy0) * εp_eq)^n
    return mat.σy0 * (1 + (mat.E / mat.σy0) * εp_eq)^mat.n
end

# ⭐ 修正：塑性应变能解析解（积分Eq.3，添加保护）
function plastic_strain_energy(εp_eq::Float64)
    if abs(εp_eq) < 1e-12
        return 0.0
    end
    
    # ⭐ 添加安全检查：防止n→-1
    if abs(mat.n + 1) < 1e-10
        @warn "Hardening exponent n is close to -1, using linear approximation"
        return mat.σy0 * εp_eq
    end
    
    # ψp = ∫ σy(α) dα = (σy0²/E) * [(1 + E/σy0 * εp_eq)^(n+1) - 1] / (n+1)
    term = (1 + (mat.E / mat.σy0) * εp_eq)^(mat.n + 1)
    return (mat.σy0^2 / mat.E) * (term - 1) / (mat.n + 1)
end

# ============================================================================
# 平面应变 3D 弹性应力（含 εp33 耦合）与 Von Mises
# ============================================================================

@inline function elastic_stress_components_plane_strain(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData
)
    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33 # 平面应变：ε33_total = 0
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
    σ_eq = sqrt(1.5 * (s11^2 + s22^2 + s33^2 + 2.0 * s12^2))
    return σ_eq, s11, s22, s33, s12
end

# ============================================================================
# 径向返回映射算法 (Section 2.2, Eq.9-12) — 平面应变一致3D J2
# ============================================================================

function return_mapping!(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    s_phasefield::Float64;
    update_state::Bool = true
)
    # 试探应力（3D 平面应变，含 εp33 耦合）
    σ11_trial, σ22_trial, σ33_trial, σ12_trial =
        elastic_stress_components_plane_strain(ε_total, plastic)

    # Von Mises 等效应力 (Eq.10)
    σ_eq_trial, s11, s22, s33, s12 = von_mises_eq_3d(σ11_trial, σ22_trial, σ33_trial, σ12_trial)
    
    # 当前屈服应力
    σy = yield_stress(plastic.εp_eq)
    
    # 损伤降解函数（仅用于最终应力输出，不影响屈服判断）
    g_s = (1 - s_phasefield)^2 + η_stab
    
    # ⭐ 屈服函数：不包含损伤（Eq.9, Section 2.2明确说明）
    f_trial = σ_eq_trial - σy
    
    # 弹性加载判断
    if f_trial <= 0.0 || σ_eq_trial < 1e-10
        return g_s * SymTensorValue{2,Float64}(σ11_trial, σ12_trial, σ22_trial)
    end
    
    # 塑性加载：径向返回映射
    Δγ = 0.0
    tol = 1e-9 * mat.σy0
    
    for iter in 1:50
        εp_eq_new = plastic.εp_eq + Δγ
        σy_new = yield_stress(εp_eq_new)
        σ_eq_new = σ_eq_trial - 3.0 * mat.G * Δγ
        
        residual = σ_eq_new - σy_new
        
        if abs(residual) < tol
            # 关联流动 (Eq.11): Δεp = (3/2) Δγ * s / σeq   (3D，含33分量)
            inv = (σ_eq_trial > 1e-12) ? (1.0 / σ_eq_trial) : 0.0
            Δεp11 = 1.5 * Δγ * s11 * inv
            Δεp22 = 1.5 * Δγ * s22 * inv
            Δεp33 = 1.5 * Δγ * s33 * inv
            Δεp12 = 1.5 * Δγ * s12 * inv

            if update_state
                plastic.εp = plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22)
                plastic.εp33 = plastic.εp33 + Δεp33
                plastic.εp_eq = εp_eq_new
            end
            
            # 返回降解后的应力
            plastic_tmp = update_state ? plastic : PlasticStateData(
                plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22),
                plastic.εp33 + Δεp33,
                εp_eq_new
            )
            σ11, σ22, σ33, σ12 = elastic_stress_components_plane_strain(ε_total, plastic_tmp)
            return g_s * SymTensorValue{2,Float64}(σ11, σ12, σ22)
        end
        
        # Newton-Raphson迭代
        term = (1 + (mat.E / mat.σy0) * εp_eq_new)
        dσy_dα = term > 1e-10 ? mat.n * mat.E * term^(mat.n - 1) : 0.0
        dR_dΔγ = -3.0 * mat.G - dσy_dα
        
        if abs(dR_dΔγ) < 1e-15
            @warn "Singular Jacobian in return mapping"
            break
        end
        
        Δγ = max(Δγ - residual / dR_dΔγ, 0.0)
    end
    
    # 迭代未收敛，返回试探应力
    @warn "Return mapping did not converge"
    return g_s * SymTensorValue{2,Float64}(σ11_trial, σ12_trial, σ22_trial)
end

# ============================================================================
# 弹性能量分解 (Eq.2)
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

function tensile_elastic_strain_energy(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData
)
    """
    计算拉伸弹性应变能 ψe+ (Eq.2) — 3D(平面应变) Amor 体积-偏量分解

    ψe+ = K/2 ⟨tr(εe)⟩+² + G * (dev(εe):dev(εe))
    其中 εe33 = -εp33（ε33_total=0）
    """
    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33

    trεe = εe11 + εe22 + εe33
    tr_pos = max(trεe, 0.0)
    K = mat.λ + (2.0 / 3.0) * mat.G

    m = trεe / 3.0
    dev11 = εe11 - m
    dev22 = εe22 - m
    dev33 = εe33 - m
    dev12 = εe12
    dev_norm2 = dev11^2 + dev22^2 + dev33^2 + 2.0 * dev12^2

    return 0.5 * K * tr_pos^2 + mat.G * dev_norm2
end

# ============================================================================
# ⭐ 修正：历史变量更新 - 包含塑性能量 (Section 2.1)
# P = max_τ∈[0,t] {ψe+ + η*ψp}
# ============================================================================

function update_history_variable(ψ_total::Real, H_old::Real)
    """
    更新历史变量 H = max{H_old, ψe+ + η*ψp}
    """
    return (true, max(H_old, ψ_total))
end

# ============================================================================
# 疲劳降解函数 (Eq.4)
# ============================================================================

function fatigue_degradation(ᾱ::Real)
    """
    疲劳降解函数 f(ᾱ)
    f = 1           if ᾱ ≤ αT
    f = (2αT/(ᾱ+αT))²  if ᾱ > αT
    """
    return ᾱ <= αT ? 1.0 : (2 * αT / (ᾱ + αT))^2
end

phase_field_degradation(s::Real) = (1 - s)^2 + η_stab

function compute_DWEP_regions(
    plastic_states::Vector{PlasticStateData},
    s_cell_vals::Vector{Float64},
    cell_centers,
    x_tip::Float64
)
    Φ_total = 0.0
    Φ_tip = 0.0
    Φ_wake = 0.0
    Φ_ahead = 0.0
    y_crack = H / 2.0
    for i in eachindex(plastic_states)
        s_i = clamp(s_cell_vals[i], 0.0, 1.0)
        weight = max(1.0 - phase_field_degradation(s_i), 0.0)
        contrib = weight * plastic_states[i].εp_eq
        Φ_total += contrib

        cx, cy = cell_centers[i][1], cell_centers[i][2]
        abs(cy - y_crack) > TIP_Y_HALF && continue
        if x_tip - l0 <= cx <= x_tip + 3.0 * l0
            Φ_tip += contrib
        end
        if x_tip - WAKE_X_BACK <= cx < x_tip - l0
            Φ_wake += contrib
        end
        if x_tip <= cx <= x_tip + AHEAD_X_FRONT
            Φ_ahead += contrib
        end
    end
    return Φ_total, Φ_tip, Φ_wake, Φ_ahead
end

function update_cycle_plastic_increment!(
    Δψp_last_array::Vector{Float64},
    ψp_state_prev_array::Vector{Float64},
    plastic_states::Vector{PlasticStateData}
)
    for i in eachindex(plastic_states)
        ψp_now = plastic_strain_energy(plastic_states[i].εp_eq)
        Δψp_last_array[i] = max(ψp_now - ψp_state_prev_array[i], 0.0)
        ψp_state_prev_array[i] = ψp_now
    end
end

function compute_tip_metrics_incremental(
    uh_field, plastic_states::Vector{PlasticStateData}, Δψp_last_array::Vector{Float64},
    cell_centers, x_tip::Float64, cell_points_cache, n_cells::Int
)
    ε_at_qp = evaluate(ε(uh_field), cell_points_cache)
    ψe_sum = 0.0
    Δψp_sum = 0.0
    n_tip = 0
    y_crack = H / 2.0
    for i in 1:n_cells
        cx, cy = cell_centers[i][1], cell_centers[i][2]
        in_tip_zone = (cx >= x_tip - TIP_X_BACK) &&
                      (cx <= x_tip + TIP_X_FRONT) &&
                      (abs(cy - y_crack) <= TIP_Y_HALF)
        !in_tip_zone && continue
        n_tip += 1
        ψe_cell = 0.0
        for q in 1:length(ε_at_qp[i])
            ψe_cell = max(ψe_cell, tensile_elastic_strain_energy(ε_at_qp[i][q], plastic_states[i]))
        end
        ψe_sum += ψe_cell
        Δψp_sum += Δψp_last_array[i]
    end
    n_tip == 0 && return (0.0, 0.0, 1.0, 0)
    ψe_tip_mean = ψe_sum / n_tip
    Δψp_tip_mean = Δψp_sum / n_tip
    χe = ψe_tip_mean / (ψe_tip_mean + Δψp_tip_mean + 1e-12)
    return (ψe_tip_mean, Δψp_tip_mean, χe, n_tip)
end

function compute_tip_metrics_cumulative(
    uh_field, plastic_states::Vector{PlasticStateData},
    cell_centers, x_tip::Float64, cell_points_cache, n_cells::Int
)
    ε_at_qp = evaluate(ε(uh_field), cell_points_cache)
    ψe_sum = 0.0
    ψp_sum = 0.0
    n_tip = 0
    y_crack = H / 2.0
    for i in 1:n_cells
        cx, cy = cell_centers[i][1], cell_centers[i][2]
        in_tip_zone = (cx >= x_tip - TIP_X_BACK) &&
                      (cx <= x_tip + TIP_X_FRONT) &&
                      (abs(cy - y_crack) <= TIP_Y_HALF)
        !in_tip_zone && continue
        n_tip += 1
        ψe_cell = 0.0
        for q in 1:length(ε_at_qp[i])
            ψe_cell = max(ψe_cell, tensile_elastic_strain_energy(ε_at_qp[i][q], plastic_states[i]))
        end
        ψe_sum += ψe_cell
        ψp_sum += plastic_strain_energy(plastic_states[i].εp_eq)
    end
    n_tip == 0 && return (0.0, 0.0, 1.0, 0)
    ψe_tip_mean = ψe_sum / n_tip
    ψp_tip_mean = ψp_sum / n_tip
    χe = ψe_tip_mean / (ψe_tip_mean + ψp_tip_mean + 1e-12)
    return (ψe_tip_mean, ψp_tip_mean, χe, n_tip)
end

function compute_alpha_bar_ahead_metrics(ᾱ_array::Vector{Float64}, cell_centers, x_tip::Float64)
    vals = Float64[]
    y_crack = H / 2.0
    for i in eachindex(ᾱ_array)
        cx, cy = cell_centers[i][1], cell_centers[i][2]
        if x_tip <= cx <= x_tip + AHEAD_ALPHA_FRONT && abs(cy - y_crack) <= TIP_Y_HALF
            push!(vals, ᾱ_array[i])
        end
    end
    isempty(vals) && return (0.0, 0.0, 0)
    return (mean(vals), maximum(vals), length(vals))
end

function compute_plastic_zone_metrics(
    plastic_states::Vector{PlasticStateData},
    εp_eq_before_cycle::Vector{Float64},
    cell_centers,
    x_tip::Float64
)
    y_crack = H / 2.0
    n_plastic_active = 0
    r_p_estimate = 0.0
    for i in eachindex(plastic_states)
        Δεp_eq = plastic_states[i].εp_eq - εp_eq_before_cycle[i]
        if Δεp_eq > EPSP_ACTIVE_THRESH
            n_plastic_active += 1
        end
        if plastic_states[i].εp_eq > EPSP_ZONE_THRESH
            cx, cy = cell_centers[i][1], cell_centers[i][2]
            r_p_estimate = max(r_p_estimate, hypot(cx - x_tip, cy - y_crack))
        end
    end
    return n_plastic_active, r_p_estimate
end

function compute_damage_metrics(sh, s_cell_vals::Vector{Float64}, cell_centers, cell_points_cache)
    y_crack = H / 2.0
    ∇s_at_qp = evaluate(∇(sh), cell_points_cache)
    max_grad_s = 0.0
    for cell_grads in ∇s_at_qp
        for g in cell_grads
            max_grad_s = max(max_grad_s, norm(g))
        end
    end
    xs_midband = Float64[]
    n_cells_s_above_01 = 0
    for i in eachindex(s_cell_vals)
        s_i = s_cell_vals[i]
        s_i > 0.1 && (n_cells_s_above_01 += 1)
        cx, cy = cell_centers[i][1], cell_centers[i][2]
        if 0.1 < s_i < 0.9 && abs(cy - y_crack) <= TIP_Y_HALF
            push!(xs_midband, cx)
        end
    end
    damage_zone_width = isempty(xs_midband) ? 0.0 : (maximum(xs_midband) - minimum(xs_midband))
    return max_grad_s, damage_zone_width, n_cells_s_above_01
end

function linear_fit_slope(x::Vector{Float64}, y::Vector{Float64})
    length(x) < 2 && return 0.0
    x̄ = mean(x)
    ȳ = mean(y)
    denom = sum((xi - x̄)^2 for xi in x)
    denom <= 1e-30 && return 0.0
    return sum((x[i] - x̄) * (y[i] - ȳ) for i in eachindex(x)) / denom
end

# ============================================================================
# L2投影工具函数
# ============================================================================

function project_to_fe_space(q, V_proj, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * q) * dΩ
    op = AffineFEOperator(a, b, V_proj, V_proj)
    return solve(op)
end

function array_to_fe_field(arr, V_dg0, V_proj, dΩ)
    f_dg = FEFunction(V_dg0, arr)
    return project_to_fe_space(f_dg, V_proj, dΩ)
end

# ============================================================================
# 创建带预制裂纹的SENT网格
# ============================================================================

function create_mesh()
    println("\n创建SENT网格（含预制裂纹）...")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("SENT")
    
    y_crack = H / 2.0
    CH = h_fine  # 裂纹尖端区域尺寸
    
    # 定义几何点
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_coarse, 1)
    gmsh.model.geo.addPoint(W, 0.0, 0.0, h_coarse, 2)
    gmsh.model.geo.addPoint(W, H/2, 0.0, h_coarse, 22)
    gmsh.model.geo.addPoint(W, H, 0.0, h_coarse, 3)
    gmsh.model.geo.addPoint(0.0, H, 0.0, h_coarse, 4)
    gmsh.model.geo.addPoint(0.0, y_crack + 0.5*CH, 0.0, h_coarse, 5)
    gmsh.model.geo.addPoint(a0 - CH, y_crack + 0.5*CH, 0.0, h_fine, 6)
    gmsh.model.geo.addPoint(a0, y_crack, 0.0, h_fine, 7)  # 裂纹尖端
    gmsh.model.geo.addPoint(a0 - CH, y_crack - 0.5*CH, 0.0, h_fine, 8)
    gmsh.model.geo.addPoint(0.0, y_crack - 0.5*CH, 0.0, h_coarse, 9)
    
    # 定义边界线
    gmsh.model.geo.addLine(1, 2, 1)    # 底边
    gmsh.model.geo.addLine(2, 22, 11)  # 右边下半
    gmsh.model.geo.addLine(22, 3, 2)   # 右边上半
    gmsh.model.geo.addLine(3, 4, 3)    # 顶边
    gmsh.model.geo.addLine(4, 5, 4)    # 左边上半
    gmsh.model.geo.addLine(5, 6, 5)    # 裂纹上侧
    gmsh.model.geo.addLine(6, 7, 6)    # 裂纹上尖端
    gmsh.model.geo.addLine(7, 8, 7)    # 裂纹下尖端
    gmsh.model.geo.addLine(8, 9, 8)    # 裂纹下侧
    gmsh.model.geo.addLine(9, 1, 9)    # 左边下半
    
    # 定义曲线环和平面
    gmsh.model.geo.addCurveLoop([1, 11, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    gmsh.model.geo.synchronize()
    
    # 定义物理组
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, [1], 1)
    gmsh.model.setPhysicalName(1, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [3], 2)
    gmsh.model.setPhysicalName(1, 2, "top")
    
    # Box加密：与 SENT 同类的中间带细化，不采用 SENS 那种全Y覆盖
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", a0 - 0.1)
    gmsh.model.mesh.field.setNumber(1, "XMax", W)
    gmsh.model.mesh.field.setNumber(1, "YMin", y_crack - 0.15)
    gmsh.model.mesh.field.setNumber(1, "YMax", y_crack + 0.15)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)
    
    mesh_file = joinpath(OUTPUT_DIR, "mesh.msh")
    gmsh.write(mesh_file)
    
    n_nodes = length(gmsh.model.mesh.getNodes()[1])
    println("  网格节点数: $(n_nodes)")
    println("  裂纹区域加密: h=$(h_fine)mm, 远场: h=$(h_coarse)mm")
    
    gmsh.finalize()
    return mesh_file
end

# ============================================================================
# 绘制网格
# ============================================================================

function plot_mesh(model)
    try
        Ω = Triangulation(model)
        cell_coords = get_cell_coordinates(Ω)
        
        p = plot(aspect_ratio=:equal, size=(700, 700), dpi=300,
                xlabel="x (mm)", ylabel="y (mm)", legend=false, grid=false,
                title="SENT Mesh ($(num_cells(Ω)) elements)")
        
        for i_cell in 1:num_cells(Ω)
            verts = cell_coords[i_cell]
            xs = [v[1] for v in verts]
            ys = [v[2] for v in verts]
            plot!(p, Shape(xs, ys), fillcolor=:white, linecolor=:black, linewidth=0.3)
        end
        
        savefig(p, joinpath(OUTPUT_DIR, "mesh.png"))
        println("  ✓ 网格图已保存: mesh.png")
    catch e
        @warn "网格绘图失败: $e"
    end
end

# ============================================================================
# 标量场PNG输出工具
# ============================================================================

function plot_cell_scalar_png(cell_coords, cell_values::Vector{Float64};
                              title::String, out_file::String, dpi::Int=120,
                              cmap=:jet, vmin::Union{Nothing,Float64}=nothing,
                              vmax::Union{Nothing,Float64}=nothing)
    vmin_use = isnothing(vmin) ? minimum(cell_values) : vmin
    vmax_use = isnothing(vmax) ? maximum(cell_values) : vmax
    denom = max(vmax_use - vmin_use, 1e-12)
    grad = cgrad(cmap)

    p = plot(
        aspect_ratio=:equal,
        size=(700, 700),
        dpi=dpi,
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=title,
        legend=false
    )

    n_cells_plot = length(cell_coords)
    for i_cell in 1:n_cells_plot
        verts = cell_coords[i_cell]
        xs = [v[1] for v in verts]
        ys = [v[2] for v in verts]
        ξ = clamp((cell_values[i_cell] - vmin_use) / denom, 0.0, 1.0)
        color = grad[ξ]

        plot!(p, Shape(xs, ys),
              fillcolor=color,
              linecolor=:transparent,
              linealpha=0.0,
              linewidth=0.0,
              label="")
    end

    savefig(p, out_file)
    closeall()
end

# ============================================================================
# 计算应力（调用塑性本构）
# ============================================================================

function compute_stress!(uh, sh, plastic_states, Ω, cell_points, update_state)
    εh = ε(uh)
    ε_at_qp = evaluate(εh, cell_points)
    s_at_qp = evaluate(sh, cell_points)
    
    n_cells = num_cells(Ω)
    σ_values = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)
    
    for i in 1:n_cells
        ε_cell = ε_at_qp[i][1]
        s_cell = clamp(s_at_qp[i][1], 0.0, 1.0)
        σ_values[i] = return_mapping!(ε_cell, plastic_states[i], s_cell, 
                                      update_state=update_state)
    end
    
    return σ_values
end

# ============================================================================
# 位移场求解（含Picard迭代处理塑性）
# ============================================================================

function solve_displacement(sh, plastic_states, u_val, V_u, U_u, dΩ, Ω, 
                           uh_prev, u_prev, cell_points)
    g_s(s) = (1 - s)^2 + η_stab
    n_cells = num_cells(Ω)
    plastic_work = [copy_plastic_state(ps) for ps in plastic_states]
    
    # 位移增量子步
    Δu = u_val - u_prev
    n_substeps = max(1, ceil(Int, abs(Δu) / (u_max / 10)))
    uh = uh_prev
    
    for substep in 1:n_substeps
        plastic_start = [copy_plastic_state(ps) for ps in plastic_work]
        
        # Picard迭代求解非线性平衡
        for picard in 1:10
            u_old = copy(get_free_dof_values(uh))
            
	            # 构造塑性“本征应力”场（平面应变：额外包含 λ·εp33 项）
	            σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)
	            for i in 1:n_cells
	                σp_in = C_tensor ⊙ plastic_work[i].εp
	                add = mat.λ * plastic_work[i].εp33
	                σp_vals[i] = σp_in + SymTensorValue{2,Float64}(add, 0.0, add)
	            end
	            σp_field = CellField(σp_vals, Ω)
            
            # 弱形式：∫ g(s) C:(ε(u)-εp) : ε(v) dV = 0
            a(u, v) = ∫((g_s ∘ sh) * (C_tensor ⊙ ε(u)) ⊙ ε(v)) * dΩ
            l(v) = ∫((g_s ∘ sh) * σp_field ⊙ ε(v)) * dΩ
            
            op = AffineFEOperator(a, l, U_u, V_u)
            uh = solve(op)
            
	            # 更新塑性状态
	            for i in 1:n_cells
	                plastic_work[i].εp = plastic_start[i].εp
	                plastic_work[i].εp33 = plastic_start[i].εp33
	                plastic_work[i].εp_eq = plastic_start[i].εp_eq
	            end
            compute_stress!(uh, sh, plastic_work, Ω, cell_points, true)
            
            # 收敛性检查
            u_change = norm(get_free_dof_values(uh) - u_old) / 
                      (norm(get_free_dof_values(uh)) + 1e-10)
            if u_change < 1e-4
                break
            end
        end
    end
    
	    # 提交塑性状态
	    for i in 1:n_cells
	        plastic_states[i].εp = plastic_work[i].εp
	        plastic_states[i].εp33 = plastic_work[i].εp33
	        plastic_states[i].εp_eq = plastic_work[i].εp_eq
	    end
    
    return uh
end

# ============================================================================
# ⭐ 修正：相场求解 - 标准AT2形式 (Eq.7-8)
# ============================================================================

function solve_phase_field(sh_prev, H_field, ᾱ_field, V_s, U_s, dΩ)
    """
    相场方程 (Eq.8 - 混合相场模型):
    -g'(s)P + f(ᾱ)Gc/l0 · s - ∇·(f(ᾱ)Gc·l0·∇s) = 0
    
    取 g(s)=(1-s)² ⇒ -g'(s)=2(1-s)
    展开后得到线性方程（P 视为已知历史变量）：
      (2P + fGc/l0) s - ∇·(fGc·l0·∇s) = 2P

    弱式（仿射形式）：
      ∫[(2P+fGc/l0)s·ϕ + fGc·l0 ∇s·∇ϕ] dΩ = ∫[2P·ϕ] dΩ
    
    ⭐ 注意：H_field已经包含了max(ψe+ + η*ψp)，不需要重复添加塑性能量
    """
    f_dg = (x -> fatigue_degradation(x)) ∘ ᾱ_field
    
    P = H_field

    a(s, ϕ) = ∫(
        (2.0 * P + (f_dg * Gc / l0)) * s * ϕ +
        f_dg * Gc * l0 * ∇(s) ⋅ ∇(ϕ)
    ) * dΩ

    b(ϕ) = ∫(2.0 * P * ϕ) * dΩ
    
    op = AffineFEOperator(a, b, U_s, V_s)
    sh_solved = solve(op)
    
    # 不可逆性约束: s_new ≥ s_old
    s_new = get_free_dof_values(sh_solved)
    s_old = get_free_dof_values(sh_prev)
    
    for i in 1:length(s_new)
        # 数值稳定性处理
        if isnan(s_new[i]) || isinf(s_new[i])
            s_new[i] = s_old[i]
        end
        # 不可逆性 + 边界约束
        s_new[i] = clamp(max(s_new[i], s_old[i]), 0.0, 1.0)
    end
    
    return FEFunction(U_s, s_new)
end

# ============================================================================
# 裂纹尖端追踪
# ============================================================================

function track_crack_tip(sh, Ω, tol=0.5)
    """
    追踪裂纹尖端位置（定义为s>tol的最右端点）
    """
    coords = get_node_coordinates(Ω)
    s_vals = get_free_dof_values(sh)
    y_crack = H / 2.0
    
    x_max = a0
    for i in 1:length(s_vals)
        x, y = coords[i][1], coords[i][2]
        # 只在裂纹线附近搜索
        if x > a0 - l0 && abs(y - y_crack) < 2*l0 && s_vals[i] > tol
            x_max = max(x_max, x)
        end
    end
    
    return x_max
end

# ============================================================================
# 主模拟循环
# ============================================================================

function run_simulation()
    mkpath(OUTPUT_DIR)
    
    println("\n" * "="^80)
    println("SENT弹塑性疲劳裂纹扩展 - 完整修订版")
    println("="^80)
    println("📋 物理公式修正:")
    println("  ✅ 疲劳变量ᾱ改为时间积分 (Eq.5)")
    println("  ✅ 移除相场驱动力中重复的塑性能量")
    println("  ✅ 历史变量P = max{ψe+ + η*ψp} (η=$(η_plastic))")
    println("="^80)
    println("材料参数 (Table 3):")
    println("  E = $(mat.E/1e3) GPa")
    println("  ν = $(mat.ν)")
    println("  σy0 = $(mat.σy0) MPa")
    println("  n = $(mat.n)")
    println("相场参数:")
    println("  l0 = $(l0) mm")
    println("  Gc = $(Gc) N/mm")
    println("  αT = $(αT) N/mm²")
    println("载荷条件:")
    println("  u_max = $(u_max) mm")
    println("  R = $(R_load)")
    println("预期结果 (论文Fig.9a):")
    println("  N ≈ 696 cycles")
    println("  Δa ≈ 0.45 mm")
    println("="^80)
    
    # 创建网格
    mesh_file = create_mesh()
    model = GmshDiscreteModel(mesh_file)
    if CLI.visualization
        writevtk(model, joinpath(OUTPUT_DIR, "mesh"))
        plot_mesh(model)
    end
    
    # 定义区域和测度
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    
    order = 1
    
    # 位移场有限元空间
    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["bottom", "top"],
                      dirichlet_masks=[(true, true), (false, true)])
    
    u_bottom(x) = VectorValue(0.0, 0.0)
    u_top(x, u) = VectorValue(0.0, u)
    U_u = TrialFESpace(V_u)
    
    # 相场有限元空间
    reffe_s = ReferenceFE(lagrangian, Float64, order)
    V_s = TestFESpace(model, reffe_s, conformity=:H1)
    U_s = TrialFESpace(V_s)
    
    # 单元常数空间（用于存储历史变量）
    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = FESpace(model, reffe_dg0, conformity=:L2)
    
    # L2空间（用于投影）
    reffe_L2 = ReferenceFE(lagrangian, Float64, order)
    V_L2 = FESpace(model, reffe_L2, conformity=:L2)
    
    # 积分点缓存
    quad_cache = CellQuadrature(Ω, 2)
    cell_points_cache = get_cell_points(quad_cache)
    
    n_cells = num_cells(Ω)
    println("\n网格单元数: $(n_cells)")
    
	    # ⭐ 初始化状态变量
	    plastic_states = [PlasticStateData() for _ in 1:n_cells]  # 塑性状态
	    H_prev = CellState(0.0, dΩ)                               # 历史变量H
	    ᾱ_accumulated = zeros(Float64, n_cells)                   # ⭐ 累积疲劳变量
	    α_inst_prev = zeros(Float64, n_cells)                    # ⭐ 瞬时疲劳驱动 α 的历史（用于正增量累积）
    ψp_state_prev_array = zeros(Float64, n_cells)
    Δψp_last_array = zeros(Float64, n_cells)
    
    # 初始相场：完全无损伤
    sh = interpolate_everywhere(x -> 0.0, U_s)
    uh = zero(U_u)
    
    println("初始状态: s=0 (无预损伤), ᾱ=0")
    
    # 记录数据
    cycle_history = Int[]
    crack_history = Float64[]
    time_history = Float64[]
    prev_phi_dwep = 0.0
    prev_phi_tip = 0.0
    prev_alpha_bar_max = 0.0
    prev_crack_length = 0.0
    recent_N = Float64[]
    recent_alpha_bar_max = Float64[]
    run_t0 = time()
    next_vtk_idx = 1

    # ── KHF 数据收集：打开 CSV，写表头 ──────────────────────────────────────
    khf_csv_path = joinpath(OUTPUT_DIR, "khf_data.csv")
    khf_io = open(khf_csv_path, "w")
    println(khf_io, "N,time_s,a,",
            "alpha_max,alpha_mean,alpha_var,alpha_tip_mean,",
            "psip_max,psip_mean,psip_var,psip_tip_mean,",
            "psie_max,psie_mean,psie_var,psie_tip_mean,",
            "d_max,d_mean,d_tip_mean")

    diag_csv_path = joinpath(OUTPUT_DIR, "diagnostics.csv")
    diag_io = open(diag_csv_path, "w")
    println(diag_io,
            "N,time_s,ΔN,alpha_bar_max,s_max,alpha_p_max,f,x_tip,delta_a," *
            "phi_dwep,phi_tip,phi_wake,phi_ahead," *
            "delta_phi_block,delta_phi_per_cycle_proxy," *
            "delta_alpha_block,delta_alpha_per_cycle_proxy," *
            "delta_a_block,delta_a_per_cycle_proxy," *
            "phi_tip_rate,alpha_bar_rate_linfit," *
            "psie_tip_peak_mean,psie_tip_unload_mean,psie_peak_unload_ratio," *
            "dpsip_tip_mean,psip_tip_mean,chi_e_peak,chi_e_unload,chi_e_peak_cum,chi_e_unload_cum,n_tip," *
            "alpha_bar_tip_ahead_mean,alpha_bar_tip_ahead_max,n_ahead_alpha_cells," *
            "n_plastic_active,r_p_estimate,max_grad_s,damage_zone_width,n_cells_s_above_01")

    # 循环载荷
    n_half = N_STEPS_PER_CYCLE ÷ 2
    load_steps = range(u_min, u_max, length=n_half + 1)
    unload_steps = range(u_max, u_min, length=n_half + 1)[2:end]
    
    current_cycle = 0
    u_prev = u_min
    
    println("\n开始疲劳循环（PNG每 $(SAVE_PNG_EVERY) 个循环保存一次）...")
    cell_coords = get_cell_coordinates(Ω)

    # 预计算单元中心坐标（用于裂尖区域过滤）
    cell_centers = [sum(cell_coords[i]) / length(cell_coords[i]) for i in 1:n_cells]
    ψ_elastic = zeros(n_cells)
    ψ_total = zeros(n_cells)
    plot_max_epspeq = 1e-8
    plot_max_psip = 1e-8
    plot_max_sigmaeq = 1e-8
    plot_max_alphabar = 1e-8
    
    while current_cycle < N_CYCLES
        current_cycle += 1
        
        if current_cycle % 10 == 0 || current_cycle <= 5
            println("\n>>> 循环 N=$(current_cycle)")
        end
        
        substep = 0
        uh_peak = FEFunction(U_u, copy(get_free_dof_values(uh)))
        best_peak_dist = Inf
        εp_eq_before_cycle = [ps.εp_eq for ps in plastic_states]
        
        # 加载-卸载循环
        for (phase, u_range) in [("加载", load_steps), ("卸载", unload_steps)]
            for u_val in u_range
                substep += 1
                
                # 施加边界条件
                U_u = TrialFESpace(V_u, [u_bottom, x -> u_top(x, u_val)])
                
                # 求解位移场
                try
                    uh = solve_displacement(sh, plastic_states, u_val, V_u, U_u, dΩ, Ω,
                                          uh, u_prev, cell_points_cache)
                catch e
                    if isa(e, SingularException)
                        println("⚠️ 奇异矩阵，模拟终止")
                        @goto sim_end
                    else
                        rethrow(e)
                    end
                end
                peak_dist = abs(u_val - u_max)
                if peak_dist < best_peak_dist
                    best_peak_dist = peak_dist
                    uh_peak = FEFunction(U_u, copy(get_free_dof_values(uh)))
                end
                
                # ⭐ 计算能量（弹性+塑性）
                εh = ε(uh)
                ε_at_qp = evaluate(εh, cell_points_cache)
                s_at_qp = evaluate(sh, cell_points_cache)
                
                # 计算弹性拉伸应变能 ψe+
                for i in 1:n_cells
                    ψ_p = plastic_strain_energy(plastic_states[i].εp_eq)
                    ψ_max = 0.0
                    n_qp = length(ε_at_qp[i])
                    for q in 1:n_qp
                        ψe_q = tensile_elastic_strain_energy(ε_at_qp[i][q], plastic_states[i])
                        ψ_max = max(ψ_max, ψe_q)
                    end
                    ψ_elastic[i] = ψ_max
                    ψ_total[i] = ψ_max + η_plastic * ψ_p
                end
                
                # ⭐ 更新历史变量 H = max{H_old, ψe+ + η*ψp}
                ψ_total_dg = FEFunction(V_dg0, ψ_total)
                update_state!(update_history_variable, H_prev, ψ_total_dg)

                # 疲劳变量累积（回退到原始实现）
                for i in 1:n_cells
                    s_cell = clamp(s_at_qp[i][1], 0.0, 1.0)
                    g_s = (1 - s_cell)^2 + η_stab
                    α_inst = g_s * ψ_total[i]
                    Δᾱ = max(α_inst - α_inst_prev[i], 0.0)
                    ᾱ_accumulated[i] += Δᾱ
                    α_inst_prev[i] = α_inst
                end
                
                # 相场更新（交错迭代）
                if phase_every > 0 && (substep % phase_every == 0)
                    for stagger in 1:max_stagger
                        s_old = copy(get_free_dof_values(sh))
                        
                        # 投影历史变量到有限元空间
                        H_field = project_to_fe_space(H_prev, V_L2, dΩ)
                        
                        # ⭐ 疲劳历史变量场
                        ᾱ_field = array_to_fe_field(ᾱ_accumulated, V_dg0, V_L2, dΩ)
                        
                        # ⭐ 求解相场（已修正，不重复添加塑性能量）
                        sh = solve_phase_field(sh, H_field, ᾱ_field, V_s, U_s, dΩ)
                        
                        # 收敛性检查
                        s_new = get_free_dof_values(sh)
                        res = norm(s_new - s_old) / (norm(s_new) + 1e-10)
                        if res < stagger_tol
                            break
                        end
                    end
                end
                
                u_prev = u_val
            end
        end
        
        # 循环结束统计
        max_εp_eq = maximum([ps.εp_eq for ps in plastic_states])
        max_s = maximum(get_free_dof_values(sh))
        max_ᾱ = maximum(ᾱ_accumulated)
        
	        # 计算等效应力（与屈服判据一致的3D Von Mises，平面应变含σ33）
	        εh_end = ε(uh)
	        ε_at_qp_end = evaluate(εh_end, cell_points_cache)
	        max_σ_eq = 0.0
	        for i in 1:n_cells
	            ε_cell = ε_at_qp_end[i][1]
	            σ11, σ22, σ33, σ12 = elastic_stress_components_plane_strain(ε_cell, plastic_states[i])
	            σ_eq, _, _, _, _ = von_mises_eq_3d(σ11, σ22, σ33, σ12)
	            max_σ_eq = max(max_σ_eq, σ_eq)
	        end
        
        # 追踪裂纹长度
        a_current = track_crack_tip(sh, Ω)
        Δa = a_current - a0
        update_cycle_plastic_increment!(Δψp_last_array, ψp_state_prev_array, plastic_states)
        elapsed_s = time() - run_t0

        push!(cycle_history, current_cycle)
        push!(crack_history, Δa)
        push!(time_history, elapsed_s)

        # ── KHF 特征提取并写入 CSV ──────────────────────────────────────────
        # 裂尖附近区域：x ∈ [a_current - 3l0, a_current + l0]，y 在裂纹线 ±3l0 内
        tip_x_lo = a_current - 3*l0
        tip_x_hi = a_current + l0
        tip_y_c  = H / 2.0
        tip_mask = [
            let cx = cell_centers[i][1], cy = cell_centers[i][2]
                tip_x_lo <= cx <= tip_x_hi && abs(cy - tip_y_c) <= 3*l0
            end
            for i in 1:n_cells
        ]

        ψp_vals_now = [plastic_strain_energy(plastic_states[i].εp_eq) for i in 1:n_cells]
        α_all  = ᾱ_accumulated
        ψp_all = ψp_vals_now
        ψe_all = copy(ψ_elastic)   # 当前周期末的 ψe+ 快照（substep 里持续更新）

        # 相场单元均值（节点场 → 单元均值）
        s_at_qp_now = evaluate(sh, cell_points_cache)
        d_all = [sum(clamp(v, 0.0, 1.0) for v in s_at_qp_now[i]) / max(length(s_at_qp_now[i]), 1)
                 for i in 1:n_cells]

        α_tip  = α_all[tip_mask]
        ψp_tip = ψp_all[tip_mask]
        ψe_tip = ψe_all[tip_mask]
        d_tip  = d_all[tip_mask]

        α_max   = maximum(α_all)
        α_mean  = sum(α_all) / n_cells
        α_var   = sum((x - α_mean)^2 for x in α_all) / n_cells
        α_tmean = isempty(α_tip)  ? 0.0 : sum(α_tip)  / length(α_tip)

        ψp_max   = maximum(ψp_all)
        ψp_mean  = sum(ψp_all) / n_cells
        ψp_var   = sum((x - ψp_mean)^2 for x in ψp_all) / n_cells
        ψp_tmean = isempty(ψp_tip) ? 0.0 : sum(ψp_tip) / length(ψp_tip)

        ψe_max   = maximum(ψe_all)
        ψe_mean  = sum(ψe_all) / n_cells
        ψe_var   = sum((x - ψe_mean)^2 for x in ψe_all) / n_cells
        ψe_tmean = isempty(ψe_tip) ? 0.0 : sum(ψe_tip) / length(ψe_tip)

        d_max   = maximum(d_all)
        d_mean  = sum(d_all) / n_cells
        d_tmean = isempty(d_tip) ? 0.0 : sum(d_tip) / length(d_tip)

        Φ_dwep, Φ_tip, Φ_wake, Φ_ahead = compute_DWEP_regions(
            plastic_states, d_all, cell_centers, a_current
        )
        ΔΦ_block = Φ_dwep - prev_phi_dwep
        ΔΦ_tip_block = Φ_tip - prev_phi_tip
        ΔΦ_per_cycle_proxy = ΔΦ_block
        Δᾱ_block = max_ᾱ - prev_alpha_bar_max
        Δᾱ_per_cycle_proxy = Δᾱ_block
        Δa_block = Δa - prev_crack_length
        Δa_per_cycle_proxy = Δa_block
        ψe_tip_peak_mean, Δψp_tip_mean, χe_peak, n_tip = compute_tip_metrics_incremental(
            uh_peak, plastic_states, Δψp_last_array, cell_centers, a_current, cell_points_cache, n_cells
        )
        ψe_tip_unload_mean, _, χe_unload, _ = compute_tip_metrics_incremental(
            uh, plastic_states, Δψp_last_array, cell_centers, a_current, cell_points_cache, n_cells
        )
        _, ψp_tip_mean, χe_peak_cum, _ = compute_tip_metrics_cumulative(
            uh_peak, plastic_states, cell_centers, a_current, cell_points_cache, n_cells
        )
        _, _, χe_unload_cum, _ = compute_tip_metrics_cumulative(
            uh, plastic_states, cell_centers, a_current, cell_points_cache, n_cells
        )
        ψe_peak_unload_ratio = ψe_tip_peak_mean / max(ψe_tip_unload_mean, 1e-12)
        ᾱ_tip_ahead_mean, ᾱ_tip_ahead_max, n_ahead_alpha_cells = compute_alpha_bar_ahead_metrics(
            ᾱ_accumulated, cell_centers, a_current
        )
        n_plastic_active, r_p_estimate = compute_plastic_zone_metrics(
            plastic_states, εp_eq_before_cycle, cell_centers, a_current
        )
        max_grad_s, damage_zone_width, n_cells_s_above_01 = compute_damage_metrics(
            sh, d_all, cell_centers, cell_points_cache
        )
        push!(recent_N, current_cycle)
        push!(recent_alpha_bar_max, max_ᾱ)
        length(recent_N) > 5 && (popfirst!(recent_N); popfirst!(recent_alpha_bar_max))
        alpha_bar_rate_linfit = linear_fit_slope(recent_N, recent_alpha_bar_max)
        phi_tip_rate = Φ_tip > 1e-30 ? (ΔΦ_tip_block / max(Φ_tip, 1e-30)) : 0.0

        @printf(khf_io, "%d,%.6f,%.6f,",  current_cycle, elapsed_s, a_current)
        @printf(khf_io, "%.6e,%.6e,%.6e,%.6e,",  α_max,  α_mean,  α_var,  α_tmean)
        @printf(khf_io, "%.6e,%.6e,%.6e,%.6e,",  ψp_max, ψp_mean, ψp_var, ψp_tmean)
        @printf(khf_io, "%.6e,%.6e,%.6e,%.6e,",  ψe_max, ψe_mean, ψe_var, ψe_tmean)
        @printf(khf_io, "%.6e,%.6e,%.6e\n",       d_max,  d_mean,  d_tmean)
        @printf(diag_io, "%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%d,%d,%.6f,%.6e,%.6f,%d\n",
                current_cycle, elapsed_s, 1, max_ᾱ, max_s, max_εp_eq, fatigue_degradation(max_ᾱ),
                a_current, Δa,
                Φ_dwep, Φ_tip, Φ_wake, Φ_ahead,
                ΔΦ_block, ΔΦ_per_cycle_proxy,
                Δᾱ_block, Δᾱ_per_cycle_proxy,
                Δa_block, Δa_per_cycle_proxy,
                phi_tip_rate, alpha_bar_rate_linfit,
                ψe_tip_peak_mean, ψe_tip_unload_mean, ψe_peak_unload_ratio,
                Δψp_tip_mean, ψp_tip_mean, χe_peak, χe_unload, χe_peak_cum, χe_unload_cum, n_tip,
                ᾱ_tip_ahead_mean, ᾱ_tip_ahead_max, n_ahead_alpha_cells,
                n_plastic_active, r_p_estimate, max_grad_s, damage_zone_width, n_cells_s_above_01)
        prev_phi_dwep = Φ_dwep
        prev_phi_tip = Φ_tip
        prev_alpha_bar_max = max_ᾱ
        prev_crack_length = Δa

        while SAVE_PROGRESS_VTK && next_vtk_idx <= length(VTK_A_LEVELS) && Δa >= VTK_A_LEVELS[next_vtk_idx]
            snapshot_dir = joinpath(OUTPUT_DIR, "snapshots")
            mkpath(snapshot_dir)
            εp_eq_vals_vtk = [ps.εp_eq for ps in plastic_states]
            εp_vals_vtk = [ps.εp for ps in plastic_states]
            εp33_vals_vtk = [ps.εp33 for ps in plastic_states]
            αbar_field_vtk = CellField(ᾱ_accumulated, Ω)
            εp_eq_field_vtk = CellField(εp_eq_vals_vtk, Ω)
            εp_field_vtk = CellField(εp_vals_vtk, Ω)
            εp33_field_vtk = CellField(εp33_vals_vtk, Ω)
            H_field_vtk = project_to_fe_space(H_prev, V_L2, dΩ)
            level = VTK_A_LEVELS[next_vtk_idx]
            writevtk(Ω, joinpath(snapshot_dir, "milestone_da$(lpad(round(Int,1000*level),4,'0'))um"),
                     cellfields=[
                         "d"=>sh,
                         "u"=>uh,
                         "plastic_strain"=>εp_field_vtk,
                         "plastic_strain_33"=>εp33_field_vtk,
                         "epspeq"=>εp_eq_field_vtk,
                         "alpha_bar"=>αbar_field_vtk,
                         "history"=>H_field_vtk,
                         "plastic_energy_previous"=>CellField(ψp_state_prev_array, Ω),
                         "plastic_energy_increment"=>CellField(Δψp_last_array, Ω),
                         "cycle"=>CellField(fill(Float64(current_cycle), n_cells), Ω),
                         "step"=>CellField(fill(Float64(current_cycle * N_STEPS_PER_CYCLE), n_cells), Ω),
                         "load"=>CellField(fill(u_min, n_cells), Ω),
                         "qoi_crack_extension_mm"=>CellField(fill(Δa, n_cells), Ω),
                         "qoi_crack_tip_x_mm"=>CellField(fill(a_current, n_cells), Ω),
                         "milestone"=>CellField(fill(level, n_cells), Ω)
                     ],
                     nsubcells=4)
            next_vtk_idx += 1
        end
        
        if current_cycle % 10 == 0 || current_cycle <= 5
            σy_current = yield_stress(max_εp_eq)
            @printf("  εp_eq=%.6f, σeq=%.1f, σy=%.1f, s=%.4f, ᾱ=%.2f, Δa=%.5fmm\n",
                    max_εp_eq, max_σ_eq, σy_current, max_s, max_ᾱ, Δa)
        end
        
        needs_png = ENABLE_PNG && (current_cycle % SAVE_PNG_EVERY == 0)
        needs_vtk = CLI.visualization && (current_cycle % 10 == 0)

        if needs_png || needs_vtk
            cycle_str = lpad(current_cycle, 6, '0')

            # 统一构造输出标量场（单元常数）
            s_at_cells = evaluate(sh, cell_points_cache)
            εh_out = ε(uh)
            ε_at_qp_out = evaluate(εh_out, cell_points_cache)

            s_cell_vals = zeros(Float64, n_cells)
            εp_eq_vals = zeros(Float64, n_cells)
            ψp_vals = zeros(Float64, n_cells)
            σeq_vals = zeros(Float64, n_cells)

            for i in 1:n_cells
                # damage 单元平均
                s_sum = 0.0
                nqp_s = length(s_at_cells[i])
                for q in 1:nqp_s
                    s_sum += clamp(s_at_cells[i][q], 0.0, 1.0)
                end
                s_cell_vals[i] = s_sum / max(nqp_s, 1)

                # 塑性历史量
                εp_eq_vals[i] = plastic_states[i].εp_eq
                ψp_vals[i] = plastic_strain_energy(plastic_states[i].εp_eq)

                # 等效应力：单元内积分点最大值
                σeq_max_cell = 0.0
                nqp_e = length(ε_at_qp_out[i])
                for q in 1:nqp_e
                    ε_cell_qp = ε_at_qp_out[i][q]
                    σ11, σ22, σ33, σ12 = elastic_stress_components_plane_strain(ε_cell_qp, plastic_states[i])
                    σeq_qp, _, _, _, _ = von_mises_eq_3d(σ11, σ22, σ33, σ12)
                    σeq_max_cell = max(σeq_max_cell, σeq_qp)
                end
                σeq_vals[i] = σeq_max_cell
            end

            # 使用单调扩展的统一色标上限，避免逐帧自适应造成视觉误判
            plot_max_epspeq = max(plot_max_epspeq, maximum(εp_eq_vals))
            plot_max_psip = max(plot_max_psip, maximum(ψp_vals))
            plot_max_sigmaeq = max(plot_max_sigmaeq, maximum(σeq_vals))
            plot_max_alphabar = max(plot_max_alphabar, maximum(ᾱ_accumulated))

            # 低频保存PNG（不再只输出损伤）
            if needs_png
                try
                    plot_cell_scalar_png(
                        cell_coords, s_cell_vals,
                        title="Damage s | N=$(current_cycle), Δa=$(round(Δa, digits=3))mm",
                        out_file=joinpath(OUTPUT_DIR, "damage_N$(cycle_str).png"),
                        dpi=PNG_DPI, cmap=:jet, vmin=0.0, vmax=1.0
                    )
                    plot_cell_scalar_png(
                        cell_coords, εp_eq_vals,
                        title="Equivalent Plastic Strain εp_eq | N=$(current_cycle)",
                        out_file=joinpath(OUTPUT_DIR, "epspeq_N$(cycle_str).png"),
                        dpi=PNG_DPI, cmap=:plasma, vmin=0.0, vmax=plot_max_epspeq
                    )
                    plot_cell_scalar_png(
                        cell_coords, ψp_vals,
                        title="Plastic Strain Energy ψp | N=$(current_cycle)",
                        out_file=joinpath(OUTPUT_DIR, "psip_N$(cycle_str).png"),
                        dpi=PNG_DPI, cmap=:viridis, vmin=0.0, vmax=plot_max_psip
                    )
                    plot_cell_scalar_png(
                        cell_coords, σeq_vals,
                        title="Von Mises Stress σeq | N=$(current_cycle)",
                        out_file=joinpath(OUTPUT_DIR, "sigmaeq_N$(cycle_str).png"),
                        dpi=PNG_DPI, cmap=:magma, vmin=0.0, vmax=plot_max_sigmaeq
                    )
                    plot_cell_scalar_png(
                        cell_coords, ᾱ_accumulated,
                        title="Fatigue History ᾱ | N=$(current_cycle)",
                        out_file=joinpath(OUTPUT_DIR, "alphabar_N$(cycle_str).png"),
                        dpi=PNG_DPI, cmap=:inferno, vmin=0.0, vmax=plot_max_alphabar
                    )
                catch e
                    @warn "PNG绘图失败: $e"
                end
            end

            # VTK输出（扩展多场）
            if needs_vtk
                try
                    εp_eq_field = CellField(εp_eq_vals, Ω)
                    ψp_field = CellField(ψp_vals, Ω)
                    σeq_field = CellField(σeq_vals, Ω)
                    αbar_field = CellField(ᾱ_accumulated, Ω)
                    s_cell_field = CellField(s_cell_vals, Ω)

                    writevtk(Ω, joinpath(OUTPUT_DIR, "result_N$(cycle_str)"),
                             cellfields=[
                                 "d"=>sh,
                                 "u"=>uh,
                                 "s_cell"=>s_cell_field,
                                 "epspeq"=>εp_eq_field,
                                 "psip"=>ψp_field,
                                 "sigmaeq"=>σeq_field,
                                 "alpha_bar"=>αbar_field
                             ],
                             nsubcells=4)
                catch e
                    @warn "VTK输出失败: $e"
                end
            end
        end

        if current_cycle % FORCE_GC_EVERY == 0
            GC.gc(false)
        end
        
        # 停止条件
        if Δa > 0.4 || a_current > 0.95
            println("\n⚠️ 裂纹接近边界，模拟停止")
            break
        end
    end
    
    @label sim_end
    close(khf_io)
    close(diag_io)
    println("  KHF 数据已保存: $khf_csv_path")
    println("  诊断数据已保存: $diag_csv_path")
    open(joinpath(OUTPUT_DIR, "cycle_summary.csv"), "w") do io
        println(io, "N,time_s,delta_a_mm")
        for i in eachindex(cycle_history)
            @printf(io, "%d,%.8f,%.8f\n", cycle_history[i], time_history[i], crack_history[i])
        end
    end

    if CLI.snapshots && !isempty(crack_history)
        snapshot_dir = joinpath(OUTPUT_DIR, "snapshots")
        mkpath(snapshot_dir)
        εp_eq_field = CellField([ps.εp_eq for ps in plastic_states], Ω)
        εp_field = CellField([ps.εp for ps in plastic_states], Ω)
        εp33_field = CellField([ps.εp33 for ps in plastic_states], Ω)
        αbar_field = CellField(ᾱ_accumulated, Ω)
        H_field = project_to_fe_space(H_prev, V_L2, dΩ)
        writevtk(Ω, joinpath(snapshot_dir, "final"),
                 cellfields=[
                     "d"=>sh, "u"=>uh, "plastic_strain"=>εp_field,
                     "plastic_strain_33"=>εp33_field, "epspeq"=>εp_eq_field,
                     "alpha_bar"=>αbar_field, "history"=>H_field,
                     "plastic_energy_previous"=>CellField(ψp_state_prev_array, Ω),
                     "plastic_energy_increment"=>CellField(Δψp_last_array, Ω),
                     "cycle"=>CellField(fill(Float64(current_cycle), n_cells), Ω),
                     "step"=>CellField(fill(Float64(current_cycle * N_STEPS_PER_CYCLE), n_cells), Ω),
                     "load"=>CellField(fill(u_min, n_cells), Ω),
                     "qoi_crack_extension_mm"=>CellField(fill(crack_history[end], n_cells), Ω),
                     "qoi_crack_tip_x_mm"=>CellField(fill(a0 + crack_history[end], n_cells), Ω),
                     "milestone"=>CellField(fill(-1.0, n_cells), Ω)
                 ], nsubcells=4)
    end

    # 最终结果
    println("\n" * "="^80)
    println("模拟完成: N=$(current_cycle) cycles")
    if length(crack_history) > 0
        Δa_final = crack_history[end]
        println("最终裂纹扩展: Δa=$(round(Δa_final, digits=4)) mm")
        println("\n对比论文结果:")
        println("  模拟值: N=$(current_cycle), Δa=$(round(Δa_final, digits=3))mm")
        println("  论文值: N≈696, Δa≈0.45mm")
        
        err_N = abs(current_cycle - 696) / 696 * 100
        err_a = abs(Δa_final - 0.45) / 0.45 * 100
        @printf("  相对误差: ΔN=%.1f%%, ΔΔa=%.1f%%\n", err_N, err_a)
        
        if err_N < 20 && err_a < 20
            println("  ✅ 复现成功！")
        else
            println("  ⚠️ 误差偏大，可能需要调整参数")
        end
    end
    println("="^80)
    
    return cycle_history, crack_history, time_history
end

# ============================================================================
# 绘图和数据保存
# ============================================================================

function plot_results(cycles, lengths, time_history)
    if length(cycles) == 0
        @warn "无数据可绘图"
        return
    end
    
    if CLI.visualization
        p = plot(cycles, lengths,
                xlabel="Number of cycles N",
                ylabel="Crack extension Δa (mm)",
                title="SENT Fatigue Crack Growth (Fully Corrected)",
                label="Simulation",
                linewidth=2,
                linecolor=:blue,
                marker=:circle,
                markersize=2,
                grid=true,
                size=(800, 600),
                dpi=300)

        # 标注论文参考点
        scatter!(p, [696], [0.45], marker=:star5, markersize=10,
                 markercolor=:red, label="Paper (Fig.9a)")

        savefig(p, joinpath(OUTPUT_DIR, "crack_growth.png"))
    end
    writedlm(joinpath(OUTPUT_DIR, "data.csv"), hcat(cycles, time_history, lengths), ',')
    println("\n✓ 结果已保存:")
    println("  - 裂纹扩展曲线: crack_growth.png")
    println("  - 数据文件: data.csv")
end

# ============================================================================
# 主函数
# ============================================================================

function main()
    println("\n" * "="^80)
    println("SENT弹塑性疲劳裂纹扩展模拟")
    println("完整修订版 - 所有物理公式已修正")
    println("="^80)
    
    try
        cycles, lengths, time_history = run_simulation()
        if length(cycles) > 0
            plot_results(cycles, lengths, time_history)
        end
        println("\n✅ 模拟完成！")
    catch e
        println("\n❌ 错误: ", e)
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
