# ============================================================================
# 单边缺口纯剪试样（SENS）弹性相场断裂
#
# 文献: Miehe, Hofacker, Welschinger, CMAME 199 (2010) 2765–2778
#       "A phase field model for rate-independent crack propagation:
#        Robust algorithmic implementation based on operator splits"
#
# 对标: Section 5.2 — Single edge notched pure shear test (Fig. 9 & Fig. 10)
#
# 物理模型:
#   - split 版: Carrara 文中 spectral split 的弹性部分定义 + AT2 + 历史变量场 + 交错算法
#   - nosplit 版: Carrara 文中 isotropic / no-split 的弹性部分定义 + AT2 + 历史变量场 + 交错算法
#
# 几何与材料 (Section 5.2，与 Section 5.1 完全相同):
#   正方形板 1×1 mm²，水平预裂纹从左边界中点延伸至中心
#   λ = 121,150 MPa, G = 80,770 MPa (E≈210,950 MPa, ν≈0.3)
#   Gc = 2.7×10⁻³ kN/mm = 2.7 N/mm
#   l₀ = 0.015 mm (l₁ in paper)
#
# 边界条件 (Fig. 6b) — 与 SENT 不同：
#   底边: ux=0, uy=0 (完全固定)
#   顶边: ux=u (水平剪切位移), uy=0 (竖向约束)
#   左边: uy=0 (防止刚体转动，仅约束竖向)
#   右边: 自由
#
# 加载方案:
#   全程恒定步长 Δu = 1e-5 mm（剪切裂纹扩展较平稳，无需细化步长）
#
# 预期结果 (Fig. 9a):
#   峰值载荷 F_peak ≈ 0.45 kN，位移 u ≈ 10e-3 mm
#   断裂位移 u ≈ 13e-3 mm
#   裂纹路径：从左侧预裂纹尖端向右下角斜向弯曲扩展（Mixed mode）
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
using Printf
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)
using DelimitedFiles

function parse_cli(args)
    output_dir = normpath(joinpath(
        @__DIR__, "..", "outputs", "results_cpu", "elastic_fracture_SENS", "julia"
    ))
    benchmark = "spectral_eta0_split"
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
    benchmark == "spectral_eta0_split" ||
        error("Supported benchmark: spectral_eta0_split")
    return (
        output_dir=abspath(output_dir),
        benchmark=benchmark,
        visualization=visualization,
        snapshots=snapshots,
    )
end

const CLI = parse_cli(ARGS)

# ============================================================================
# 几何参数（与 SENT 相同）
# ============================================================================
const W = 1.0
const H = 1.0
const a0 = 0.5
const y_notch = 0.5
const CH_PRECRACK_FACTOR = 0.5   # 裂纹总宽度取 h_fine，因此半宽为 0.5*h_fine

# ============================================================================
# 材料参数（与 SENT 完全相同）
# ============================================================================
const λ_mat = 121_150.0
const G_mat =  80_770.0
const E_mat = G_mat * (3λ_mat + 2G_mat) / (λ_mat + G_mat)
const ν_mat = λ_mat / (2 * (λ_mat + G_mat))

# ============================================================================
# 相场参数
# ============================================================================
const Gc  = 2.7
const l0  = 0.015
const η_k = 0.0
# ============================================================================
# 网格参数
const h_fine   = 0.005
const h_coarse = 0.05
const REFINE_XMIN_SENS = 0.49
const REFINE_XMAX_SENS = 1.00
const REFINE_YMIN_SENS = 0.00
const REFINE_YMAX_SENS = 1.00

# ============================================================================
# 加载参数
# ============================================================================
const Δu_shear = 2.0e-5    # 全程恒定步长（总步数约1000）
const u_end    = 20.0e-3   # 最大水平位移（当前设置下总步数约1000）
const F_STOP_TOL = 1.0e-3  # 反力停止阈值 [N]，视为“已降到0”

# ============================================================================
# 输出控制
# ============================================================================
const OUTPUT_DIR = CLI.output_dir
const SAVE_VTK   = CLI.visualization
const ENABLE_PNG = CLI.visualization
const SAVE_SNAPSHOTS = CLI.snapshots
const SNAPSHOT_THRESHOLDS = (0.20, 0.50, 0.80, 0.95, 0.99)
const PNG_DPI    = 150
const PNG_NX     = 520
const PNG_NY     = 520
const VTK_ONSET_LEVEL = 0.999
const VTK_FORCE_DROP_LEVELS = (0.75, 0.50, 0.25)

const CASES_TO_RUN = [("split", true)]

abs_output(path::String) = joinpath(abspath(OUTPUT_DIR), path)

function safe_writevtk(trian, filebase; cellfields, nsubcells=4)
    mkpath(dirname(filebase))
    writevtk(trian, filebase, cellfields=cellfields, nsubcells=nsubcells)
end

function write_state_snapshot(Ω, tag::String, label::String, uh, dh, H_fe)
    filebase = joinpath(abspath(OUTPUT_DIR), "snapshots", "snapshot_$(label)_$(tag)")
    safe_writevtk(
        Ω,
        filebase,
        cellfields=["displacement"=>uh, "damage"=>dh, "history"=>H_fe],
        nsubcells=4,
    )
end

# ============================================================================
# Miehe 2010 谱分解（与 SENT 相同）
# ============================================================================
@inline function miehe_psi_plus(ε_2d::SymTensorValue{2,Float64,3})
    ε11 = ε_2d[1,1]
    ε22 = ε_2d[2,2]
    ε12 = ε_2d[1,2]
    avg = 0.5 * (ε11 + ε22)
    dif = 0.5 * (ε11 - ε22)
    rad = sqrt(dif^2 + ε12^2)
    ε1  = avg + rad
    ε2  = avg - rad
    ε3  = 0.0
    trp = max(ε1 + ε2 + ε3, 0.0)
    return 0.5 * λ_mat * trp^2 +
           G_mat * (max(ε1, 0.0)^2 + max(ε2, 0.0)^2 + max(ε3, 0.0)^2)
end

@inline function elastic_psi_undamaged(ε_2d::SymTensorValue{2,Float64,3})
    ε11 = ε_2d[1,1]
    ε22 = ε_2d[2,2]
    ε12 = ε_2d[1,2]
    trε = ε11 + ε22
    return 0.5 * λ_mat * trε^2 + G_mat * (ε11^2 + ε22^2 + 2.0 * ε12^2)
end

@inline g_deg(d::Float64) = (1.0 - clamp(d, 0.0, 1.0))^2 + η_k

# ============================================================================
# 弹性刚度张量
# ============================================================================
function elastic_stiffness()
    return SymFourthOrderTensorValue(
        λ_mat + 2G_mat, 0.0,   λ_mat,
        0.0,            G_mat, 0.0,
        λ_mat,          0.0,   λ_mat + 2G_mat
    )
end
const C_elastic = elastic_stiffness()
const ε_op = symmetric_gradient

# 历史变量更新
# Gridap update_state! 要求返回 (need_to_update::Bool, new_state...)
function update_H(state, psi_plus)
    new_H = max(state, psi_plus)
    return new_H > state, new_H
end

# ============================================================================
# 网格生成
# ============================================================================
function create_mesh_split()
    mkpath(OUTPUT_DIR)
    println("\n生成 SENS 网格（预裂纹 a₀ = $(a0) mm，剪切加载，三角形局部加密）...")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("SENS_shear")

    CH = CH_PRECRACK_FACTOR * h_fine

    # ── 与 SENT 完全相同的裂纹槽几何 ──
    gmsh.model.geo.addPoint(0.0,     0.0,          0.0, h_coarse, 1)
    gmsh.model.geo.addPoint(W,       0.0,          0.0, h_coarse, 2)
    gmsh.model.geo.addPoint(W,       y_notch,      0.0, h_coarse, 10)
    gmsh.model.geo.addPoint(W,       H,            0.0, h_coarse, 3)
    gmsh.model.geo.addPoint(0.0,     H,            0.0, h_coarse, 4)
    gmsh.model.geo.addPoint(0.0,     y_notch+CH,   0.0, h_coarse, 5)
    gmsh.model.geo.addPoint(a0,      y_notch+CH,   0.0, h_fine,   6)
    gmsh.model.geo.addPoint(a0,      y_notch,      0.0, h_fine,   7)   # 裂纹尖端
    gmsh.model.geo.addPoint(a0,      y_notch-CH,   0.0, h_fine,   8)
    gmsh.model.geo.addPoint(0.0,     y_notch-CH,   0.0, h_coarse, 9)

    gmsh.model.geo.addLine(1, 2, 1)   # 底边
    gmsh.model.geo.addLine(2, 10, 2)  # 右边下半
    gmsh.model.geo.addLine(10, 3, 10) # 右边上半
    gmsh.model.geo.addLine(3, 4, 3)   # 顶边
    gmsh.model.geo.addLine(4, 5, 4)   # 左边上半
    gmsh.model.geo.addLine(5, 6, 5)   # 裂纹上唇
    gmsh.model.geo.addLine(6, 7, 6)   # 裂纹右上→尖端
    gmsh.model.geo.addLine(7, 8, 7)   # 裂纹尖端→右下
    gmsh.model.geo.addLine(8, 9, 8)   # 裂纹下唇
    gmsh.model.geo.addLine(9, 1, 9)   # 左边下半

    gmsh.model.geo.addCurveLoop([1, 2, 10, 3, 4, 5, 6, 7, 8, 9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    # ── 物理组 ──
    # SENS 需要额外标记：左边（uy=0 约束）
    gmsh.model.addPhysicalGroup(2, [1], 1);  gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, [1], 1);  gmsh.model.setPhysicalName(1, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [3], 2);  gmsh.model.setPhysicalName(1, 2, "top")
    # 左边两段（上半线4、下半线9）合并为 "left" 用于 uy=0 约束
    gmsh.model.addPhysicalGroup(1, [4, 9], 3); gmsh.model.setPhysicalName(1, 3, "left")

    # ── 固定矩形细化区，不随 l0 改变 ──
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn",  h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", REFINE_XMIN_SENS)
    gmsh.model.mesh.field.setNumber(1, "XMax", REFINE_XMAX_SENS)
    gmsh.model.mesh.field.setNumber(1, "YMin", REFINE_YMIN_SENS)
    gmsh.model.mesh.field.setNumber(1, "YMax", REFINE_YMAX_SENS)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Smoothing", 2)

    gmsh.model.mesh.generate(2)
    mesh_file = joinpath(OUTPUT_DIR, "mesh_SENS.msh")
    gmsh.write(mesh_file)

    n_nodes = length(gmsh.model.mesh.getNodes()[1])
    n_quads = length(gmsh.model.mesh.getElementsByType(3)[1])
    n_trias = length(gmsh.model.mesh.getElementsByType(2)[1])
    println("  网格节点数:   $(n_nodes)")
    println("  三角形单元数: $(n_trias)")
    println("  四边形单元数: $(n_quads)")
    println("  固定细化框: x ∈ [$(REFINE_XMIN_SENS), $(REFINE_XMAX_SENS)], y ∈ [$(REFINE_YMIN_SENS), $(REFINE_YMAX_SENS)]")

    gmsh.finalize()
    return mesh_file
end

# ============================================================================
# 绘制网格
# ============================================================================
function plot_mesh(mesh_file::String)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(mesh_file)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = Dict{Int,Tuple{Float64,Float64}}()
    for i in eachindex(node_tags)
        j = 3 * (i - 1)
        node_map[Int(node_tags[i])] = (node_coords[j + 1], node_coords[j + 2])
    end

    quad_nodes = gmsh.model.mesh.getElementsByType(3)[2]
    tri_nodes  = gmsh.model.mesh.getElementsByType(2)[2]
    n_quads = length(quad_nodes) ÷ 4
    n_trias = length(tri_nodes) ÷ 3
    n_cells = n_quads + n_trias

    p = plot(aspect_ratio=:equal, size=(700,700), dpi=200,
             xlabel="x (mm)", ylabel="y (mm)", legend=false, grid=false,
             title="SENS Mesh  ($(n_cells) elements)")

    for i in 1:n_quads
        ids = quad_nodes[4*(i-1)+1:4*i]
        pts = [node_map[Int(id)] for id in ids]
        xs = [pt[1] for pt in pts]
        ys = [pt[2] for pt in pts]
        plot!(p, Shape(xs, ys), fillcolor=:white, linecolor=:gray55, linewidth=0.35)
    end

    for i in 1:n_trias
        ids = tri_nodes[3*(i-1)+1:3*i]
        pts = [node_map[Int(id)] for id in ids]
        xs = [pt[1] for pt in pts]
        ys = [pt[2] for pt in pts]
        plot!(p, Shape(xs, ys), fillcolor=:white, linecolor=:gray55, linewidth=0.25)
    end

    savefig(p, joinpath(OUTPUT_DIR, "mesh_SENS.png"))
    gmsh.finalize()
    println("  ✓ 网格图: mesh_SENS.png")
end

# ============================================================================
# 标量场云图（PNG，直接按单元坐标填色，和VTU单元场保持一致）
# ============================================================================
function save_damage_plot(cell_coords, d_vals::Vector{Float64};
                          title_str::String, out_file::String,
                          dpi::Int=PNG_DPI, cmap=:jet, vmin=0.0, vmax=1.0)
    grad = cgrad(cmap)
    p = plot(aspect_ratio=:equal, size=(820,700), dpi=dpi,
             xlabel="x (mm)", ylabel="y (mm)", title=title_str,
             legend=false, grid=false, framestyle=:box,
             xlims=(0.0, W), ylims=(0.0, H))
    for i in eachindex(cell_coords)
        xs = [v[1] for v in cell_coords[i]]
        ys = [v[2] for v in cell_coords[i]]
        ξ = clamp((clamp(d_vals[i], vmin, vmax) - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        plot!(p, Shape(xs, ys), fillcolor=grad[ξ],
              linecolor=:transparent, linewidth=0.0, label="")
    end
    plot!(p, [0.0, a0], [y_notch, y_notch], color=:white, linewidth=1.2, label=false)
    savefig(p, out_file)
end

# ============================================================================
# 主求解循环
# ============================================================================
case_marker(tag::String) = joinpath(OUTPUT_DIR, "done_$(tag).flag")

function case_outputs(tag::String)
    return (
        csv = abs_output("F_u_curve_$(tag).csv"),
        curve_png = abs_output("F_u_curve_SENS_$(tag).png"),
        final_png = abs_output("d_final_SENS_$(tag).png"),
        final_vtu = abs_output("result_final_$(tag)"),
    )
end

function split_case_already_done()
    out = case_outputs("split")
    old_csv = joinpath(OUTPUT_DIR, "F_u_curve.csv")
    old_png = joinpath(OUTPUT_DIR, "F_u_curve_SENS.png")
    old_dpng = joinpath(OUTPUT_DIR, "d_final_SENS.png")
    old_vtu = joinpath(OUTPUT_DIR, "result_final.vtu")
    return isfile(case_marker("split")) ||
           (isfile(out.csv) && isfile(out.curve_png) && isfile(out.final_png)) ||
           (isfile(old_csv) && isfile(old_png) && isfile(old_dpng) && isfile(old_vtu))
end

function generic_case_done(tag::String)
    out = case_outputs(tag)
    # Old nosplit runs used a different mesh file pair; if those artifacts exist,
    # force a rerun so the current shared-mesh setup is regenerated.
    if tag == "nosplit"
        old_mesh_msh = joinpath(OUTPUT_DIR, "mesh_SENS_nosplit.msh")
        old_mesh_png = joinpath(OUTPUT_DIR, "mesh_SENS_nosplit.png")
        if isfile(old_mesh_msh) || isfile(old_mesh_png)
            return false
        end
    end
    return isfile(case_marker(tag)) ||
           (isfile(out.csv) && isfile(out.curve_png) && isfile(out.final_png) && isfile(out.final_vtu * ".vtu"))
end

function run_case(case_tag::String, use_split::Bool)
    mkpath(OUTPUT_DIR)
    out = case_outputs(case_tag)
    label = use_split ? "谱分解 (spectral) + AT2" : "各向同性退化 (isotropic/no-split) + AT2"

    if use_split && split_case_already_done()
        println("\n>>> 检测到 split 结果已存在，跳过分解版")
        return nothing
    elseif !use_split && generic_case_done(case_tag)
        println("\n>>> 检测到 $(case_tag) 结果已存在，跳过该算例")
        return nothing
    end

    println("="^70)
    println(" SENS 弹性相场断裂（纯剪）  |  Miehe 2010, Section 5.2")
    println("="^70)
    @printf "  λ = %.0f MPa,  G = %.0f MPa\n" λ_mat G_mat
    @printf "  E ≈ %.0f MPa,  ν ≈ %.4f\n" E_mat ν_mat
    @printf "  Gc = %.4f N/mm,  l₀ = %.4f mm\n" Gc l0
    @printf "  h_fine = %.4f mm = l₀/%.1f\n" h_fine (l0/h_fine)
    println("  加载方式: 水平剪切位移 (顶边 ux=u, uy=0)")
    println("  驱动力: $(label)")
    println("="^70)

    # ── 网格 ──
    mesh_file = create_mesh_split()
    model = GmshDiscreteModel(mesh_file)
    if ENABLE_PNG
        mesh_png = joinpath(OUTPUT_DIR, "mesh_SENS.png")
        if !isfile(mesh_png)
            plot_mesh(mesh_file)
        end
    end

    # ── 区域与测度 ──
    Ω  = Triangulation(model)
    dΩ = Measure(Ω, 2)

    n_cells     = num_cells(Ω)
    cell_coords = get_cell_coordinates(Ω)
    println("\n网格单元数: $(n_cells)")

    # ── 有限元空间 ──
    order    = 1
    reffe_u  = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    reffe_d  = ReferenceFE(lagrangian, Float64, order)
    reffe_dg = ReferenceFE(lagrangian, Float64, 0)

    # 对齐 PIDL 的 SENS_bifurcation：仅顶/底仿射剪切，不额外约束左边
    # 这样 split / nosplit 的差别只来自分解方式本身
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags  = ["bottom", "top"],
                      dirichlet_masks = [(true,true), (true,true)])

    # 相场空间（无 Dirichlet BC）
    V_d = TestFESpace(model, reffe_d, conformity=:H1)
    U_d = TrialFESpace(V_d)

    # DG0 存储历史变量
    V_dg = FESpace(model, reffe_dg, conformity=:L2)

    # ── 初始场 ──
    dh     = interpolate_everywhere(x -> 0.0, U_d)
    H_vals = zeros(n_cells)
    H_fe   = FEFunction(V_dg, H_vals)

    quad_cache = CellQuadrature(Ω, 2)
    cell_pts   = get_cell_points(quad_cache)

    # ── 顶边反力积分所需边界对象 ──
    Γ_top  = BoundaryTriangulation(model; tags=["top"])
    dΓ_top = Measure(Γ_top, 2)
    n_top  = get_normal_vector(Γ_top)
    e_x    = VectorValue(1.0, 0.0)

    # ── 数据记录 ──
    u_history   = Float64[]
    F_history   = Float64[]
    F_kN_history = Float64[]
    time_history = Float64[]

    step      = 0
    u_current = 0.0
    next_vtk_level = 1
    next_snapshot_level = 1
    crack_started = false
    F_onset = 0.0
    uh = nothing
    run_t0 = time()

    println("\n开始单调剪切加载 | case=$(case_tag)")
    println("停止条件: 顶边反力 F ≤ $(F_STOP_TOL) N")
    println("VTK 输出：第一张在 d_max≈1 时保存，随后按反力降至起裂时刻的 $(collect(VTK_FORCE_DROP_LEVELS)) 倍保存，并额外保存最终帧")
    println("PNG 输出：仅保存最终损伤云图")

    while u_current < u_end
        step     += 1
        u_current = min(u_current + Δu_shear, u_end)

        # ── 剪切边界条件 ──
        u_bc_bottom(x) = VectorValue(0.0, 0.0)
        u_bc_top(x)    = VectorValue(u_current, 0.0)
        U_u = TrialFESpace(V_u, [u_bc_bottom, u_bc_top])

        # ── 交错迭代 ──
        min_stagger = 2
        max_stagger = 20
        stagger_tol = 1e-4
        d_old = dh
        uh = nothing
        H_prev_vals = copy(H_vals)
        H_step_vals = copy(H_vals)
        H_step_fe = H_fe
        u_err = Inf
        d_err = Inf
        u_prev_vals = nothing

        for stagger_iter in 1:max_stagger

            # Step 1: 位移求解
            g_d = g_deg ∘ dh
            a_u(u, v) = ∫( g_d * (C_elastic ⊙ ε_op(u) ⊙ ε_op(v)) )dΩ
            l_u(v)    = ∫( VectorValue(0.0,0.0) ⋅ v )dΩ
            op_u = AffineFEOperator(a_u, l_u, U_u, V_u)
            uh   = solve(op_u)

            u_curr_vals = copy(get_free_dof_values(uh))
            if stagger_iter > 1
                u_err = norm(u_curr_vals .- u_prev_vals) / (norm(u_curr_vals) + 1e-15)
            end
            u_prev_vals = u_curr_vals

            # Step 2: 参考 phasefieldx，每次交错都用当前 u 重算驱动力，
            #         但只与上一步历史 H_n 比较，不在当前步内继续累积
            ε_qp   = evaluate(ε_op(uh), cell_pts)
            psi_qp = zeros(n_cells)
            for ic in 1:n_cells
                psi_max = 0.0
                for iq in 1:length(ε_qp[ic])
                    # 对齐 Carrara 2020：
                    # split   -> spectral split active energy
                    # nosplit -> isotropic model, i.e. whole undamaged elastic energy is active
                    ψ_drive = use_split ? miehe_psi_plus(ε_qp[ic][iq]) : elastic_psi_undamaged(ε_qp[ic][iq])
                    psi_max = max(psi_max, ψ_drive)
                end
                psi_qp[ic] = psi_max
            end
            H_step_vals .= max.(H_prev_vals, psi_qp)
            H_step_fe = FEFunction(V_dg, H_step_vals)

            # Step 3: 相场求解
            if use_split
                # Miehe 2010 / AT2
                a_d_split(d, q) = ∫( (Gc/l0 + 2.0*H_step_fe) * d * q
                                   + Gc * l0 * (∇(d) ⋅ ∇(q)) )dΩ
                l_d_split(q)    = ∫( (2.0*H_step_fe) * q )dΩ
                op_d = AffineFEOperator(a_d_split, l_d_split, U_d, V_d)
            else
                # 只改变“是否分解”，不改变相场正则化类型。
                # nosplit 对齐 Carrara 的 isotropic / no-split：
                # 整个未损伤弹性能都作为 active part 被退化。
                a_d_nosplit(d, q) = ∫( (Gc/l0 + 2.0*H_step_fe) * d * q
                                     + Gc * l0 * (∇(d) ⋅ ∇(q)) )dΩ
                l_d_nosplit(q)    = ∫( (2.0*H_step_fe) * q )dΩ
                op_d = AffineFEOperator(a_d_nosplit, l_d_nosplit, U_d, V_d)
            end
            dh_new = solve(op_d)

            # Step 4: 不可逆性
            d_new  = get_free_dof_values(dh_new)
            d_prev = get_free_dof_values(d_old)
            d_new .= max.(d_new, d_prev)
            d_new .= clamp.(d_new, 0.0, 1.0)
            dh_new = FEFunction(U_d, d_new)

            # Step 5: 收敛判断
            Δd_norm = norm(get_free_dof_values(dh_new) .- get_free_dof_values(dh))
            d_norm  = norm(get_free_dof_values(dh_new)) + 1e-15
            d_err   = Δd_norm / d_norm
            dh = dh_new

            if stagger_iter >= min_stagger && max(u_err, d_err) < stagger_tol
                break
            end
        end
        H_vals .= H_step_vals
        H_fe = H_step_fe

        # ── 计算顶边水平反力 Fx = ∫_top (σ·n)·e_x dΓ ──
        σ_deg = (g_deg ∘ dh) * (C_elastic ⊙ ε_op(uh))
        traction_top = σ_deg ⋅ n_top
        F_x  = sum(∫( traction_top ⋅ e_x )dΓ_top)
        F_kN = F_x * 1e-3

        d_max = maximum(get_free_dof_values(dh))

        push!(u_history,    u_current)
        push!(F_history,    F_x)
        push!(F_kN_history, F_kN)
        push!(time_history, time() - run_t0)

        if step % 50 == 0 || d_max > 0.3 || step <= 5
            @printf "  步 %4d | u = %.2e mm | F = %.4f kN | d_max = %.4f\n" step u_current F_kN d_max
        end

        while SAVE_SNAPSHOTS &&
              next_snapshot_level <= length(SNAPSHOT_THRESHOLDS) &&
              d_max >= SNAPSHOT_THRESHOLDS[next_snapshot_level]
            threshold = SNAPSHOT_THRESHOLDS[next_snapshot_level]
            label = @sprintf("dmax_%03d", round(Int, 100 * threshold))
            write_state_snapshot(Ω, case_tag, label, uh, dh, H_fe)
            next_snapshot_level += 1
        end

        # ── VTK 输出：仅保存关键裂纹阶段 5 帧 ──
        if SAVE_VTK && !crack_started && d_max >= VTK_ONSET_LEVEL
            crack_started = true
            F_onset = max(F_x, F_STOP_TOL)
            fname = abs_output(@sprintf("result_onset_%s_step%04d", case_tag, step))
            safe_writevtk(Ω, fname,
                          cellfields=["d" => dh, "uh" => uh, "H" => H_fe],
                          nsubcells=4)
            println(@sprintf("  ✓ 保存起裂 VTK: d_max ≈ 1 at step %d", step))
        end

        while SAVE_VTK && crack_started && next_vtk_level <= length(VTK_FORCE_DROP_LEVELS) &&
              F_x <= VTK_FORCE_DROP_LEVELS[next_vtk_level] * F_onset
            target_ratio = VTK_FORCE_DROP_LEVELS[next_vtk_level]
            fname = abs_output(@sprintf("result_drop%02d_%s_step%04d",
                                        round(Int, 100 * target_ratio), case_tag, step))
            safe_writevtk(Ω, fname,
                          cellfields=["d" => dh, "uh" => uh, "H" => H_fe],
                          nsubcells=4)
            println(@sprintf("  ✓ 保存扩展 VTK: F <= %.2f * F_onset at step %d", target_ratio, step))
            next_vtk_level += 1
        end

        if step > 1 && F_x <= F_STOP_TOL
            println("\n>>> 反力已降到 0 附近 (F = $(round(F_x,digits=6)) N)，模拟结束")
            break
        end
    end

    if SAVE_VTK
        final_vtu = out.final_vtu
        safe_writevtk(Ω, final_vtu,
                      cellfields=["d" => dh, "uh" => uh, "H" => H_fe],
                      nsubcells=4)
        println("✓ 最终 VTK: $(basename(final_vtu))")
    end
    if SAVE_SNAPSHOTS
        write_state_snapshot(Ω, case_tag, "final", uh, dh, H_fe)
    end

    # ── 保存 F-u 数据 ──
    step_history = collect(1:length(u_history))
    fu_data = hcat(step_history, time_history, u_history, F_history, F_kN_history)
    open(out.csv, "w") do f
        println(f, "step,time_s,u_mm,F_N,F_kN")
        writedlm(f, fu_data, ',')
    end
    println("\n✓ F-u 数据已保存: $(out.csv)")

    if ENABLE_PNG
        # ── 绘制 F-u 曲线 ──
        p_fu = plot(u_history .* 1e3, F_kN_history,
                    xlabel="水平位移 u (×10⁻³ mm)", ylabel="剪切反力 F (kN)",
                    title="SENS 纯剪 F-u 曲线 ($(label))",
                    linewidth=2, color=:firebrick, legend=false,
                    size=(800, 500), dpi=150, grid=true)
        annotate!(p_fu, u_history[end]*1e3*0.5, maximum(F_kN_history)*0.9,
                  text("$(case_tag), l₀=$(l0) mm, Gc=$(Gc) N/mm", :left, 10))
        savefig(p_fu, out.curve_png)
        println("✓ F-u 曲线图: $(out.curve_png)")

        # ── 最终损伤云图 ──
        d_qp_final = evaluate(dh, cell_pts)
        d_cell_final = [mean(d_qp_final[i]) for i in 1:length(d_qp_final)]
        save_damage_plot(cell_coords, d_cell_final;
            title_str = "Final phase-field d  (SENS shear, $(label))",
            out_file  = out.final_png,
            vmin=0.0, vmax=1.0)
        println("✓ 最终损伤云图: $(out.final_png)")
    end
    open(case_marker(case_tag), "w") do io
        println(io, "done")
    end

    println("\n" * "="^70)
    println(" case=$(case_tag) 模拟完成！输出目录: $(OUTPUT_DIR)/")
    println("="^70)

    return u_history, F_kN_history, dh
end

# ============================================================================
# 运行
# ============================================================================
function run_simulation()
    for (tag, use_split) in CASES_TO_RUN
        run_case(tag, use_split)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
