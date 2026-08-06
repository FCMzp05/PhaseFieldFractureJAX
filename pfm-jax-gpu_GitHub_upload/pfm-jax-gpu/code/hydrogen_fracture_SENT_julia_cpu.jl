# ============================================================================
# Liu 2025 hydrogen-assisted fracture benchmark, single-edge notched plate
#
# 目标：
# - 先复现 Liu 2025 氢辅助准静态断裂图，而不是直接进入疲劳循环。
# - 方程采用论文/COMSOL 模型中的 u-d-cH 耦合框架：
#   体积-偏差拉伸能、氢致 Gc 退化、AT2 相场方程、应力辅助扩散。
# - 疲劳历史变量 alphaB/alphaN 后续再加入；本脚本作为氢辅助断裂基线。
#
# 图题、CSV 表头和终端打印使用英文，便于后处理和论文对图。
# ============================================================================

using Gmsh: gmsh
using Gridap
using Gridap.Geometry
using Gridap.Geometry: get_cell_coordinates
using Gridap.TensorValues
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Algebra
using GridapGmsh
using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles

const HAS_CAIROMAKIE = let ok = false
    try
        @eval using CairoMakie
        ok = true
    catch
        ok = false
    end
    ok
end

const SCRIPT_VERSION = "canonical_hydrogen_fracture_SENT_julia_cpu_v2"

# ============================================================================
# 默认参数来自 Liu 2025 作者 COMSOL 模型 hydrogen fatigue crack.mph
# 单位统一为 mm, N, MPa, s, mol。
# ============================================================================

const W = 1.0
const H = 1.0
const YC = 0.0
const A0 = 0.5

const E = 210000.0
const NU = 0.3
const LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
const MU = E / (2.0 * (1.0 + NU))
const K_BULK = LAMBDA + 2.0 * MU / 3.0
const EPSILON = symmetric_gradient

const DEFAULT_L0 = 0.010
const DEFAULT_GC0 = 2.7
const DEFAULT_KAPPA = 1.0e-5
const DEFAULT_H_FINE = DEFAULT_L0 / 4.0
const DEFAULT_H_COARSE = 0.01
const DEFAULT_NOTCH_OPENING = 0.020
const DEFAULT_FINE_X_BACK_LENGTH_FACTOR = 4.0
const DEFAULT_FINE_Y_HALF_HEIGHT_FACTOR = 1.25
const DEFAULT_U_MAX = 0.0065   # 对齐 DEM(_chebykan.py) 默认；旧值 0.002 太小常停在弹性段未起裂。
const DEFAULT_STEPS = 120
const DEFAULT_MAX_STAGGER = 200
const DEFAULT_STAGGER_TOL = 1.0e-6
const DEFAULT_STOP_CRACK_LENGTH = 0.45

# ch_env is the model lattice concentration (mol/mm^3, consistent with the
# mass-transport units used below) that corresponds to 1 wt-ppm of hydrogen in
# the environment; hydrogen_boundary_value = hydrogen_wtppm * ch_env sets the
# Dirichlet concentration. The default 5.5e-6 is a calibration value taken from
# the Liu 2025 COMSOL reference model (hydrogen fatigue crack.mph) rather than a
# first-principles wt-ppm -> lattice-molar conversion; if a rigorous
# Sievert/Oriani conversion is needed, recalibrate this constant. Override with
# --ch-env.
const DEFAULT_CH_ENV = 5.5e-6
const DEFAULT_HYDROGEN_WTPPM = 1.0
const DEFAULT_HYDROGEN_BOUNDARY = "crack"
const DEFAULT_HYDROGEN_ENRICHMENT_CAP = 4.0
const DEFAULT_D_H = 0.0127
const DEFAULT_VH = 2000.0
const DEFAULT_R_GAS = 8.314
const DEFAULT_T0 = 300.0
# Stress is represented in MPa and V_H in mm^3/mol, hence V_H*sigma has
# units N*mm/mol. R_GAS is in J/(mol*K), and 1 J = 1000 N*mm.
const JOULE_TO_N_MM = 1000.0
const DEFAULT_DELTA_GB = 30000.0
const DEFAULT_HYDROGEN_CHI = 0.89
const DEFAULT_STRESS_DIFFUSION = true

struct SimArgs
    case_name::String
    steps::Int
    u_max::Float64
    l0::Float64
    gc0::Float64
    kappa::Float64
    h_fine::Float64
    h_coarse::Float64
    notch_opening::Float64
    fine_x_min::Float64
    fine_x_max::Float64
    fine_y_half_height::Float64
    ch_env::Float64
    hydrogen_wtppm::Float64
    hydrogen_boundary::String
    hydrogen_enrichment_cap::Float64
    d_h::Float64
    vh::Float64
    r_gas::Float64
    temperature::Float64
    delta_gb::Float64
    hydrogen_chi::Float64
    stress_assisted_diffusion::Bool
    max_stagger::Int
    stagger_tol::Float64
    stop_crack_length::Float64
    output_dir::String
    run_air_hydrogen::Bool
    mesh_only::Bool
    plot_mesh::Bool
    plot::Bool
    plot_field_events::Bool
    plot_step_fields::Bool
    save_every::Int
    save_vtu::Bool
    hydrogen_sweep::Bool
end

function arg_value(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

has_flag(flag::String) = any(==(flag), ARGS)

function concentration_tag(value::Float64)
    text = replace(@sprintf("%.8g", value), "-" => "m", "." => "p")
    return "$(text)wppm"
end

function default_output_dir(hydrogen_wtppm::Float64)
    script_dir = @__DIR__
    root = basename(script_dir) == "code" ? dirname(script_dir) : script_dir
    return joinpath(
        root,
        "outputs",
        "results_cpu",
        "hydrogen_fracture_SENT",
        "julia",
        concentration_tag(hydrogen_wtppm),
    )
end

const SAVE_SCI_SNAPSHOTS = !has_flag("--no-snapshots")

function parse_args()
    value_flags = Set(["--output-dir", "--hydrogen-wtppm", "--ch-env-1wppm"])
    allowed = union(
        value_flags,
        Set(["--no-visualization", "--no-snapshots", "--help", "-h"]),
    )
    i = 1
    while i <= length(ARGS)
        ARGS[i] in allowed || error("Unknown argument: $(ARGS[i]). This source runs only the canonical case.")
        if ARGS[i] in value_flags
            i == length(ARGS) && error("$(ARGS[i]) requires a value")
            i += 1
        end
        i += 1
    end
    hydrogen_wtppm = parse(Float64, arg_value("--hydrogen-wtppm", "1.0"))
    ch_env_1wppm = parse(Float64, arg_value("--ch-env-1wppm", string(DEFAULT_CH_ENV)))
    hydrogen_wtppm >= 0.0 || error("--hydrogen-wtppm must be non-negative")
    ch_env_1wppm > 0.0 || error("--ch-env-1wppm must be positive")
    visualization = !has_flag("--no-visualization")
    return SimArgs(
        "hydrogen", DEFAULT_STEPS, DEFAULT_U_MAX, DEFAULT_L0, DEFAULT_GC0,
        DEFAULT_KAPPA, DEFAULT_H_FINE, DEFAULT_H_COARSE, DEFAULT_NOTCH_OPENING,
        A0 - DEFAULT_FINE_X_BACK_LENGTH_FACTOR * DEFAULT_L0, W,
        DEFAULT_FINE_Y_HALF_HEIGHT_FACTOR * DEFAULT_L0, ch_env_1wppm,
        hydrogen_wtppm, DEFAULT_HYDROGEN_BOUNDARY,
        DEFAULT_HYDROGEN_ENRICHMENT_CAP, DEFAULT_D_H, DEFAULT_VH, DEFAULT_R_GAS,
        DEFAULT_T0, DEFAULT_DELTA_GB, DEFAULT_HYDROGEN_CHI,
        DEFAULT_STRESS_DIFFUSION, DEFAULT_MAX_STAGGER, DEFAULT_STAGGER_TOL,
        DEFAULT_STOP_CRACK_LENGTH,
        abspath(arg_value("--output-dir", default_output_dir(hydrogen_wtppm))),
        false, false, false, visualization, false, false, 0,
        SAVE_SCI_SNAPSHOTS, false,
    )
end

function print_help()
    println("Usage: julia hydrogen_fracture_SENT_julia_cpu.jl [options]")
    println("  --output-dir PATH       output folder")
    println("  --hydrogen-wtppm VALUE  environmental hydrogen concentration (default: 1)")
    println("  --ch-env-1wppm VALUE    lattice concentration for 1 wt-ppm in mol/mm^3")
    println("  --no-visualization      disable optional PNG plots")
    println("  --no-snapshots          disable milestone/final VTK and CSV field snapshots")
    println("The geometry and numerical defaults follow Liu 2025 Fig. 7; hydrogen concentration is configurable.")
end

function with_case(args::SimArgs, case_name::String)
    return SimArgs(
        case_name, args.steps, args.u_max, args.l0, args.gc0, args.kappa,
        args.h_fine, args.h_coarse, args.notch_opening,
        args.fine_x_min, args.fine_x_max, args.fine_y_half_height,
        args.ch_env,
        args.hydrogen_wtppm, args.hydrogen_boundary, args.hydrogen_enrichment_cap,
        args.d_h, args.vh, args.r_gas, args.temperature, args.delta_gb,
        args.hydrogen_chi, args.stress_assisted_diffusion, args.max_stagger,
        args.stagger_tol, args.stop_crack_length, args.output_dir,
        false, args.mesh_only, args.plot_mesh, args.plot, args.plot_field_events,
        args.plot_step_fields, args.save_every,
        args.save_vtu, false,
    )
end

function case_folder(args::SimArgs)
    if args.case_name == "air"
        return "air"
    end
    htag = replace(@sprintf("%.3g", args.hydrogen_wtppm), "." => "p")
    return "hydrogen_$(htag)wppm_$(args.hydrogen_boundary)"
end

function write_settings(path::String, args::SimArgs, mesh_file::String)
    open(path, "w") do io
        println(io, "script_version: $(SCRIPT_VERSION)")
        for name in fieldnames(SimArgs)
            println(io, "$(name): $(getfield(args, name))")
        end
        println(io, "mesh_file: $(mesh_file)")
        println(io, "model_scope: Gridap FEM u-d-cH Liu 2025 SENT hydrogen-assisted fracture baseline")
    end
end

function read_msh_triangles(mesh_file::String)
    nodes = Dict{Int,Tuple{Float64,Float64}}()
    triangles = NTuple{3,Int}[]
    lines = readlines(mesh_file)
    i = 1
    while i <= length(lines)
        tag = strip(lines[i])
        if tag == "\$Nodes"
            n_nodes = parse(Int, strip(lines[i + 1]))
            for row in (i + 2):(i + 1 + n_nodes)
                parts = split(strip(lines[row]))
                node_id = parse(Int, parts[1])
                nodes[node_id] = (parse(Float64, parts[2]), parse(Float64, parts[3]))
            end
            i += n_nodes + 2
        elseif tag == "\$Elements"
            n_elements = parse(Int, strip(lines[i + 1]))
            for row in (i + 2):(i + 1 + n_elements)
                parts = split(strip(lines[row]))
                elem_type = parse(Int, parts[2])
                n_tags = parse(Int, parts[3])
                if elem_type == 2
                    first_node = 4 + n_tags
                    push!(
                        triangles,
                        (
                            parse(Int, parts[first_node]),
                            parse(Int, parts[first_node + 1]),
                            parse(Int, parts[first_node + 2]),
                        ),
                    )
                end
            end
            i += n_elements + 2
        else
            i += 1
        end
    end
    return nodes, triangles
end

function write_mesh_tables(mesh_file::String, out_dir::String)
    nodes, triangles = read_msh_triangles(mesh_file)
    open(joinpath(out_dir, "mesh_nodes.csv"), "w") do io
        println(io, "node,x,y")
        for node_id in sort(collect(keys(nodes)))
            p = nodes[node_id]
            @printf(io, "%d,%.12e,%.12e\n", node_id, p[1], p[2])
        end
    end
    open(joinpath(out_dir, "mesh_triangles.csv"), "w") do io
        println(io, "cell,n1,n2,n3")
        for (cell_id, tri) in enumerate(triangles)
            @printf(io, "%d,%d,%d,%d\n", cell_id, tri[1], tri[2], tri[3])
        end
    end
end

function phase_degradation(d::Real, args::SimArgs)
    return (1.0 - d)^2 + args.kappa
end

hydrogen_boundary_value(args::SimArgs) = max(args.hydrogen_wtppm, 0.0) * args.ch_env

function hydrogen_toughness_factor(c_h::Real, args::SimArgs)
    if args.case_name == "air"
        return 1.0
    end
    theta = c_h / (c_h + exp(-args.delta_gb / (args.r_gas * args.temperature)))
    return clamp(1.0 - args.hydrogen_chi * theta, 0.0, 1.0)
end

function tensile_energy_voldev(eps)
    exx = eps[1, 1]
    eyy = eps[2, 2]
    exy = eps[1, 2]
    tr_eps = exx + eyy
    tr_pos = max(tr_eps, 0.0)
    dev_xx = exx - tr_eps / 3.0
    dev_yy = eyy - tr_eps / 3.0
    dev_zz = -tr_eps / 3.0
    dev_norm2 = dev_xx^2 + dev_yy^2 + dev_zz^2 + 2.0 * exy^2
    return 0.5 * K_BULK * tr_pos^2 + MU * dev_norm2
end

function split_stress(eps, d, args::SimArgs)
    exx = eps[1, 1]
    eyy = eps[2, 2]
    exy = eps[1, 2]
    tr_eps = exx + eyy
    tr_pos = max(tr_eps, 0.0)
    tr_neg = min(tr_eps, 0.0)
    dev_xx = exx - tr_eps / 3.0
    dev_yy = eyy - tr_eps / 3.0
    dev_xy = exy
    g = phase_degradation(d, args)
    sigma_p = TensorValue(
        K_BULK * tr_pos + 2.0 * MU * dev_xx, 2.0 * MU * dev_xy,
        2.0 * MU * dev_xy, K_BULK * tr_pos + 2.0 * MU * dev_yy,
    )
    sigma_n = TensorValue(
        K_BULK * tr_neg, 0.0,
        0.0, K_BULK * tr_neg,
    )
    return g * sigma_p + sigma_n
end

function split_stress_tangent_positive(d, args::SimArgs)
    g = phase_degradation(d, args)
    c11 = g * (K_BULK + 4.0 * MU / 3.0)
    c12 = g * (K_BULK - 2.0 * MU / 3.0)
    c66 = g * MU
    return SymFourthOrderTensorValue(
        c11, 0.0, c12,
        0.0, c66, 0.0,
        c12, 0.0, c11,
    )
end

function polygon_area(points)
    area = 0.0
    n = length(points)
    for i in 1:n
        p = points[i]
        q = points[i == n ? 1 : i + 1]
        area += p[1] * q[2] - q[1] * p[2]
    end
    return 0.5 * abs(area)
end

function create_mesh(args::SimArgs, out_dir::String)
    mkpath(out_dir)
    mesh_file = joinpath(out_dir, "mesh.msh")

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("liu2025_single_edge_hydrogen_fracture")

        co = args.notch_opening
        y_top = H / 2.0
        y_bot = -H / 2.0

        gmsh.model.geo.addPoint(0.0, y_bot, 0.0, args.h_coarse, 1)
        gmsh.model.geo.addPoint(W, y_bot, 0.0, args.h_coarse, 2)
        gmsh.model.geo.addPoint(W, YC, 0.0, args.h_fine, 3)
        gmsh.model.geo.addPoint(W, y_top, 0.0, args.h_coarse, 4)
        gmsh.model.geo.addPoint(0.0, y_top, 0.0, args.h_coarse, 5)
        gmsh.model.geo.addPoint(0.0, YC + 0.5 * co, 0.0, args.h_coarse, 6)
        gmsh.model.geo.addPoint(A0, YC + 0.5 * co, 0.0, args.h_fine, 7)
        gmsh.model.geo.addPoint(A0, YC, 0.0, args.h_fine, 8)
        gmsh.model.geo.addPoint(A0, YC - 0.5 * co, 0.0, args.h_fine, 9)
        gmsh.model.geo.addPoint(0.0, YC - 0.5 * co, 0.0, args.h_coarse, 10)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 7, 6)
        gmsh.model.geo.addLine(7, 8, 7)
        gmsh.model.geo.addLine(8, 9, 8)
        gmsh.model.geo.addLine(9, 10, 9)
        gmsh.model.geo.addLine(10, 1, 10)
        gmsh.model.geo.addLine(8, 3, 11)

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [11], 2, 1)

        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "domain")
        gmsh.model.addPhysicalGroup(1, [1], 1)
        gmsh.model.setPhysicalName(1, 1, "bottom")
        gmsh.model.addPhysicalGroup(1, [4], 2)
        gmsh.model.setPhysicalName(1, 2, "top")
        gmsh.model.addPhysicalGroup(1, [6, 7, 8, 9], 3)
        gmsh.model.setPhysicalName(1, 3, "crack")
        gmsh.model.addPhysicalGroup(1, [2, 3], 4)
        gmsh.model.setPhysicalName(1, 4, "right")

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", args.h_fine)
        gmsh.model.mesh.field.setNumber(1, "VOut", args.h_coarse)
        gmsh.model.mesh.field.setNumber(1, "XMin", args.fine_x_min)
        gmsh.model.mesh.field.setNumber(1, "XMax", args.fine_x_max)
        gmsh.model.mesh.field.setNumber(1, "YMin", YC - args.fine_y_half_height)
        gmsh.model.mesh.field.setNumber(1, "YMax", YC + args.fine_y_half_height)
        gmsh.model.mesh.field.setNumber(1, "ZMin", -1.0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", 1.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(mesh_file)

        node_count = length(gmsh.model.mesh.getNodes()[1])
        elem_blocks = gmsh.model.mesh.getElements(2)[2]
        elem_count = isempty(elem_blocks) ? 0 : sum(length(block) for block in elem_blocks)
        @printf("Mesh: nodes=%d, elements=%d, h_fine=%.6g mm\n", node_count, elem_count, args.h_fine)
    finally
        gmsh.finalize()
    end

    return mesh_file
end

function mesh_segments(mesh_file::String)
    points = Point2f[]
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mesh_file)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = reshape(node_coords, 3, :)
        node_index = Dict(Int(tag) => i for (i, tag) in enumerate(node_tags))
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        for (etype, tags) in zip(elem_types, elem_node_tags)
            _, _, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            conn = reshape(tags, Int(n_nodes), :)
            for e in 1:size(conn, 2)
                tri = conn[1:min(3, size(conn, 1)), e]
                length(tri) < 3 && continue
                ids = [node_index[Int(t)] for t in tri]
                xy = [Point2f(Float32(coords[1, id]), Float32(coords[2, id])) for id in ids]
                push!(points, xy[1], xy[2], xy[2], xy[3], xy[3], xy[1])
            end
        end
    finally
        gmsh.finalize()
    end
    return points
end

function maybe_plot_mesh(mesh_file::String, out_dir::String)
    if !HAS_CAIROMAKIE
        println("CairoMakie not available, skip mesh plot.")
        return
    end
    points = mesh_segments(mesh_file)
    fig = Figure(size=(900, 760))
    ax = Axis(fig[1, 1], xlabel="x (mm)", ylabel="y (mm)", title="Liu 2025 single-edge mesh", aspect=DataAspect())
    linesegments!(ax, points, color=:black, linewidth=0.25)
    xlims!(ax, 0.0, W)
    ylims!(ax, -H / 2.0, H / 2.0)
    save(joinpath(out_dir, "mesh.png"), fig)

    fig_zoom = Figure(size=(900, 520))
    ax_zoom = Axis(fig_zoom[1, 1], xlabel="x (mm)", ylabel="y (mm)",
                   title="Crack-path mesh refinement", aspect=DataAspect())
    linesegments!(ax_zoom, points, color=:black, linewidth=0.35)
    xlims!(ax_zoom, A0 - 0.08, W)
    ylims!(ax_zoom, YC - 0.04, YC + 0.04)
    save(joinpath(out_dir, "mesh_crack_path_zoom.png"), fig_zoom)
    println("Mesh plots: mesh.png, mesh_crack_path_zoom.png")
end

function solve_displacement(dh, Uu, Vu, dOmega, args::SimArgs)
    c_eff = (x -> split_stress_tangent_positive(x, args)) ∘ dh
    a(u, v) = ∫((c_eff ⊙ EPSILON(u)) ⊙ EPSILON(v))dOmega
    b(v) = ∫(VectorValue(0.0, 0.0) ⋅ v)dOmega
    return solve(AffineFEOperator(a, b, Uu, Vu))
end

function solve_cell_projection(values::Vector{Float64}, U, V, dOmega, omega)
    value_field = CellField(values, omega)
    a(s, v) = ∫(s * v)dOmega
    b(v) = ∫(value_field * v)dOmega
    return solve(AffineFEOperator(a, b, U, V))
end

function solve_hydrogen(ch_old, sigma_h_field, Uc, Vc, dOmega, args::SimArgs)
    if args.case_name == "air"
        return ch_old
    end
    beta = args.stress_assisted_diffusion ?
           args.vh / (JOULE_TO_N_MM * args.r_gas * args.temperature) : 0.0
    a(c, v) = ∫(
        args.d_h * (∇(c) ⋅ ∇(v)) -
        args.d_h * beta * c * (∇(sigma_h_field) ⋅ ∇(v)) +
        1.0e-14 * c * v
    )dOmega
    b(v) = ∫(0.0 * v)dOmega
    ch_trial = solve(AffineFEOperator(a, b, Uc, Vc))
    vals = get_free_dof_values(ch_trial)
    c_boundary = hydrogen_boundary_value(args)
    c_cap = max(args.hydrogen_enrichment_cap, 1.0) * max(c_boundary, 1.0e-30)
    for i in eachindex(vals)
        vals[i] = clamp(vals[i], 0.0, c_cap)
    end
    return FEFunction(Uc, vals)
end

function solve_phase_field(d_old, history_field, gc_factor_field, Ud, Vd, dOmega, args::SimArgs)
    a(d, q) = ∫(
        (gc_factor_field * args.gc0 / args.l0 + 2.0 * history_field) * d * q +
        gc_factor_field * args.gc0 * args.l0 * (∇(d) ⋅ ∇(q))
    )dOmega
    b(q) = ∫((2.0 * history_field) * q)dOmega

    d_trial = solve(AffineFEOperator(a, b, Ud, Vd))
    d_new = get_free_dof_values(d_trial)
    d_prev = get_free_dof_values(d_old)
    for i in eachindex(d_new)
        if isnan(d_new[i]) || isinf(d_new[i])
            d_new[i] = d_prev[i]
        end
        d_new[i] = clamp(max(d_new[i], d_prev[i]), 0.0, 1.0)
    end
    return FEFunction(Ud, d_new)
end

function cell_centers(omega)
    coords = get_cell_coordinates(omega)
    return [
        VectorValue(
            sum(p[1] for p in coords[i]) / length(coords[i]),
            sum(p[2] for p in coords[i]) / length(coords[i]),
        )
        for i in eachindex(coords)
    ]
end

function cell_areas(omega)
    coords = get_cell_coordinates(omega)
    return [polygon_area(coords[i]) for i in eachindex(coords)]
end

function cell_mean_scalar(field, cell_points, n_cells::Int)
    vals = evaluate(field, cell_points)
    return [sum(vals[i]) / max(length(vals[i]), 1) for i in 1:n_cells]
end

function cell_mean_damage(dh, cell_points, n_cells::Int)
    vals = evaluate(dh, cell_points)
    return [
        sum(clamp(v, 0.0, 1.0) for v in vals[i]) / max(length(vals[i]), 1)
        for i in 1:n_cells
    ]
end

function update_history!(history::Vector{Float64}, uh, omega, cell_points)
    eps_vals = evaluate(EPSILON(uh), cell_points)
    n_cells = num_cells(omega)
    max_psi = 0.0
    for i in 1:n_cells
        local_max = 0.0
        for q in 1:length(eps_vals[i])
            local_max = max(local_max, tensile_energy_voldev(eps_vals[i][q]))
        end
        history[i] = max(history[i], local_max)
        max_psi = max(max_psi, local_max)
    end
    return max_psi
end

function hydrostatic_stress_cells(uh, dh, cell_points, n_cells::Int, args::SimArgs)
    eps_vals = evaluate(EPSILON(uh), cell_points)
    d_vals = evaluate(dh, cell_points)
    sigma_h = zeros(Float64, n_cells)
    for i in 1:n_cells
        local_vals = Float64[]
        nq = length(eps_vals[i])
        for q in 1:nq
            d_q = clamp(d_vals[i][q], 0.0, 1.0)
            sigma = split_stress(eps_vals[i][q], d_q, args)
            push!(local_vals, (sigma[1, 1] + sigma[2, 2]) / 3.0)
        end
        sigma_h[i] = mean(local_vals)
    end
    return sigma_h
end

function vertical_stress_cells(uh, dh, cell_points, n_cells::Int, args::SimArgs)
    eps_vals = evaluate(EPSILON(uh), cell_points)
    d_vals = evaluate(dh, cell_points)
    sigma_yy = zeros(Float64, n_cells)
    for i in 1:n_cells
        local_vals = Float64[]
        nq = length(eps_vals[i])
        for q in 1:nq
            d_q = clamp(d_vals[i][q], 0.0, 1.0)
            sigma = split_stress(eps_vals[i][q], d_q, args)
            push!(local_vals, sigma[2, 2])
        end
        sigma_yy[i] = mean(local_vals)
    end
    return sigma_yy
end

function gc_factor_cells(ch, cell_points, n_cells::Int, args::SimArgs)
    if args.case_name == "air"
        # Fast path for the hydrogen-free reference: no toughness degradation and
        # zero concentration field. This is numerically equivalent to the general
        # branch below (hydrogen_toughness_factor returns 1.0 for case=="air" and
        # ch is identically zero in air), but skips the cell-mean projection.
        # Keep the two paths in sync: if hydrogen_toughness_factor's air handling
        # or the air-case ch initialization ever changes, revisit this shortcut.
        return ones(Float64, n_cells), zeros(Float64, n_cells)
    end
    cmean = cell_mean_scalar(ch, cell_points, n_cells)
    return [hydrogen_toughness_factor(c, args) for c in cmean], cmean
end

function crack_tip_by_threshold(damage, coords, centers, args::SimArgs, threshold::Float64)
    x_tip = A0
    for i in eachindex(damage)
        x = centers[i][1]
        y = centers[i][2]
        if x >= A0 - 2.0 * args.l0 && abs(y - YC) <= 4.0 * args.l0 && damage[i] >= threshold
            x_tip = max(x_tip, maximum(p[1] for p in coords[i]))
        end
    end
    return max(x_tip - A0, 0.0), x_tip
end

function phasefield_crack_length(dh, omega, cell_points, areas, args::SimArgs)
    d_vals = evaluate(dh, cell_points)
    grad_vals = evaluate(∇(dh), cell_points)
    length_gamma = 0.0
    for i in 1:num_cells(omega)
        nq = max(length(d_vals[i]), 1)
        gamma_mean = 0.0
        for q in 1:nq
            d = clamp(d_vals[i][q], 0.0, 1.0)
            grad_d = grad_vals[i][q]
            gamma_mean += d^2 / (2.0 * args.l0) + 0.5 * args.l0 * (grad_d ⋅ grad_d)
        end
        length_gamma += areas[i] * gamma_mean / nq
    end
    return length_gamma
end

function reaction_force_y(uh, dh, dGammaTop, args::SimArgs)
    sigma_yy(eps, d) = split_stress(eps, d, args)[2, 2]
    sigma_yy_field = sigma_yy ∘ (EPSILON(uh), dh)
    return sum(∫(sigma_yy_field)dGammaTop)
end

function finite_range(values; limits=nothing, qclip=nothing, symmetric::Bool=false)
    finite_vals = [Float64(v) for v in values if isfinite(Float64(v))]
    if isempty(finite_vals)
        return 0.0, 1.0
    end
    if limits === nothing
        if qclip === nothing
            vmin = minimum(finite_vals)
            vmax = maximum(finite_vals)
        else
            vmin = quantile(finite_vals, qclip[1])
            vmax = quantile(finite_vals, qclip[2])
        end
    else
        vmin, vmax = limits
    end
    if symmetric
        vmax_abs = max(abs(vmin), abs(vmax), 1.0e-12)
        return -vmax_abs, vmax_abs
    end
    if abs(vmax - vmin) < 1.0e-12
        return vmin, vmin + 1.0e-12
    end
    return vmin, vmax
end

function maybe_plot_cell_values(omega, values, out_file::String, title::String, label::String; colormap=:viridis, limits=nothing, qclip=nothing, symmetric::Bool=false)
    if !HAS_CAIROMAKIE
        return
    end
    coords = get_cell_coordinates(omega)
    vmin, vmax = finite_range(values; limits=limits, qclip=qclip, symmetric=symmetric)
    span = max(vmax - vmin, 1.0e-12)
    colors = Makie.to_colormap(colormap)
    fig = Figure(size=(760, 620))
    ax = Axis(fig[1, 1], xlabel="x (mm)", ylabel="y (mm)", title=title, aspect=DataAspect())
    for i in eachindex(values)
        value_i = clamp(Float64(values[i]), vmin, vmax)
        t = (value_i - vmin) / span
        idx = clamp(round(Int, 1 + t * (length(colors) - 1)), 1, length(colors))
        poly!(
            ax,
            Point2f[(Float32(p[1]), Float32(p[2])) for p in coords[i]],
            color=colors[idx],
            strokecolor=:transparent,
            strokewidth=0.0,
        )
    end
    Colorbar(fig[1, 2], limits=(vmin, vmax), colormap=colormap, label=label)
    save(out_file, fig)
end

function xml_escape(text::AbstractString)
    escaped = replace(text, "&" => "&amp;")
    escaped = replace(escaped, "<" => "&lt;")
    escaped = replace(escaped, ">" => "&gt;")
    escaped = replace(escaped, "\"" => "&quot;")
    return replace(escaped, "'" => "&apos;")
end

function write_vtu_cell_fields(
    path::String,
    omega,
    uh,
    dh,
    ch,
    damage,
    cmean,
    sigma_h,
    sigma_yy,
    gc_factor,
    history,
    areas,
    step::Int,
    displacement::Float64,
)
    mkpath(dirname(path))
    vtk_base = endswith(path, ".vtu") ? path[1:end - 4] : path
    writevtk(
        omega,
        vtk_base,
        cellfields=[
            "u"=>uh,
            "d"=>dh,
            "cH_node"=>ch,
            "alpha"=>CellField(damage, omega),
            "cH"=>CellField(cmean, omega),
            "sigma_h_mpa"=>CellField(sigma_h, omega),
            "sigma_yy_mpa"=>CellField(sigma_yy, omega),
            "Gc_factor"=>CellField(gc_factor, omega),
            "history_energy"=>CellField(history, omega),
            "area"=>CellField(areas, omega),
            "step"=>CellField(fill(Float64(step), length(areas)), omega),
            "top_displacement_mm"=>CellField(fill(displacement, length(areas)), omega),
        ],
        nsubcells=1,
    )
end

function write_pvd(path::String, records)
    latest = Dict{String,Any}()
    for record in records
        record.vtu_path == "" && continue
        latest[record.vtu_path] = record
    end
    ordered = sort(collect(values(latest)), by = r -> (r.displacement, r.step, r.label))
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for record in ordered
            rel_path = xml_escape(basename(record.vtu_path))
            label = xml_escape(record.label)
            println(io, "    <DataSet timestep=\"$(record.displacement)\" group=\"$label\" part=\"0\" file=\"$rel_path\"/>")
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
end

function write_snapshot_index(path::String, records)
    open(path, "w") do io
        println(io, "label,step,top_displacement_mm,reaction_force_N_per_mm,crack_length_gamma_mm,crack_length95_mm,max_d,max_cH,min_Gc_factor,vtu_file")
        for record in records
            @printf(io, "%s,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%s\n",
                    record.label, record.step, record.displacement, record.reaction,
                    record.crack_gamma, record.crack_len95, record.max_d,
                    record.max_h, record.min_gc_factor, record.vtu_path)
        end
    end
end

function write_snapshot_outputs(
    case_dir::String,
    label::String,
    snap,
    omega,
    cell_points,
    areas,
    args::SimArgs,
    records;
    write_csv::Bool=true,
    write_vtu::Bool=true,
    write_plot::Bool=false,
)
    SAVE_SCI_SNAPSHOTS || return nothing
    snapshot_t0 = time()
    field_base = joinpath(case_dir, "fields_$(label)")
    if write_csv
        write_cell_fields(
            "$(field_base).csv",
            omega,
            snap.uh,
            snap.dh,
            snap.ch,
            snap.cmean,
            snap.sigma_h,
            snap.sigma_yy,
            snap.gc_factor,
            cell_points,
            areas,
        )
    end

    vtu_path = ""
    if args.save_vtu && write_vtu
        vtu_path = "$(field_base).vtu"
        write_vtu_cell_fields(
            vtu_path,
            omega,
            snap.uh,
            snap.dh,
            snap.ch,
            snap.damage,
            snap.cmean,
            snap.sigma_h,
            snap.sigma_yy,
            snap.gc_factor,
            snap.history,
            areas,
            snap.step,
            snap.u_top_value,
        )
    end

    if write_plot
        c_cap = max(args.hydrogen_enrichment_cap, 1.0) * max(hydrogen_boundary_value(args), 1.0e-30)
        maybe_plot_cell_values(omega, snap.damage, joinpath(case_dir, "damage_$(label).png"),
                               "Damage field | $(args.case_name) | $(label)", "d"; limits=(0.0, 1.0))
        maybe_plot_cell_values(omega, snap.cmean, joinpath(case_dir, "hydrogen_$(label).png"),
                               "Hydrogen concentration | $(args.case_name) | $(label)", "cH"; colormap=:turbo, limits=(0.0, c_cap))
        maybe_plot_cell_values(omega, snap.sigma_h, joinpath(case_dir, "sigma_h_$(label).png"),
                               "Hydrostatic stress | $(args.case_name) | $(label)", "sigma_h (MPa)";
                               colormap=:balance, qclip=(0.01, 0.99), symmetric=true)
        GC.gc()
    end

    push!(records, (
        label=label,
        step=snap.step,
        displacement=snap.u_top_value,
        reaction=snap.reaction,
        crack_gamma=snap.crack_gamma,
        crack_len95=snap.crack_len95,
        max_d=snap.max_d,
        max_h=snap.max_h,
        min_gc_factor=snap.min_gc_factor,
        vtu_path=vtu_path,
    ))
    @printf("  snapshot=%s  csv=%s  vtu=%s  plot=%s  elapsed=%.1fs\n",
            label, string(write_csv), string(args.save_vtu && write_vtu),
            string(write_plot), time() - snapshot_t0)
    return nothing
end

function write_cell_fields(
    path::String,
    omega,
    uh,
    dh,
    ch,
    cmean,
    sigma_h,
    sigma_yy,
    gc_factor,
    cell_points,
    areas,
)
    coords = get_cell_coordinates(omega)
    u_vals = evaluate(uh, cell_points)
    d_vals = evaluate(dh, cell_points)
    open(path, "w") do io
        println(io, "cell,x,y,u,v,alpha,cH,sigma_h_mpa,sigma_yy_mpa,Gc_factor,x1,y1,x2,y2,x3,y3,area")
        for i in 1:num_cells(omega)
            pts = coords[i]
            cx = sum(p[1] for p in pts) / length(pts)
            cy = sum(p[2] for p in pts) / length(pts)
            nq_u = max(length(u_vals[i]), 1)
            nq_d = max(length(d_vals[i]), 1)
            ux = sum(u_vals[i][q][1] for q in 1:nq_u) / nq_u
            uy = sum(u_vals[i][q][2] for q in 1:nq_u) / nq_u
            da = sum(clamp(d_vals[i][q], 0.0, 1.0) for q in 1:nq_d) / nq_d
            p1 = pts[1]
            p2 = pts[min(2, length(pts))]
            p3 = pts[min(3, length(pts))]
            @printf(
                io,
                "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                i,
                cx,
                cy,
                ux,
                uy,
                da,
                cmean[i],
                sigma_h[i],
                sigma_yy[i],
                gc_factor[i],
                p1[1],
                p1[2],
                p2[1],
                p2[2],
                p3[1],
                p3[2],
                areas[i],
            )
        end
    end
end

function maybe_plot_curve(data_path::String, out_file::String, title::String)
    if !HAS_CAIROMAKIE || !isfile(data_path)
        return
    end
    data = readdlm(data_path, ',', skipstart=1)
    size(data, 1) == 0 && return
    fig = Figure(size=(760, 540))
    ax = Axis(fig[1, 1], xlabel="Top displacement (mm)", ylabel="Reaction force (N/mm)", title=title)
    lines!(ax, data[:, 2], data[:, 3], linewidth=2, color=:black)
    scatter!(ax, data[:, 2], data[:, 3], markersize=4, color=:red)
    save(out_file, fig)
end

function append_timing_summary(args::SimArgs, n_steps::Int, total_wall::Float64,
                               total_stagger::Int, n_cells::Int)
    base = args.output_dir
    mkpath(base)
    path = joinpath(base, "timing_summary.csv")
    write_header = !isfile(path)
    open(path, "a") do io
        if write_header
            println(io, "solver,case,hydrogen_wtppm,n_steps,total_wall_s,wall_per_step_s,avg_stagger,n_elements")
        end
        wps = n_steps > 0 ? total_wall / n_steps : 0.0
        ast = n_steps > 0 ? total_stagger / n_steps : 0.0
        @printf(io, "FEM,%s,%.6g,%d,%.3f,%.4f,%.4f,%d\n",
                args.case_name, args.hydrogen_wtppm, n_steps, total_wall, wps, ast, n_cells)
    end
    return path
end

function run_case(args::SimArgs)
    if !(args.case_name in ("air", "hydrogen"))
        error("Unsupported --case $(args.case_name). Use air or hydrogen.")
    end
    if !(args.hydrogen_boundary in ("right", "crack"))
        error("Unsupported --hydrogen-boundary $(args.hydrogen_boundary). Use right or crack.")
    end

    case_dir = args.output_dir
    mkpath(case_dir)

    println("="^72)
    println("Liu 2025 single-edge hydrogen-assisted fracture Gridap reproduction")
    println("script_version=$(SCRIPT_VERSION)")
    @printf("case=%s, steps=%d, u_max=%.6g mm\n", args.case_name, args.steps, args.u_max)
    @printf("l0=%.6g mm, Gc0=%.6g N/mm, kappa=%.3e\n", args.l0, args.gc0, args.kappa)
    @printf("h_fine=%.6g mm, h_fine/l0=%.3f, refined_box_x=[%.6g, %.6g] mm, refined_box_y=%.6g +/- %.6g mm\n",
            args.h_fine, args.h_fine / args.l0, args.fine_x_min, args.fine_x_max, YC, args.fine_y_half_height)
    @printf("hydrogen_wtppm=%.6g, ch_boundary=%.6e, hydrogen_boundary=%s\n",
            args.hydrogen_wtppm, hydrogen_boundary_value(args), args.hydrogen_boundary)
    @printf("ch_env_1wppm=%.6e, enrichment_cap=%.6g, D_H=%.6g mm^2/s, stress_assisted_diffusion=%s, max_stagger=%d\n",
            args.ch_env, args.hydrogen_enrichment_cap, args.d_h, string(args.stress_assisted_diffusion), args.max_stagger)
    @printf("hydrogen_chi=%.6g, delta_gb=%.6g J/mol, T=%.6g K\n",
            args.hydrogen_chi, args.delta_gb, args.temperature)

    mesh_file = create_mesh(args, case_dir)
    write_settings(joinpath(case_dir, "settings.txt"), args, mesh_file)
    write_mesh_tables(mesh_file, case_dir)
    args.plot_mesh && maybe_plot_mesh(mesh_file, case_dir)
    if args.mesh_only
        println("Mesh-only run completed.")
        return nothing
    end

    model = GmshDiscreteModel(mesh_file)
    omega = Triangulation(model)
    dOmega = Measure(omega, 2)
    gamma_top = BoundaryTriangulation(model, tags=["top"])
    dGammaTop = Measure(gamma_top, 2)
    quad = CellQuadrature(omega, 2)
    cell_points = get_cell_points(quad)
    centers = cell_centers(omega)
    areas = cell_areas(omega)
    n_cells = num_cells(omega)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    Vu = TestFESpace(
        model,
        reffe_u,
        conformity=:H1,
        dirichlet_tags=["bottom", "top"],
        dirichlet_masks=[(true, true), (false, true)],
    )

    reffe_d = ReferenceFE(lagrangian, Float64, 1)
    Vd = TestFESpace(model, reffe_d, conformity=:H1)
    Ud = TrialFESpace(Vd)
    dh = interpolate_everywhere(x -> 0.0, Ud)

    reffe_c = ReferenceFE(lagrangian, Float64, 1)
    Vsigma = TestFESpace(model, reffe_c, conformity=:H1)
    Usigma = TrialFESpace(Vsigma)
    if args.case_name == "hydrogen"
        Vc = TestFESpace(model, reffe_c, conformity=:H1, dirichlet_tags=[args.hydrogen_boundary])
        c_boundary = hydrogen_boundary_value(args)
        Uc = TrialFESpace(Vc, x -> c_boundary)
        ch = interpolate_everywhere(x -> 0.0, Uc)
    else
        Vc = TestFESpace(model, reffe_c, conformity=:H1)
        Uc = TrialFESpace(Vc)
        ch = interpolate_everywhere(x -> 0.0, Uc)
    end

    history = zeros(Float64, n_cells)
    sigma_h = zeros(Float64, n_cells)

    base_gamma = phasefield_crack_length(dh, omega, cell_points, areas, args)
    data_path = joinpath(case_dir, "data.csv")
    diag_path = joinpath(case_dir, "diagnostics.csv")
    open(data_path, "w") do io
        println(io, "step,top_displacement_mm,reaction_force_N_per_mm,crack_length_gamma_mm,crack_length95_mm,x_tip95_mm,max_d,max_cH,min_Gc_factor,hydrogen_wtppm,ch_env_1wppm_mol_per_mm3,cH_boundary_mol_per_mm3")
    end
    open(diag_path, "w") do io
        println(io, "step,top_displacement_mm,max_history,max_sigma_h_mpa,min_sigma_h_mpa,max_cH,min_Gc_factor,stagger_iterations")
    end
    timing_path = joinpath(case_dir, "timing.csv")
    open(timing_path, "w") do io
        println(io, "step,top_displacement_mm,stagger_iterations,step_wall_s,cumulative_wall_s")
    end

    # 自适应位移步进：断裂(软化)段在位移轴上极窄，均匀 du 会一步跨过整个软化区
    # 导致 F-U 曲线/快照看不见裂纹扩展。这里以 du_base=u_max/steps 起步，每步算完用
    # 双判据(Δmax_d 与 Δcrack_len95)检查：任一超上限就回退该步、du 二分重解(恢复
    # dh/history/ch 的不可逆状态)，直到单步增量可控或触底 du_min；都很小就放大 du。
    # --no-adaptive-steps 退回旧的均匀步进。
    adaptive_steps = !has_flag("--no-adaptive-steps")
    adapt_refine_max = max(parse(Int, arg_value("--adapt-refine-max", "16")), 1)
    adapt_dmax_hi = parse(Float64, arg_value("--adapt-dmax-hi", "0.08"))
    adapt_len_hi = parse(Float64, arg_value("--adapt-len-hi", "0.02"))
    adapt_dmax_lo = parse(Float64, arg_value("--adapt-dmax-lo", "0.02"))
    adapt_len_lo = parse(Float64, arg_value("--adapt-len-lo", "0.005"))
    du_base = args.u_max / args.steps
    du_min = du_base / adapt_refine_max
    du = du_base
    u_prev = 0.0
    step = 0
    prev_max_d = 0.0
    prev_crack95 = 0.0
    attempt = 0
    max_attempts = args.steps * adapt_refine_max * 4 + 50
    run_t0 = time()
    last_t = run_t0
    total_stagger = 0
    peak_reaction = -Inf
    max_damage_seen = -Inf
    snapshot_records = NamedTuple[]
    previous_snapshot = nothing
    peak_snapshot = nothing
    max_damage_snapshot = nothing
    elastic_saved = false
    pre_crack_saved = false
    crack_onset_saved = false
    post_crack_saved = false
    damage_milestones = [
        ("dmax_020", 0.20),
        ("dmax_050", 0.50),
        ("dmax_080", 0.80),
        ("dmax_095", 0.95),
        ("dmax_099", 0.99),
    ]
    damage_milestone_saved = Dict(label => false for (label, _) in damage_milestones)
    elastic_step = max(1, round(Int, 0.15 * args.steps))
    elastic_damage_threshold = 0.02
    crack_onset_damage_threshold = 0.20
    post_crack_length_threshold = max(5.0 * args.l0, 0.05)

    while u_prev < args.u_max - 1.0e-12 && attempt < max_attempts
        attempt += 1
        u_top_value = min(u_prev + du, args.u_max)
        dh_save = copy(get_free_dof_values(dh))
        history_save = copy(history)
        ch_save = copy(get_free_dof_values(ch))
        uh = nothing
        ch_trial = ch
        gc_factor = ones(Float64, n_cells)
        cmean = zeros(Float64, n_cells)
        max_psi = 0.0
        fp_iters = 0

        for fp in 1:args.max_stagger
            fp_iters = fp
            d_old_vals = copy(get_free_dof_values(dh))

            u_bottom(x) = VectorValue(0.0, 0.0)
            u_top(x) = VectorValue(0.0, u_top_value)
            Uu = TrialFESpace(Vu, [u_bottom, u_top])
            uh = solve_displacement(dh, Uu, Vu, dOmega, args)

            sigma_h = hydrostatic_stress_cells(uh, dh, cell_points, n_cells, args)
            sigma_h_field = solve_cell_projection(sigma_h, Usigma, Vsigma, dOmega, omega)
            ch_trial = solve_hydrogen(ch, sigma_h_field, Uc, Vc, dOmega, args)
            gc_factor, cmean = gc_factor_cells(ch_trial, cell_points, n_cells, args)

            max_psi = update_history!(history, uh, omega, cell_points)
            history_field = CellField(history, omega)
            gc_factor_field = CellField(gc_factor, omega)
            dh = solve_phase_field(dh, history_field, gc_factor_field, Ud, Vd, dOmega, args)

            rel = norm(get_free_dof_values(dh) - d_old_vals) / (norm(get_free_dof_values(dh)) + 1.0e-12)
            rel < args.stagger_tol && break
        end

        ch = ch_trial
        coords = get_cell_coordinates(omega)
        damage = cell_mean_damage(dh, cell_points, n_cells)
        crack_len95, x_tip95 = crack_tip_by_threshold(damage, coords, centers, args, 0.95)
        crack_gamma = max(phasefield_crack_length(dh, omega, cell_points, areas, args) - base_gamma, 0.0)
        max_d = maximum(damage)
        max_h = isempty(cmean) ? 0.0 : maximum(cmean)
        min_gc_factor = minimum(gc_factor)
        reaction = reaction_force_y(uh, dh, dGammaTop, args)
        sigma_yy = vertical_stress_cells(uh, dh, cell_points, n_cells, args)

        Δmax_d = max_d - prev_max_d
        Δlen = crack_len95 - prev_crack95
        if adaptive_steps && du > du_min * (1.0 + 1.0e-9) && (Δmax_d > adapt_dmax_hi || Δlen > adapt_len_hi)
            # 回退该步：恢复损伤/历史/氢的不可逆状态，du 二分重解。
            dh = FEFunction(Ud, copy(dh_save))
            history .= history_save
            ch = FEFunction(Uc, copy(ch_save))
            du = max(du / 2.0, du_min)
            continue
        end
        step += 1

        open(data_path, "a") do io
            @printf(io, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    step, u_top_value, reaction, crack_gamma, crack_len95,
                    x_tip95, max_d, max_h, min_gc_factor,
                    args.hydrogen_wtppm, args.ch_env,
                    hydrogen_boundary_value(args))
        end
        open(diag_path, "a") do io
            @printf(io, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d\n",
                    step, u_top_value, max_psi, maximum(sigma_h), minimum(sigma_h),
                    max_h, min_gc_factor, fp_iters)
        end
        now_t = time()
        step_wall = now_t - last_t
        last_t = now_t
        total_stagger += fp_iters
        open(timing_path, "a") do io
            @printf(io, "%d,%.10e,%d,%.6f,%.6f\n",
                    step, u_top_value, fp_iters, step_wall, now_t - run_t0)
        end

        @printf("step=%4d  u=%.6e mm  F=%.6e N/mm  Lgamma=%.6e mm  a95=%.6e mm  max_d=%.4f  max_cH=%.3e  min_fH=%.4f  fp=%d  elapsed=%.1fs\n",
                step, u_top_value, reaction, crack_gamma, crack_len95, max_d, max_h,
                min_gc_factor, fp_iters, time() - run_t0)

        current_snapshot = (
            label="current",
            step=step,
            u_top_value=u_top_value,
            uh=uh,
            dh=dh,
            ch=ch,
            cmean=copy(cmean),
            sigma_h=copy(sigma_h),
            sigma_yy=copy(sigma_yy),
            gc_factor=copy(gc_factor),
            damage=copy(damage),
            history=copy(history),
            crack_gamma=crack_gamma,
            crack_len95=crack_len95,
            reaction=reaction,
            max_d=max_d,
            max_h=max_h,
            min_gc_factor=min_gc_factor,
        )

        # Save the same damage milestones used by the canonical JAX run so
        # post-processing can place Julia above JAX in matched field blocks.
        for (label, threshold) in damage_milestones
            if !damage_milestone_saved[label] && max_d >= threshold
                write_snapshot_outputs(
                    case_dir,
                    label,
                    current_snapshot,
                    omega,
                    cell_points,
                    areas,
                    args,
                    snapshot_records;
                    write_plot=false,
                )
                damage_milestone_saved[label] = true
            end
        end

        if args.save_every > 0 && (step % args.save_every == 0 || step == args.steps)
            write_snapshot_outputs(
                case_dir,
                "step$(lpad(step, 4, '0'))",
                current_snapshot,
                omega,
                cell_points,
                areas,
                args,
                snapshot_records;
                write_plot=args.plot && args.plot_step_fields,
            )
        end
        if reaction > peak_reaction
            peak_reaction = reaction
            peak_snapshot = current_snapshot
        end
        if max_d > max_damage_seen
            max_damage_seen = max_d
            max_damage_snapshot = current_snapshot
        end

        damage_onset = (max_d >= crack_onset_damage_threshold) || (crack_len95 > 0.0)
        if !elastic_saved && step >= elastic_step && max_d <= elastic_damage_threshold && crack_len95 <= 0.0
            write_snapshot_outputs(case_dir, "elastic", current_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
            elastic_saved = true
        end
        if damage_onset
            if !elastic_saved && previous_snapshot !== nothing && previous_snapshot.max_d <= 0.05
                write_snapshot_outputs(case_dir, "elastic", previous_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
                elastic_saved = true
            end
            if !pre_crack_saved && previous_snapshot !== nothing
                write_snapshot_outputs(case_dir, "pre_crack", previous_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
                pre_crack_saved = true
            end
            if !crack_onset_saved
                write_snapshot_outputs(case_dir, "crack_onset", current_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
                crack_onset_saved = true
            end
        end
        if !post_crack_saved && crack_len95 >= post_crack_length_threshold
            write_snapshot_outputs(case_dir, "post_crack", current_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
            post_crack_saved = true
        end
        previous_snapshot = current_snapshot
        prev_max_d = max_d
        prev_crack95 = crack_len95
        u_prev = u_top_value
        if adaptive_steps && Δmax_d < adapt_dmax_lo && Δlen < adapt_len_lo
            du = min(du * 1.5, du_base)
        end

        if args.stop_crack_length > 0.0 && crack_len95 >= args.stop_crack_length
            @printf("Stop: crack_length95=%.6f mm reached stop_crack_length=%.6f mm at step=%d\n",
                    crack_len95, args.stop_crack_length, step)
            break
        end
    end

    if previous_snapshot !== nothing
        if !elastic_saved && previous_snapshot.max_d <= 0.05
            write_snapshot_outputs(case_dir, "elastic", previous_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
        end
        if peak_snapshot !== nothing
            write_snapshot_outputs(case_dir, "peak_load", peak_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
        end
        if max_damage_snapshot !== nothing
            write_snapshot_outputs(case_dir, "max_damage", max_damage_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
        end
        write_snapshot_outputs(case_dir, "final", previous_snapshot, omega, cell_points, areas, args, snapshot_records; write_plot=args.plot && args.plot_field_events)
        write_snapshot_index(joinpath(case_dir, "snapshots.csv"), snapshot_records)
        args.save_vtu && write_pvd(joinpath(case_dir, "fields_snapshots.pvd"), snapshot_records)
    end
    maybe_plot_curve(data_path, joinpath(case_dir, "load_displacement.png"),
                     "Load-displacement | $(args.case_name)")
    total_wall = time() - run_t0
    summary_path = append_timing_summary(args, step, total_wall, total_stagger, n_cells)
    @printf("Timing: n_steps=%d  total_wall=%.1fs  wall/step=%.3fs  avg_stagger=%.2f\n",
            step, total_wall, step > 0 ? total_wall / step : 0.0,
            step > 0 ? total_stagger / step : 0.0)
    println("Done.")
    println("Output directory: $(case_dir)")
    println("Main outputs: data.csv, diagnostics.csv, timing.csv")
    println("Timing summary (appended): $(summary_path)")
    return nothing
end

function main()
    if has_flag("--help") || has_flag("-h")
        print_help()
        return nothing
    end
    args = parse_args()
    run_case(args)
    return nothing
end

main()
