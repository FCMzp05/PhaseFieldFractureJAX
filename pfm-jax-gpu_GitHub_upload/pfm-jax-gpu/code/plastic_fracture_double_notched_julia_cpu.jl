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
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)

function parse_cli(args)
    output_dir = normpath(joinpath(
        @__DIR__, "..", "outputs", "results_cpu",
        "plastic_fracture_double_notched", "julia"
    ))
    benchmark = "paper"
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
    benchmark == "paper" || error("Supported benchmark: paper")
    return (output_dir=abspath(output_dir), benchmark=benchmark,
            visualization=visualization, snapshots=snapshots)
end

const CLI = parse_cli(ARGS)

# ============================================================================
# Ambati et al. (2015) / Miehe-style ductile phase-field benchmark
# Double notched specimen, monotonic loading
#
# Reference:
#   Phase-field modeling of ductile fracture
#   Section 5.5 "Double notched specimen"
#
# Benchmark quantities followed from the paper:
# - geometry: compact 10 mm x 10 mm specimen with two opposite-corner notches
# - boundary conditions: top and left edges displaced vertically and restrained
#   horizontally; bottom and right edges fixed
# - material: Material II from Table 3
# - fracture length scale: l = 0.03 mm
#
# Implementation route:
# - same small-strain J2 + AT2 staggered structure used in the other scripts
#   of this repository
# - triangular mesh with diagonal refinement band between the two notch roots
# - outputs: mesh, force-displacement, process VTU/PNG, final d / epspeq
# ============================================================================

# ============================================================================
# Geometry
# ============================================================================

const W = 10.0
const H_specimen = 10.0
const BL_NOTCH_CX = 0.0
const BL_NOTCH_CY = 0.0
const TR_NOTCH_CX = W
const TR_NOTCH_CY = H_specimen

struct GeometryCase
    tag::String
    title::String
    bl_radius::Float64
    tr_radius::Float64
    u_total::Float64
    left_scale::Float64
end

const CASES = [
    GeometryCase("paper", "Paper baseline", 2.5, 2.0, 0.40, 1.00),
]

# ============================================================================
# Material
# ============================================================================

struct MaterialData
    E::Float64
    ν::Float64
    G::Float64
    λ::Float64
    σy0::Float64
    H_hard::Float64
end

function create_material()
    G = 70_300.0
    K = 136_500.0
    E = 9.0 * K * G / (3.0 * K + G)
    ν = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    λ = K - 2.0 * G / 3.0
    σy0 = 443.0
    H_hard = 300.0
    return MaterialData(E, ν, G, λ, σy0, H_hard)
end

const lc = 0.03
const Gc = 20.9
const η_stab = 1e-4
const THICKNESS = 1.0

# ============================================================================
# Loading / solver controls
# ============================================================================

const N_STEPS = 400

const STOP_FORCE_RATIO = 0.05
const STOP_AFTER_STEPS = 6

const max_stagger = 30
const stagger_tol = 1e-6

# ============================================================================
# Mesh / output
# ============================================================================

const h_fine = 0.07
const h_coarse = 0.45

const OUTPUT_DIR = CLI.output_dir
const SNAPSHOT_DAMAGE_LEVELS = [0.20, 0.50, 0.80, 0.95, 0.99]

# ============================================================================
# Elasticity
# ============================================================================

function elastic_stiffness_plane_strain(mat::MaterialData)
    λ, μ = mat.λ, mat.G
    C1111 = λ + 2μ
    C1122 = λ
    C1212 = μ
    return SymFourthOrderTensorValue(
        C1111, 0.0, C1122,
        0.0, C1212, 0.0,
        C1122, 0.0, C1111
    )
end

const ε = symmetric_gradient

# ============================================================================
# Plasticity state
# ============================================================================

mutable struct PlasticStateData
    εp::SymTensorValue{2,Float64,3}
    εp33::Float64
    εp_eq::Float64
end

PlasticStateData() = PlasticStateData(SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0, 0.0)

function copy_plastic_state(ps::PlasticStateData)
    return PlasticStateData(ps.εp, ps.εp33, ps.εp_eq)
end

# ============================================================================
# Linear isotropic hardening elastoplasticity
# ============================================================================

function yield_stress(εp_eq::Float64, mat::MaterialData)
    return mat.σy0 + mat.H_hard * εp_eq
end

function plastic_strain_energy(εp_eq::Float64, mat::MaterialData)
    return mat.σy0 * εp_eq + 0.5 * mat.H_hard * εp_eq^2
end

function return_mapping!(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    mat::MaterialData,
    d_phasefield::Float64;
    update_state::Bool = true
)
    g_d = (1 - d_phasefield)^2 + η_stab
    λ = mat.λ
    μ = mat.G
    EG3 = 3.0 * μ

    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33

    trεe = εe11 + εe22 + εe33
    σ11_0 = λ * trεe + 2μ * εe11
    σ22_0 = λ * trεe + 2μ * εe22
    σ33_0 = λ * trεe + 2μ * εe33
    σ12_0 = 2μ * εe12

    p = (σ11_0 + σ22_0 + σ33_0) / 3.0
    s11 = σ11_0 - p
    s22 = σ22_0 - p
    s33 = σ33_0 - p
    s12 = σ12_0
    σ_eq = sqrt(max(1.5 * (s11^2 + s22^2 + s33^2 + 2.0 * s12^2), 0.0))

    σy = yield_stress(plastic.εp_eq, mat)

    if σ_eq <= (1.0 + 1e-6) * σy || σ_eq < 1e-10
        SSE = (σ11_0 * εe11 + σ22_0 * εe22 + σ33_0 * εe33 + 2.0 * σ12_0 * εe12) / 2.0
        return g_d * SymTensorValue{2,Float64}(σ11_0, σ12_0, σ22_0), SSE
    end

    DEQPL = 0.0
    SYIELD = σy
    for _ in 1:20
        RHS = σ_eq - EG3 * DEQPL - SYIELD
        DEQPL = DEQPL + RHS / (EG3 + mat.H_hard)
        SYIELD = yield_stress(plastic.εp_eq + DEQPL, mat)
        if abs(RHS) < 1e-6 * mat.σy0
            break
        end
    end

    inv_σ = 1.0 / σ_eq
    flow11 = s11 * inv_σ
    flow22 = s22 * inv_σ
    flow33 = s33 * inv_σ
    flow12 = s12 * inv_σ

    σ11_new = flow11 * SYIELD + p
    σ22_new = flow22 * SYIELD + p
    σ33_new = flow33 * SYIELD + p
    σ12_new = flow12 * SYIELD

    Δεp11 = 1.5 * flow11 * DEQPL
    Δεp22 = 1.5 * flow22 * DEQPL
    Δεp33 = 1.5 * flow33 * DEQPL
    Δεp12 = 1.5 * flow12 * DEQPL

    εe11_new = εe11 - Δεp11
    εe22_new = εe22 - Δεp22
    εe33_new = εe33 - Δεp33
    εe12_new = εe12 - Δεp12

    SSE = (σ11_new * εe11_new + σ22_new * εe22_new +
           σ33_new * εe33_new + 2.0 * σ12_new * εe12_new) / 2.0

    if update_state
        plastic.εp = plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22)
        plastic.εp33 = plastic.εp33 + Δεp33
        plastic.εp_eq = plastic.εp_eq + DEQPL
    end

    return g_d * SymTensorValue{2,Float64}(σ11_new, σ12_new, σ22_new), SSE
end

# ============================================================================
# History variable
# ============================================================================

function update_history_variable(ψ_total::Real, H_old::Real)
    return (true, max(H_old, ψ_total))
end

# ============================================================================
# FE helpers
# ============================================================================

function project_to_fe_space(q, V_proj, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * q) * dΩ
    op = AffineFEOperator(a, b, V_proj, V_proj)
    return solve(op)
end

function array_to_fe_field(arr, V_dg0, V_proj, dΩ)
    f_dg = FEFunction(V_dg0, arr)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * f_dg) * dΩ
    op = AffineFEOperator(a, b, V_proj, V_proj)
    return solve(op)
end

# ============================================================================
# Mesh
# ============================================================================

function create_mesh(case_dir::String, geom::GeometryCase)
    bl_radius = geom.bl_radius
    tr_radius = geom.tr_radius
    bl_intersection = bl_radius
    tr_intersection = H_specimen - tr_radius

    println("\nCreating double-notched monotonic mesh...")
    println("  W = $(W) mm, H = $(H_specimen) mm")
    println("  notch centers = ($(BL_NOTCH_CX), $(BL_NOTCH_CY)) and ($(TR_NOTCH_CX), $(TR_NOTCH_CY))")
    println("  radii = $(bl_radius) mm / $(tr_radius) mm")
    println("  h_fine = $(h_fine) mm, h_coarse = $(h_coarse) mm")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Ductile_double_notched")

    pt = 1
    gmsh.model.geo.addPoint(bl_intersection, 0.0, 0.0, h_fine, pt); p_BB = pt; pt += 1
    gmsh.model.geo.addPoint(W, 0.0, 0.0, h_coarse, pt); p_BR = pt; pt += 1
    gmsh.model.geo.addPoint(W, tr_intersection, 0.0, h_fine, pt); p_RR = pt; pt += 1
    gmsh.model.geo.addPoint(W, H_specimen, 0.0, h_coarse, pt); p_TR = pt; pt += 1
    gmsh.model.geo.addPoint(tr_intersection, H_specimen, 0.0, h_fine, pt); p_TT = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, H_specimen, 0.0, h_coarse, pt); p_TL = pt; pt += 1
    gmsh.model.geo.addPoint(0.0, bl_intersection, 0.0, h_fine, pt); p_LL = pt; pt += 1

    gmsh.model.geo.addPoint(BL_NOTCH_CX, BL_NOTCH_CY, 0.0, h_fine, pt); p_CBL = pt; pt += 1
    gmsh.model.geo.addPoint(TR_NOTCH_CX, TR_NOTCH_CY, 0.0, h_fine, pt); p_CTR = pt; pt += 1

    bl_root_x = bl_radius / sqrt(2.0)
    bl_root_y = bl_radius / sqrt(2.0)
    tr_root_x = W - tr_radius / sqrt(2.0)
    tr_root_y = H_specimen - tr_radius / sqrt(2.0)
    gmsh.model.geo.addPoint(bl_root_x, bl_root_y, 0.0, h_fine, pt); p_diag_start = pt; pt += 1
    gmsh.model.geo.addPoint(tr_root_x, tr_root_y, 0.0, h_fine, pt); p_diag_end = pt; pt += 1

    line = 1
    gmsh.model.geo.addLine(p_BB, p_BR, line); line_bottom = line; line += 1
    gmsh.model.geo.addLine(p_BR, p_RR, line); line_right = line; line += 1
    gmsh.model.geo.addCircleArc(p_RR, p_CTR, p_TT, line); line_arc_tr = line; line += 1
    gmsh.model.geo.addLine(p_TT, p_TL, line); line_top = line; line += 1
    gmsh.model.geo.addLine(p_TL, p_LL, line); line_left = line; line += 1
    gmsh.model.geo.addCircleArc(p_LL, p_CBL, p_BB, line); line_arc_bl = line; line += 1
    gmsh.model.geo.addLine(p_diag_start, p_diag_end, line); line_diag = line; line += 1

    gmsh.model.geo.addCurveLoop([
        line_bottom,
        line_right,
        line_arc_tr,
        line_top,
        line_left,
        line_arc_bl
    ], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    gmsh.model.addPhysicalGroup(1, [line_bottom], 1)
    gmsh.model.setPhysicalName(1, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [line_right], 2)
    gmsh.model.setPhysicalName(1, 2, "right")
    gmsh.model.addPhysicalGroup(1, [line_top], 3)
    gmsh.model.setPhysicalName(1, 3, "top")
    gmsh.model.addPhysicalGroup(1, [line_left], 4)
    gmsh.model.setPhysicalName(1, 4, "left")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [line_diag, line_arc_bl, line_arc_tr])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 160)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", h_fine)
    gmsh.model.mesh.field.setNumber(2, "LcMax", h_coarse)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 2.00)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 9.00)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)

    num_nodes = length(gmsh.model.mesh.getNodes()[1])
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_tri = 0
    for (etype, tags) in zip(elem_types, elem_tags)
        if etype == 2
            n_tri += length(tags)
        end
    end

    println("  nodes = $(num_nodes)")
    println("  triangles = $(n_tri)")
    println("  diagonal refinement band + quarter-circle notch arcs")

    mesh_file = joinpath(case_dir, "mesh_double_notched.msh")
    gmsh.write(mesh_file)
    gmsh.finalize()
    return mesh_file
end

# ============================================================================
# Stress update helpers
# ============================================================================

function compute_elastoplastic_stress!(
    uh, dh, plastic_states::Vector{PlasticStateData},
    mat::MaterialData, Ω, update::Bool;
    cell_points=nothing
)
    nc = num_cells(Ω)
    if cell_points === nothing
        quad = CellQuadrature(Ω, 2)
        cell_points = get_cell_points(quad)
    end
    εh = ε(uh)
    ε_qp = evaluate(εh, cell_points)
    d_qp = evaluate(dh, cell_points)

    σ_vals = Vector{SymTensorValue{2,Float64,3}}(undef, nc)
    sse_vals = zeros(nc)
    for i in 1:nc
        ε_cell = ε_qp[i][1]
        d_cell = clamp(d_qp[i][1], 0.0, 1.0)
        σ_vals[i], sse_vals[i] = return_mapping!(ε_cell, plastic_states[i], mat, d_cell, update_state=update)
    end
    return σ_vals, sse_vals
end

# ============================================================================
# Displacement solve
# ============================================================================

function solve_displacement(
    dh, plastic_states::Vector{PlasticStateData},
    mat::MaterialData, C_tensor,
    u_val::Float64, geom::GeometryCase, V_u, dΩ, Ω,
    uh_prev, u_prev::Float64;
    cell_points_cache=nothing
)
    g_d(d) = (1 - d)^2 + η_stab
    nc = num_cells(Ω)
    plastic_work = [copy_plastic_state(ps) for ps in plastic_states]

    Δu_step = u_val - u_prev
    n_sub = max(1, ceil(Int, abs(Δu_step) / 0.001))
    uh = uh_prev

    for sub in 1:n_sub
        u_cur = u_prev + Δu_step * sub / n_sub
        u_bottom(x) = VectorValue(0.0, 0.0)
        u_right(x) = VectorValue(0.0, 0.0)
        u_top(x) = VectorValue(0.0, u_cur)
        u_left(x) = VectorValue(0.0, geom.left_scale * u_cur)
        U_u_cur = TrialFESpace(V_u, [u_bottom, u_right, u_top, u_left])

        plastic_sub_start = [copy_plastic_state(ps) for ps in plastic_work]

        for _ in 1:10
            u_old = copy(get_free_dof_values(uh))

            σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, nc)
            for i in 1:nc
                σp_vals[i] = C_tensor ⊙ plastic_work[i].εp
            end
            σp_field = CellField(σp_vals, Ω)

            a(u, v) = ∫((g_d ∘ dh) * (C_tensor ⊙ ε(u)) ⊙ ε(v)) * dΩ
            l(v) = ∫((g_d ∘ dh) * σp_field ⊙ ε(v)) * dΩ

            op = AffineFEOperator(a, l, U_u_cur, V_u)
            uh = solve(op)

            for i in 1:nc
                plastic_work[i].εp = plastic_sub_start[i].εp
                plastic_work[i].εp33 = plastic_sub_start[i].εp33
                plastic_work[i].εp_eq = plastic_sub_start[i].εp_eq
            end
            compute_elastoplastic_stress!(uh, dh, plastic_work, mat, Ω, true,
                                          cell_points=cell_points_cache)

            u_change = norm(get_free_dof_values(uh) - u_old) / (norm(get_free_dof_values(uh)) + 1e-10)
            if u_change < 1e-5
                break
            end
        end
    end

    for i in 1:nc
        plastic_states[i].εp = plastic_work[i].εp
        plastic_states[i].εp33 = plastic_work[i].εp33
        plastic_states[i].εp_eq = plastic_work[i].εp_eq
    end
    return uh
end

# ============================================================================
# Phase-field solve
# ============================================================================

function solve_phase_field(dh_prev, H_field, V_d, U_d, dΩ)
    a(d, φ) = ∫(
        (Gc / lc + 2.0 * H_field) * d * φ +
        Gc * lc * ∇(d) ⋅ ∇(φ)
    ) * dΩ
    b(φ) = ∫(2.0 * H_field * φ) * dΩ

    op = AffineFEOperator(a, b, U_d, V_d)
    dh_solved = solve(op)

    d_new = get_free_dof_values(dh_solved)
    d_old = get_free_dof_values(dh_prev)
    for i in eachindex(d_new)
        if !isfinite(d_new[i])
            d_new[i] = d_old[i]
        end
        d_new[i] = clamp(max(d_new[i], d_old[i]), 0.0, 1.0)
    end
    return FEFunction(U_d, d_new)
end

# ============================================================================
# Plotting / output helpers
# ============================================================================

function plot_mesh_png(Ω, output_dir)
    cell_coords = get_cell_coordinates(Ω)
    nc = num_cells(Ω)

    p = plot(aspect_ratio=:equal, size=(650, 650), dpi=220,
             xlabel="x (mm)", ylabel="y (mm)",
             title="Double-notched ductile specimen mesh ($(nc) elements)",
             legend=false, grid=false)

    for i in 1:nc
        verts = cell_coords[i]
        xs = [v[1] for v in verts]; push!(xs, xs[1])
        ys = [v[2] for v in verts]; push!(ys, ys[1])
        plot!(p, xs, ys, linecolor=:black, linewidth=0.15, label="")
    end

    savefig(p, joinpath(output_dir, "mesh_double_notched.png"))
    closeall()
end

function plot_cell_scalar_png(cell_coords, cell_values::Vector{Float64};
    title::String, out_file::String, cmap=:jet, dpi::Int=180,
    vmin::Union{Nothing,Float64}=nothing, vmax::Union{Nothing,Float64}=nothing)

    cmin = isnothing(vmin) ? minimum(cell_values) : vmin
    cmax = isnothing(vmax) ? maximum(cell_values) : vmax
    if !(cmax > cmin)
        cmax = cmin + 1e-12
    end

    p = plot(aspect_ratio=:equal, size=(650, 650), dpi=dpi,
             xlabel="x (mm)", ylabel="y (mm)", title=title, legend=false)
    grad = cgrad(cmap)

    for i in eachindex(cell_coords)
        verts = cell_coords[i]
        xs = [v[1] for v in verts]
        ys = [v[2] for v in verts]
        val = isfinite(cell_values[i]) ? cell_values[i] : cmin
        t = clamp((val - cmin) / (cmax - cmin), 0.0, 1.0)
        plot!(p, Shape(xs, ys), fillcolor=grad[t], linecolor=:transparent, linewidth=0.0, label="")
    end
    savefig(p, out_file)
    closeall()
end

function save_snapshot(
    tag::String,
    uh,
    dh,
    plastic_states,
    H_prev,
    Ω,
    V_dg0,
    V_proj,
    dΩ,
    cell_points_cache,
    output_dir;
    step::Int,
    load::Float64,
    qoi_force_kN::Float64,
    milestone_value::Float64
)
    snapshot_dir = joinpath(output_dir, "snapshots")
    mkpath(snapshot_dir)
    epspeq_vals = [ps.εp_eq for ps in plastic_states]
    epsp_vals = [ps.εp for ps in plastic_states]
    epsp33_vals = [ps.εp33 for ps in plastic_states]
    H_field = project_to_fe_space(H_prev, V_proj, dΩ)
    epspeq_field = array_to_fe_field(epspeq_vals, V_dg0, V_proj, dΩ)
    epsp_field = CellField(epsp_vals, Ω)
    epsp33_field = CellField(epsp33_vals, Ω)
    nc = length(plastic_states)

    writevtk(
        Ω,
        joinpath(snapshot_dir, tag),
        cellfields=[
            "d" => dh,
            "u" => uh,
            "plastic_strain" => epsp_field,
            "plastic_strain_33" => epsp33_field,
            "epspeq" => epspeq_field,
            "H" => H_field,
            "cycle" => CellField(fill(0.0, nc), Ω),
            "step" => CellField(fill(Float64(step), nc), Ω),
            "load" => CellField(fill(load, nc), Ω),
            "qoi_force_kN" => CellField(fill(qoi_force_kN, nc), Ω),
            "milestone" => CellField(fill(milestone_value, nc), Ω)
        ],
        nsubcells=4
    )
end

function plot_force_displacement(u_hist, f_hist, output_dir)
    p = plot(
        u_hist, f_hist,
        linewidth=2.2,
        color=:blue,
        xlabel="Displacement u (mm)",
        ylabel="Reaction force (kN)",
        title="Double-notched ductile specimen | load-displacement",
        legend=false,
        grid=true,
        size=(700, 500),
        dpi=220
    )
    savefig(p, joinpath(output_dir, "F_u_curve_double_notched.png"))
    closeall()
end

# ============================================================================
# Main simulation
# ============================================================================

function run_case(geom::GeometryCase)
    case_tag = geom.tag
    case_dir = joinpath(OUTPUT_DIR, case_tag)
    mkpath(case_dir)

    mesh_file = create_mesh(case_dir, geom)
    model_mesh = GmshDiscreteModel(mesh_file)
    if CLI.visualization
        writevtk(model_mesh, joinpath(case_dir, "mesh"))
        plot_mesh_png(Triangulation(model_mesh), case_dir)
    end

    mat = create_material()
    C_tensor = elastic_stiffness_plane_strain(mat)
    Δu = geom.u_total / N_STEPS

    println("\n" * "=" ^ 78)
    println("Ductile double-notched specimen | $(geom.title)")
    println("=" ^ 78)
    @printf("  E = %.1f MPa, ν = %.4f\n", mat.E, mat.ν)
    @printf("  G = %.1f MPa, K = %.1f MPa\n", mat.G, mat.λ + 2.0 * mat.G / 3.0)
    @printf("  σy = %.1f MPa, H = %.1f MPa (paper)\n", mat.σy0, mat.H_hard)
    @printf("  Gc = %.3f MPa·mm, l = %.4f mm\n", Gc, lc)
    @printf("  u_total = %.4f mm, Δu = %.3e mm, steps = %d\n", geom.u_total, Δu, N_STEPS)
    @printf("  notch radii = %.2f / %.2f mm\n", geom.bl_radius, geom.tr_radius)
    @printf("  loading scales: top = 1.00u, left = %.2fu\n", geom.left_scale)
    println("=" ^ 78)

    run_t0 = time()
    model = GmshDiscreteModel(mesh_file)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)

    Γ_top = BoundaryTriangulation(model, tags="top")
    Γ_left = BoundaryTriangulation(model, tags="left")
    dΓ_top = Measure(Γ_top, 2)
    dΓ_left = Measure(Γ_left, 2)
    n_top = get_normal_vector(Γ_top)
    n_left = get_normal_vector(Γ_left)

    order = 1

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["bottom", "right", "top", "left"],
                      dirichlet_masks=[(true, true), (true, true), (true, true), (true, true)])

    reffe_d = ReferenceFE(lagrangian, Float64, order)
    V_d = TestFESpace(model, reffe_d, conformity=:H1)
    U_d = TrialFESpace(V_d)

    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = FESpace(model, reffe_dg0, conformity=:L2)
    reffe_L2 = ReferenceFE(lagrangian, Float64, order)
    V_L2 = FESpace(model, reffe_L2, conformity=:L2)

    quad_cache = CellQuadrature(Ω, 2)
    cell_points_cache = get_cell_points(quad_cache)

    nc = num_cells(Ω)
    println("\n  cells = $(nc)")

    plastic_states = [PlasticStateData() for _ in 1:nc]
    H_prev = CellState(0.0, dΩ)

    dh = interpolate_everywhere(x -> 0.0, U_d)
    u_bottom_init(x) = VectorValue(0.0, 0.0)
    u_right_init(x) = VectorValue(0.0, 0.0)
    u_top_init(x) = VectorValue(0.0, 0.0)
    u_left_init(x) = VectorValue(0.0, 0.0)
    U_u_init = TrialFESpace(V_u, [u_bottom_init, u_right_init, u_top_init, u_left_init])
    uh = zero(U_u_init)

    disp_history = Float64[]
    force_history = Float64[]
    dmax_history = Float64[]
    epspeq_history = Float64[]
    time_history = Float64[]
    u_prev = 0.0
    peak_force_kN = -Inf
    stop_trigger_step::Union{Nothing,Int} = nothing
    snapshot_done = falses(length(SNAPSHOT_DAMAGE_LEVELS))

    println("\nStarting monotonic displacement loading...")

    for step in 1:N_STEPS
        u_val = step * Δu

        uh = solve_displacement(
            dh, plastic_states, mat, C_tensor,
            u_val, geom, V_u, dΩ, Ω, uh, u_prev;
            cell_points_cache=cell_points_cache
        )

        σ_cell, sse_cell = compute_elastoplastic_stress!(uh, dh, plastic_states, mat, Ω, false,
                                                         cell_points=cell_points_cache)

        ψ_total_arr = zeros(nc)
        for i in 1:nc
            ψp = plastic_strain_energy(plastic_states[i].εp_eq, mat)
            ψ_total_arr[i] = sse_cell[i] + ψp
        end
        ψ_dg = FEFunction(V_dg0, ψ_total_arr)
        update_state!(update_history_variable, H_prev, ψ_dg)

        for _ in 1:max_stagger
            d_old = copy(get_free_dof_values(dh))
            H_field = project_to_fe_space(H_prev, V_L2, dΩ)
            dh = solve_phase_field(dh, H_field, V_d, U_d, dΩ)
            d_new = get_free_dof_values(dh)
            res = norm(d_new - d_old) / (norm(d_new) + 1e-10)
            if res < stagger_tol
                break
            end
        end

        σ_field = CellField(σ_cell, Ω)
        reaction_top = sum(∫(n_top ⋅ σ_field) * dΓ_top)
        reaction_left = sum(∫(n_left ⋅ σ_field) * dΓ_left)
        F_y = (reaction_top[2] + reaction_left[2]) * THICKNESS
        F_y_kN = abs(F_y) / 1000.0

        d_max = maximum(get_free_dof_values(dh))
        epspeq_max = maximum(ps.εp_eq for ps in plastic_states)
        elapsed_s = time() - run_t0

        push!(disp_history, u_val)
        push!(force_history, F_y_kN)
        push!(dmax_history, d_max)
        push!(epspeq_history, epspeq_max)
        push!(time_history, elapsed_s)

        peak_force_kN = max(peak_force_kN, F_y_kN)

        if step % 10 == 0 || step <= 5
            @printf("  step %4d/%d | u = %.5f mm | F = %.4f kN | d_max = %.4f | epspeq_max = %.5f\n",
                    step, N_STEPS, u_val, F_y_kN, d_max, epspeq_max)
        end

        for (i, threshold) in enumerate(SNAPSHOT_DAMAGE_LEVELS)
            if CLI.snapshots && !snapshot_done[i] && d_max >= threshold
                tag = @sprintf("milestone_d%03d", round(Int, 100 * threshold))
                save_snapshot(tag, uh, dh, plastic_states, H_prev, Ω, V_dg0, V_L2, dΩ,
                              cell_points_cache, case_dir; step=step, load=u_val,
                              qoi_force_kN=F_y_kN, milestone_value=threshold)
                snapshot_done[i] = true
            end
        end

        if isnothing(stop_trigger_step)
            if peak_force_kN > 1e-8 && d_max >= 0.99 && F_y_kN <= STOP_FORCE_RATIO * peak_force_kN
                stop_trigger_step = step
                @printf("  reaction dropped below %.1f%% of peak at step %d; stopping after %d more steps\n",
                        100 * STOP_FORCE_RATIO, step, STOP_AFTER_STEPS)
            end
        elseif step >= stop_trigger_step + STOP_AFTER_STEPS
            println("  post-peak tail completed, stopping.")
            u_prev = u_val
            break
        end

        u_prev = u_val
    end

    csv_file = joinpath(case_dir, "force_displacement.csv")
    open(csv_file, "w") do io
        println(io, "step,time_s,u_mm,F_kN,d_max,epspeq_max")
        for i in eachindex(disp_history)
            @printf(io, "%d,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    i, time_history[i], disp_history[i], force_history[i], dmax_history[i], epspeq_history[i])
        end
    end

    if CLI.visualization
        plot_force_displacement(disp_history, force_history, case_dir)
    end
    if CLI.snapshots
        save_snapshot("final", uh, dh, plastic_states, H_prev, Ω, V_dg0, V_L2, dΩ,
                      cell_points_cache, case_dir; step=length(disp_history),
                      load=disp_history[end], qoi_force_kN=force_history[end],
                      milestone_value=-1.0)
    end

    println("\nOutputs saved to: $(case_dir)")
    return (
        case_tag = geom.tag,
        H_hard = mat.H_hard,
        runtime_s = time() - run_t0,
        f_peak = isempty(force_history) ? 0.0 : maximum(force_history),
        d_final = isempty(dmax_history) ? 0.0 : dmax_history[end],
        epspeq_final = isempty(epspeq_history) ? 0.0 : epspeq_history[end],
        n_steps = length(disp_history),
        bl_radius = geom.bl_radius,
        tr_radius = geom.tr_radius,
        u_total = geom.u_total,
        left_scale = geom.left_scale,
    )
end

function run_simulation()
    mkpath(OUTPUT_DIR)
    rows = NamedTuple[]
    for geom in CASES
        push!(rows, run_case(geom))
    end

    summary_csv = joinpath(OUTPUT_DIR, "scan_summary.csv")
    open(summary_csv, "w") do io
        println(io, "case_tag,H_hard_MPa,bl_radius_mm,tr_radius_mm,u_total_mm,left_scale,runtime_s,n_steps,f_peak_kN,d_final,epspeq_final")
        for row in rows
            @printf(io, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%d,%.8f,%.8f,%.8f\n",
                    row.case_tag, row.H_hard, row.bl_radius, row.tr_radius, row.u_total, row.left_scale, row.runtime_s, row.n_steps,
                    row.f_peak, row.d_final, row.epspeq_final)
        end
    end
    println("\nScan summary saved: $(summary_csv)")
end

function main()
    run_simulation()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
