#!/usr/bin/env julia
# ============================================================================
# A phase-field length scale insensitive model for fatigue failure in brittle
# materials — AT1 baseline
#
# TPB benchmark based on:
#   kf05_Fatigue_AT1_LengthScaleInsens_ThreePointBending(g(s)2)-ls0.06.ipynb
#
# Runs the AT1 baseline case on the benchmark mesh with the notebook
# material parameters.
# ============================================================================

using GridapGmsh
using Gridap
using Gridap.Geometry
using Gridap.TensorValues
using Gridap.Arrays
using Gridap.ReferenceFEs
using Plots
using LinearAlgebra
using LineSearches: BackTracking
using DelimitedFiles
using Printf
using Gmsh: gmsh

ENV["GKSwstype"] = "100"
gr(show = false)

function parse_cli(args)
    output_dir = normpath(joinpath(
        @__DIR__, "..", "outputs", "results_cpu", "elastic_fatigue_TPB_AT1", "julia"
    ))
    benchmark = "AT1"
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
    benchmark == "AT1" || error("Supported benchmark: AT1")
    return (output_dir=abspath(output_dir), benchmark=benchmark,
            visualization=visualization, snapshots=snapshots)
end

const CLI = parse_cli(ARGS)

# ============================================================================
# Base benchmark parameters from the notebook / paper
# ============================================================================
const E_mat = 20.8
const ν_mat = 0.3
const Gc = 5e-4
const ls = 0.06
const η = 1e-15

const L = 8.0
const LL = 0.475 * L
const LR = 0.525 * L
const H = 2.0
const CH = 0.4
const CW = 0.2

const hfc = 0.03 / 100
const hf = 0.03 / 5
const h = 50 * hf

const σc = 0.2
const m = 3 * E_mat * Gc / (4 * ls * σc^2)
const ψ_Crit = σc^2 / (2 * E_mat)
const kf = 0.5
const αT = ψ_Crit / kf

const T_TOTAL = 1.0
const N_CYCLES = 400
const N_STEPS_PER_CYCLE = 10
const INNER_MAX = 5
const REL_TOL = 1e-6
const DAMAGE_TOL = 0.05
const U_MAX = 0.025 / 2
const R_CASES = [0.0]
const OUTPUT_ROOT = CLI.output_dir
const SAVE_CYCLE_INTERVAL = 40
const SNAPSHOT_MILESTONES = [
    (0.05, "delta_a_0.05mm"), (0.10, "delta_a_0.10mm"),
    (0.20, "delta_a_0.20mm"), (0.40, "delta_a_0.40mm"),
    (0.60, "delta_a_0.60mm"),
]

# ============================================================================
# Constitutive definitions copied from notebook
# ============================================================================
function ElasFourthOrderConstTensor(E, ν, PlanarState)
    if PlanarState == 1
        C1111 = E / (1 - ν * ν)
        C1122 = (ν * E) / (1 - ν * ν)
        C1112 = 0.0
        C2222 = E / (1 - ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    elseif PlanarState == 2
        C1111 = (E * (1 - ν * ν)) / ((1 + ν) * (1 - ν - 2 * ν * ν))
        C1122 = (ν * E) / (1 - ν - 2 * ν * ν)
        C1112 = 0.0
        C2222 = (E * (1 - ν)) / (1 - ν - 2 * ν * ν)
        C2212 = 0.0
        C1212 = E / (2 * (1 + ν))
    else
        error("Unsupported planar state")
    end
    return SymFourthOrderTensorValue(C1111, C1112, C1122,
                                     C1112, C1212, C2212,
                                     C1122, C2212, C2222)
end

const C_mat = ElasFourthOrderConstTensor(E_mat, ν_mat, 2)
const I2 = SymTensorValue{2,Float64}(1.0, 0.0, 1.0)
const I4 = I2 ⊗ I2
const I4_sym = one(SymFourthOrderTensorValue{2,Float64})
const P_vol = 0.5 * I4
const P_dev = I4_sym - P_vol

function degDer(ϕ)
    return (m * ϕ) / (0.25 * m^2 * ϕ^4 - 0.5 * m^2 * ϕ^2 + 0.25 * m^2 -
                       m * ϕ^4 + m * ϕ^2 + ϕ^4)
end

function phase_degradation(s)
    return (s^2) / (s^2 + m * (1 - s) * (0.5 + 0.5 * s))
end

function σfun(ε_total, ε_prev, s_prev)
    σ_elas = C_mat ⊙ ε_total
    gs = phase_degradation(s_prev)
    if tr(ε_prev) >= 0
        σ = gs * σ_elas
    else
        σ = gs * (P_dev ⊙ σ_elas) + (P_vol ⊙ σ_elas)
    end
    return σ
end

function ψPos(ε_in, Fdg)
    σ_elas = C_mat ⊙ ε_in
    if tr(ε_in) >= 0
        return 0.5 * (ε_in ⊙ σ_elas) / Fdg
    else
        return 0.5 * ((P_dev ⊙ σ_elas) ⊙ (P_dev ⊙ ε_in)) / Fdg
    end
end

function ψPosOrg(ε_in)
    σ_elas = C_mat ⊙ ε_in
    if tr(ε_in) >= 0
        return 0.5 * (ε_in ⊙ σ_elas)
    else
        return 0.5 * ((P_dev ⊙ σ_elas) ⊙ (P_dev ⊙ ε_in))
    end
end

function new_EnergyState(ψPlusPrev_in, ψhPos_in)
    ψPlus_in = ψhPos_in
    ψPlus_out = ψPlus_in >= ψPlusPrev_in ? ψPlus_in : ψPlusPrev_in
    return true, ψPlus_out
end

function project_field(q, model, dΩ, order)
    reffe = ReferenceFE(lagrangian, Float64, order)
    V = FESpace(model, reffe, conformity = :L2)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * q) * dΩ
    return solve(AffineFEOperator(a, b, V, V))
end

FatigueDegrad(αb) = @. ifelse(αb >= αT, ((2 * αT) / (αb + αT))^2, 1.0)

function FatiguehistoryVariable(ψhPos_in, αPrev, αbPrev)
    α = ψhPos_in
    αb = α >= αPrev ? αbPrev + abs(α - αPrev) : αbPrev
    return αb
end

function αNC(s_in, ψhPos_in)
    return phase_degradation(s_in) * ψhPos_in
end

# ============================================================================
# Mesh
# ============================================================================
function create_mesh(mesh_file::String)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("BeamWithNotchSymThreePtBending")

    gmsh.model.geo.addPoint((L/2)+(CW/2), 0.0, 0.0, hf, 1)
    gmsh.model.geo.addPoint(L, 0.0, 0.0, h, 2)
    gmsh.model.geo.addPoint(L, H, 0.0, h, 3)
    gmsh.model.geo.addPoint(LR, H, 0.0, hf, 4)
    gmsh.model.geo.addPoint(LL, H, 0.0, hf, 5)
    gmsh.model.geo.addPoint(0.0, H, 0.0, h, 6)
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h, 7)
    gmsh.model.geo.addPoint((L/2)-(CW/2), 0.0, 0.0, hf, 8)
    gmsh.model.geo.addPoint((L/2), CH, 0.0, hfc, 9)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 1, 9)

    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8,9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.addPhysicalGroup(1, [4], 1)
    gmsh.model.addPhysicalGroup(0, [7], 2)
    gmsh.model.addPhysicalGroup(0, [2], 3)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.setPhysicalName(1, 1, "LoadLine")
    gmsh.model.setPhysicalName(0, 2, "LeftSupport")
    gmsh.model.setPhysicalName(0, 3, "RightSupport")

    gmsh.model.mesh.field.add("Box", 10)
    gmsh.model.mesh.field.setNumber(10, "VIn", hf)
    gmsh.model.mesh.field.setNumber(10, "VOut", h)
    gmsh.model.mesh.field.setNumber(10, "XMin", (L/2) - CW)
    gmsh.model.mesh.field.setNumber(10, "XMax", (L/2) + CW)
    gmsh.model.mesh.field.setNumber(10, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(10, "YMax", H)
    gmsh.model.mesh.field.setAsBackgroundMesh(10)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)

    n_nodes = length(gmsh.model.mesh.getNodes()[1])
    n_tris = length(gmsh.model.mesh.getElementsByType(2)[1])
    println("  nodes = $(n_nodes)")
    println("  triangles = $(n_tris)")
    println("  hf = $(round(hf, digits=5)) mm, h_tip = $(round(hfc, digits=5)) mm, h_coarse = $(round(h, digits=5)) mm")
    gmsh.finalize()
end

function plot_mesh(mesh_file::String, output_dir::String)
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
    p = plot(aspect_ratio=:equal, size=(900,300), dpi=200, xlabel="x (mm)", ylabel="y (mm)",
             legend=false, grid=false, title="TPB mesh")
    for i in 1:(length(tri_nodes) ÷ 3)
        ids = tri_nodes[3*(i-1)+1:3*i]
        pts = [node_map[Int(id)] for id in ids]
        xs = [pt[1] for pt in pts]
        ys = [pt[2] for pt in pts]
        plot!(p, Shape(xs, ys), fillcolor=:white, linecolor=:gray55, linewidth=0.12)
    end
    savefig(p, joinpath(output_dir, "mesh.png"))
    gmsh.finalize()
end

function save_damage_plot(Ω, sh, output_file::String; title_str::String)
    cell_coords = get_cell_coordinates(Ω)
    qc = CellQuadrature(Ω, 2)
    vals = evaluate(sh, get_cell_points(qc))
    cell_vals = [1.0 - mean(vals[i]) for i in eachindex(vals)]
    grad = cgrad(:jet)
    p = plot(aspect_ratio=:equal, size=(900,300), dpi=200, xlabel="x (mm)", ylabel="y (mm)",
             legend=false, grid=false, title=title_str, framestyle=:box)
    for i in eachindex(cell_coords)
        xs = [v[1] for v in cell_coords[i]]
        ys = [v[2] for v in cell_coords[i]]
        ξ = clamp(cell_vals[i], 0.0, 1.0)
        plot!(p, Shape(xs, ys), fillcolor=grad[ξ], linecolor=:transparent, linewidth=0.0)
    end
    heatmap!(p, [NaN NaN; NaN NaN], c=:jet, clims=(0,1), colorbar_title="d")
    savefig(p, output_file)
end

function write_scientific_snapshot(Ω, snapshot_dir::String, name::String,
                                   uh, sh, ψhistory, αh, αbh, fdeg;
                                   overwrite::Bool=false)
    mkpath(snapshot_dir)
    base = joinpath(snapshot_dir, name)
    if !overwrite && (isfile(base * ".vtu") || isfile(base * ".pvtu"))
        return false
    end
    writevtk(
        Ω, base,
        cellfields=[
            "uh" => uh, "s" => sh, "damage" => (1.0 - sh),
            "history" => ψhistory, "alpha" => αh,
            "alphabar" => αbh, "Fdeg" => fdeg,
        ],
    )
    return true
end

# ============================================================================
# Case runner
# ============================================================================
function run_case(model, case_label::String, R_ratio::Float64, output_dir::String, at_model::Symbol)
    println("\n" * "-"^78)
    println("Running case: $(case_label)")
    println("-"^78)

    order = 1
    reffe_PF = ReferenceFE(lagrangian, Float64, order)
    V0_PF = TestFESpace(model, reffe_PF; conformity=:H1)
    U_PF = TrialFESpace(V0_PF)
    sh = zero(V0_PF)

    reffe_Disp = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V0_Disp = TestFESpace(model, reffe_Disp; conformity=:H1,
        dirichlet_tags=["LeftSupport","RightSupport","LoadLine"],
        dirichlet_masks=[(false,true), (true,true), (false,true)])
    uh = zero(V0_Disp)

    degree = 2 * order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    labels = get_face_labeling(model)
    load_tag_id = get_tag_from_name(labels, "LoadLine")
    Γ_Load = BoundaryTriangulation(model, tags = load_tag_id)
    dΓ_Load = Measure(Γ_Load, degree)
    n_Γ_Load = get_normal_vector(Γ_Load)

    nls = NLSolver(show_trace=false, method=:newton, linesearch=BackTracking(), iterations=10)
    solver = FESolver(nls)

    function crack_tip_tracker_local(sh_in, tol)
        damage_ids = Int[]
        coords = get_node_coordinates(Ω)
        sVec = get_free_dof_values(sh_in)
        for i in eachindex(sVec)
            if sVec[i] <= tol
                push!(damage_ids, i)
            end
        end
        isempty(damage_ids) && return nothing
        xloc = Float64[]
        yloc = Float64[]
        for idx in damage_ids
            push!(xloc, coords[idx][1])
            push!(yloc, coords[idx][2])
        end
        locx, index = findmax(xloc)
        return (locx, yloc[index])
    end

    function stepPhaseField(x0, ψPlusPrev_in)
        U_PF_loc = TrialFESpace(V0_PF)
        res_at1(s, ϕ) = ∫((3/4)*Gc*ls*∇(ϕ)⋅∇(s) + (degDer∘(s))*ψPlusPrev_in*ϕ - ((3/8)*Gc/ls)*ϕ) * dΩ
        res_at2(s, ϕ) = ∫(Gc*ls*∇(ϕ)⋅∇(s) + (degDer∘(s))*ψPlusPrev_in*ϕ + (Gc/ls)*s*ϕ - (Gc/ls)*ϕ) * dΩ
        op = FEOperator(at_model == :AT1 ? res_at1 : res_at2, U_PF_loc, V0_PF)
        sh_out = FEFunction(U_PF_loc, x0)
        sh_out, cache = solve!(sh_out, solver, op, nothing)
        return sh_out, get_free_dof_values(sh_out), cache
    end

    function stepDisp(uh_in, sh_in, vApp)
        uApp1(x) = VectorValue(0.0, 0.0)
        uApp2(x) = VectorValue(0.0, 0.0)
        uApp3(x) = VectorValue(0.0, -vApp)
        U_Disp = TrialFESpace(V0_Disp, [uApp1, uApp2, uApp3])
        a_Disp(u,v) = ∫((ε(v) ⊙ (σfun ∘ (ε(u), ε(uh_in), sh_in)))) * dΩ
        b_Disp(v) = 0.0
        op_Disp = AffineFEOperator(a_Disp, b_Disp, U_Disp, V0_Disp)
        return solve(op_Disp)
    end

    count = 0
    cycle = 0.0
    tp = (T_TOTAL / N_CYCLES) / N_STEPS_PER_CYCLE
    t = 0.0
    dc = 1 / N_STEPS_PER_CYCLE
    u_min = R_ratio * U_MAX
    ψPlusPrev = CellState(ψ_Crit, dΩ)
    sPrev = CellState(1.0, dΩ)
    sh = project_field(sPrev, model, dΩ, order)
    x0_PF = ones(Float64, num_free_dofs(V0_PF))
    αPrev = CellState(0.0, dΩ)
    αbPrev = CellState(0.0, dΩ)
    αhPlusPrev = project_field(αPrev, model, dΩ, order)
    αbhPlusPrev = project_field(αbPrev, model, dΩ, order)
    FdhPrev = FatigueDegrad ∘ αbhPlusPrev

    Load = Float64[0.0]
    Displacement = Float64[0.0]
    time_hist = Float64[0.0]
    wall_time_hist = Float64[0.0]
    AppDisplacement = Float64[0.0]
    αEnergy = Float64[0.0]
    αDegrad = Float64[1.0]
    FracEnergy = Float64[0.0]
    NoofCycles = Float64[0.0]
    Xloccrack = Float64[L/2]
    Yloccrack = Float64[CH]
    run_t0 = time()
    snapshot_dir = joinpath(output_dir, "snapshots")
    snapshot_io_s = 0.0

    while t < T_TOTAL
        t += tp
        period = T_TOTAL / N_CYCLES
        amp = U_MAX - u_min
        vAppC = u_min + abs(4 * amp / period * abs((((t - period/4) % period) + period) % period - period/2) - amp)
        vApp = round(vAppC, digits=6)
        count += 1
        cycle += dc
        FdhPrev = FatigueDegrad ∘ αbhPlusPrev

        for inner = 1:INNER_MAX
            ψhPlusPrev = project_field(ψPlusPrev, model, dΩ, order)
            if at_model == :AT1
                rel_num = abs(sum(∫((3/4)*Gc*ls*∇(sh)⋅∇(sh) + (degDer∘(sh))*ψhPlusPrev*sh) * dΩ -
                                  ∫(((3/8)*Gc/ls)*sh) * dΩ))
                rel_den = abs(sum(∫(((3/8)*Gc/ls)*sh) * dΩ)) + 1e-12
            else
                rel_num = abs(sum(∫(Gc*ls*∇(sh)⋅∇(sh) + (degDer∘(sh))*ψhPlusPrev*sh + (Gc/ls)*sh*sh) * dΩ -
                                  ∫((Gc/ls)*sh) * dΩ))
                rel_den = abs(sum(∫((Gc/ls)*sh) * dΩ)) + 1e-12
            end
            RelErr = rel_num / rel_den
            sh, x0_PF, _ = stepPhaseField(x0_PF, ψhPlusPrev)
            uh = stepDisp(uh, sh, vApp)
            ψhPos_in = ψPos ∘ (ε(uh), FdhPrev)
            update_state!(new_EnergyState, ψPlusPrev, ψhPos_in)
            RelErr < REL_TOL && break
        end

        tip = crack_tip_tracker_local(sh, DAMAGE_TOL)
        if tip === nothing
            push!(Xloccrack, 0.5 * L)
            push!(Yloccrack, CH)
        else
            push!(Xloccrack, tip[1])
            push!(Yloccrack, tip[2])
        end

        ψhPosOrg_in = ψPosOrg ∘ (ε(uh))
        αPrev = αNC ∘ (sh, ψhPosOrg_in)
        αbPrev = FatiguehistoryVariable ∘ (αPrev, αhPlusPrev, αbhPlusPrev)
        αhPlusPrev = project_field(αPrev, model, dΩ, order)
        αbhPlusPrev = project_field(αbPrev, model, dΩ, order)
        FdhPrev = FatigueDegrad ∘ (αbhPlusPrev)

        if at_model == :AT1
            FracEnergyExp = sum(∫((3/8) * ((1-sh)/ls + ls*(∇(sh) ⋅ ∇(sh)))) * dΩ)
        else
            FracEnergyExp = sum(∫(0.5 * ((1-sh)*(1-sh)/ls + ls*(∇(sh) ⋅ ∇(sh)))) * dΩ)
        end
        αbPrevVal = evaluate(αbPrev, VectorValue(0.5 * L, 2 * CH))
        Fdegrad = evaluate(FdhPrev, VectorValue(0.5 * L, 2 * CH))

        push!(time_hist, t)
        push!(wall_time_hist, time() - run_t0 - snapshot_io_s)
        push!(AppDisplacement, vApp)
        push!(NoofCycles, cycle)
        push!(αEnergy, αbPrevVal)
        push!(αDegrad, Fdegrad)
        push!(FracEnergy, FracEnergyExp)

        Node_Force = sum(∫(n_Γ_Load ⋅ (σfun ∘ (ε(uh), ε(uh), sh))) * dΓ_Load)
        push!(Load, -Node_Force[2])
        push!(Displacement, vApp)

        if CLI.snapshots
            snapshot_t0 = time()
            δa = Yloccrack[end] - CH
            pending = [
                (milestone, name) for (milestone, name) in SNAPSHOT_MILESTONES
                if δa >= milestone &&
                   !isfile(joinpath(snapshot_dir, name) * ".vtu") &&
                   !isfile(joinpath(snapshot_dir, name) * ".pvtu")
            ]
            if !isempty(pending)
                ψhistory = project_field(ψPlusPrev, model, dΩ, order)
                for (_, name) in pending
                    write_scientific_snapshot(
                        Ω, snapshot_dir, name, uh, sh, ψhistory,
                        αhPlusPrev, αbhPlusPrev, FdhPrev,
                    )
                end
            end
            snapshot_io_s += time() - snapshot_t0
        end

        if mod(count, N_STEPS_PER_CYCLE) == 0
            cyc = round(Int, cycle)
            if CLI.visualization && (cyc % SAVE_CYCLE_INTERVAL == 0 || cyc == 1 || cyc == N_CYCLES)
                writevtk(Ω, joinpath(output_dir, @sprintf("cycle_%04d", cyc)),
                         cellfields = ["uh" => uh, "s" => sh, "epsi" => ε(uh), "Fdeg" => FdhPrev, "alphabar" => αbhPlusPrev])
            end
            @printf("  cycle %4d | Δa = %.4f mm | Fmax = %.5f | f(P)=%.4f\n",
                    cyc, Yloccrack[end] - CH, maximum(Load[max(1,end-N_STEPS_PER_CYCLE+1):end]), Fdegrad)
        end
    end

    if CLI.visualization
        save_damage_plot(Ω, sh, joinpath(output_dir, "damage_final.png");
                         title_str = "Length-scale insensitive TPB fatigue, $(case_label)")
    end
    if CLI.snapshots
        ψhistory = project_field(ψPlusPrev, model, dΩ, order)
        write_scientific_snapshot(
            Ω, snapshot_dir, "final", uh, sh, ψhistory,
            αhPlusPrev, αbhPlusPrev, FdhPrev; overwrite=true,
        )
    end
    open(joinpath(output_dir, "history.csv"), "w") do io
        println(io, "step,time,wall_time_s,cycle,u_app,load,x_tip,y_tip,delta_a,alpha_bar,f_degrad,frac_energy")
        for i in eachindex(time_hist)
            δa = Yloccrack[i] - CH
            println(io, join([i-1, time_hist[i], wall_time_hist[i], NoofCycles[i], AppDisplacement[i], Load[i],
                              Xloccrack[i], Yloccrack[i], δa, αEnergy[i], αDegrad[i], FracEnergy[i]], ","))
        end
    end
    return NoofCycles, Yloccrack .- CH, Load, Displacement
end

function main()
    mkpath(OUTPUT_ROOT)
    println("="^78)
    println("Length-scale insensitive TPB fatigue | AT1 baseline")
    println("="^78)
    @printf("  E = %.3f, nu = %.3f, Gc = %.6f, ls = %.4f\n", E_mat, ν_mat, Gc, ls)
    @printf("  alphaT = %.8f, U_MAX = %.5f, cycles = %d, steps/cycle = %d\n", αT, U_MAX, N_CYCLES, N_STEPS_PER_CYCLE)

    mesh_file = joinpath(OUTPUT_ROOT, "BeamWithNotchSymThreePtBending.msh")
    create_mesh(mesh_file)
    model = GmshDiscreteModel(mesh_file)
    CLI.visualization && plot_mesh(mesh_file, OUTPUT_ROOT)

    p_crack = CLI.visualization ? plot(xlabel="N", ylabel="Δa (mm)", title="TPB fatigue crack growth (AT1)", grid=true, size=(850,600), dpi=220) : nothing
    p_load = CLI.visualization ? plot(xlabel="u (mm)", ylabel="Load", title="TPB load-displacement (AT1)", grid=true, size=(850,600), dpi=220) : nothing

    R_ratio = R_CASES[1]
    for at_model in [:AT1]
        case_label = @sprintf("%s, R=%.1f", at_model, R_ratio)
        case_dir = joinpath(OUTPUT_ROOT, string(at_model))
        mkpath(case_dir)
        cycles, delta_a, Load, Disp = run_case(model, case_label, R_ratio, case_dir, at_model)
        if CLI.visualization
            plot!(p_crack, cycles, delta_a, linewidth=2, label=string(at_model))
            plot!(p_load, Disp, Load, linewidth=2, label=string(at_model))
        end
    end

    if CLI.visualization
        savefig(p_crack, joinpath(OUTPUT_ROOT, "a_N_curve.png"))
        savefig(p_load, joinpath(OUTPUT_ROOT, "F_u_curve.png"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
