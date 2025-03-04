using LinearAlgebra, HDF5, JLD2, Glob, JSON, ProgressMeter, Random, SimilaritySearch, Printf, Dates, StatsBase, Statistics

function load_database_f16(file; key, blocksize=0, normalize=false)::Matrix{Float16} # SISAP25 datasets are already normalized
    @info "==== loading $file"
    if blocksize == 0
        jldopen(file) do f
            X = f[key]
            normalize && for c in eachcol(X) normalize!(c) end
            eltype(X) === Float16 ? X : Matrix{Float16}(X)
        end 
    else
        h5open(file) do f
            D = f[key]
            X = Matrix{Float16}(undef, size(D))
            for (i, c) in enumerate(eachcol(D))
                normalize && normalize!(c)
                X[:, i] .= c
            end
            
            X
        end
    end
end

function save_results(knns::Matrix, dists::Matrix, meta, resfile::AbstractString)
    mkpath(dirname(resfile))
    h5open(resfile, "w") do f
        f["knns"] = knns
        f["dists"] = dists
        A = attributes(f)
        A["algo"] = meta["algo"]
        A["buildtime"] = meta["buildtime"]
        A["optimtime"] = get(meta, "optimtime", 0.0)
        A["querytime"] = meta["querytime"]
        A["params"] = meta["params"]
        A["searchparams"] = meta["searchparams"]
        A["size"] = meta["size"]
    end
end

function pretty_params(params)
    params = replace(params, r"\s+"ism => " ")
    params = replace(params, r"\(.*?\)" => "")
    params = replace(params, "Float32" => "")
    params = replace(params, "minrecall" => "")
    params = replace(params, ":" => "")
    params = replace(params, "f0" => "")
    params = replace(params, r"\s+"ism => " ")
    params = replace(params, " tol " => " ")
    replace(params, r"\s+$"ism => "")
end

function run_search(
        idx::SearchGraph,
        ctx,
        queries::AbstractDatabase,
        k::Integer,
        meta,
        outdir,
        optimsearch;
        step=1.05,
        maxvisits=0,
        step_pow_list=-2:1:12
    )

    meta["optimtime"] = @elapsed bestlist = optimize_index!(idx, ctx, optimsearch, ksearch=k+2)
    @info bestlist[1]
    
    Δ_, maxvisits_ = idx.algo.Δ, idx.algo.maxvisits
    idx.algo.maxvisits = max(maxvisits, maxvisits_)

    # produces a result file for each search hyperparameters
    @info "Optimized search parameters:" idx.algo
    for i in step_pow_list
        Δ = Δ_ * step^i
        idx.algo.Δ = round(Δ; digits=3)
        β, maxvisits = idx.algo.bsize, idx.algo.maxvisits
        searchparams = pretty_params(@sprintf "k=%d optimsearch=%s Δ=%0.3f β=%03d k=%d maxv=%s" k string(optimsearch) Δ β k (maxvisits >= 10^8 ? "free" : string(maxvisits)))
        resfile = joinpath(outdir, "ABS $searchparams.h5")
        @info "searching $resfile"
        @info idx.algo
        meta["searchparams"] = searchparams
        GC.enable(false)
        querytime = @elapsed knns, dists = searchbatch(idx, ctx, queries, k)
        GC.enable(true)
        meta["querytime"] = querytime
        @info querytime
        save_results(knns, dists, meta, resfile)
    end

    idx.algo.maxvisits = maxvisits_
    idx.algo.Δ = Δ_
end


function task1(;
        benchmark,
        outdir,
        indexfile,
        maxvisits=0,
        optim,
        optimsearch,
        neighborhood=Neighborhood(SatNeighborhood(; nndist=0.001); logbase=1.3),
    )
    dist = NormalizedCosine_asf32()  # 1 - dot(·, ·)
    db = load_database_f16(benchmark.file, key="train") |> StrideMatrixDatabase
    queries = load_database_f16(benchmark.file, key=benchmark.queries) |> StrideMatrixDatabase

    ctx = SearchGraphContext(;
                hyperparameters_callback = OptimizeParameters(optim, verbose=false),
                neighborhood,
               )

    @info "indexing" 
    if isfile(indexfile)
        @info "loading index"
        G, m_ = loadindex(indexfile, db)
        buildtime = m_.buildtime
    else
        mkpath(dirname(indexfile))
        G = SearchGraph(; db, dist)
        buildtime = @elapsed G = index!(G, ctx)
        @time "saving index" saveindex(indexfile, G; meta=(; buildtime), store_db=false)
    end
    meta = Dict()
    meta["algo"] = "ABS"
    meta["buildtime"] = buildtime
    meta["params"] = pretty_params(string("fp16; ", optim, "; ", neighborhood))
    meta["size"] = length(db)
    run_search(G, ctx, queries, benchmark.k, meta, outdir, optimsearch; maxvisits)
end

function main_task1(; k::Int=30, outdir="results", optim=MinRecall(0.95), optimsearch=MinRecall(0.7), maxvisits=10^5)
    for file in glob("data/benchmark-dev-*.h5")
        name = replace(basename(file), ".h5" => "")
        indexfile = joinpath(outdir, "index-$(name).jl2")
        benchmark = (; file, k, queries="otest/queries")
        task1(; benchmark, indexfile, outdir=joinpath(outdir, "$name-otest"), optim, optimsearch, maxvisits)
    end
end

function task2(;
        optim,
        benchmark,
        outdir,
        indexfile,
        optimsearch,
        neighborhood=Neighborhood(SatNeighborhood(; nndist=0.001); logbase=1.3),
    )
    dist = NormalizedCosine_asf32()  # 1 - dot(·, ·)
    db = load_database_f16(benchmark.file, key="train") |> StrideMatrixDatabase

    ctx = SearchGraphContext(;
                hyperparameters_callback = OptimizeParameters(optim, verbose=false),
                neighborhood,
               )

    @info "indexing" 
    if isfile(indexfile)
        @info "loading index"
        G, m_ = loadindex(indexfile, db)
        buildtime = m_.buildtime
    else
        mkpath(dirname(indexfile))
        G = SearchGraph(; db, dist)
        buildtime = @elapsed G = index!(G, ctx)
        @time "saving index" saveindex(indexfile, G; meta=(; buildtime), store_db=false)
    end

    meta = Dict()
    meta["algo"] = "ABS"
    meta["buildtime"] = buildtime
    meta["params"] = pretty_params(string("fp16; ", optim, "; ", neighborhood))
    meta["size"] = length(db)
    meta["optimtime"] = @elapsed bestlist = optimize_index!(G, ctx, optimsearch, ksearch=benchmark.k+1)
    meta["querytime"] = @elapsed knns, dists = allknn(G, ctx, benchmark.k)
    meta["searchparams"] = "allknn algorithm"
    resfile = joinpath(outdir, "ABS " * meta["params"] * ".h5")
    save_results(knns, dists, meta, resfile) 
end

function main_task2(; k::Int=16, outdir="results", optim=MinRecall(0.95), optimsearch=MinRecall(0.8))
    for file in glob("data/benchmark-dev-*.h5")
        name = replace(basename(file), ".h5" => "")
        indexfile = joinpath(outdir, "index-$(name).jl2")
        benchmark = (; file, k)
        task2(; benchmark, indexfile, outdir=joinpath(outdir, "$name-allknn"), optim, optimsearch)
    end
end
