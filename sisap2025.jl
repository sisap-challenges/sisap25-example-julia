using LinearAlgebra, HDF5, JLD2, Glob, JSON, ProgressMeter, Random, SimilaritySearch, Printf, Dates, StatsBase, Statistics, MultivariateStats

function i8!(X, min_, max_)
    c = 127.5 / (max_ - min_)
    Threads.@threads :static for i in CartesianIndices(X)
        X[i] = clamp(round((X[i] - min_) * c - 127.5; digits=0), -128, 127)
    end
    X
end

function load_pca_i8(file; trainkey="train", testkey="otest/queries", maxoutdim=192) # SISAP25 datasets are already normalized
    @info "==== loading $file"
    pca, train, min_, max_ = h5open(file) do f
        D = f[trainkey]
        n = size(D, 2)
        X = Matrix{Int8}(undef, maxoutdim, n)
        S = Iterators.Stateful(Iterators.partition(1:n, 10^5))
        L = popfirst!(S)
        B = D[:, L]
        @time "PCA" pca = fit(PCA, B; maxoutdim)
        b = predict(pca, B)
        min_, max_ = extrema(b)
        X[:, L] .= i8!(b, min_, max_)
        @showprogress dt=1 desc="PCA projection train data" for L in S
            B = D[:, L]
            X[:, L] .= i8!(predict(pca, B), min_, max_)
        end

        pca, X, min_, max_
    end

    test = h5open(file) do f
        D = f[testkey]
        n = size(D, 2)
        X = Matrix{Int8}(undef, maxoutdim, n)
        @showprogress dt=1 desc="PCA projection train data" for L in Iterators.partition(1:n, 10^3)
            B = D[:, L]
            X[:, L] .= i8!(predict(pca, B), min_, max_)
        end
        X
    end

    train, test, (pca, min_, max_), SqL2_asf32()
end

function load_pca_f16(file; trainkey="train", testkey="otest/queries", maxoutdim=192) # SISAP25 datasets are already normalized
    @info "==== loading $file"
    pca, train = h5open(file) do f
        D = f[trainkey]
        n = size(D, 2)
        X = Matrix{Float16}(undef, maxoutdim, n)
        S = Iterators.Stateful(Iterators.partition(1:n, 10^5))
        L = popfirst!(S)
        B = D[:, L]
        @time "PCA" pca = fit(PCA, B; maxoutdim)
        X[:, L] .= predict(pca, B)
        @showprogress dt=1 desc="PCA projection train data" for L in S
            B = D[:, L]
            X[:, L] .= predict(pca, B)
        end
        pca, X
    end

    test = h5open(file) do f
        D = f[testkey]
        n = size(D, 2)
        X = Matrix{Float16}(undef, maxoutdim, n)
        @showprogress dt=1 desc="PCA projection train data" for L in Iterators.partition(1:n, 10^3)
            B = D[:, L]
            X[:, L] .= predict(pca, B)
        end
        X
    end

    train, test, pca, SqL2_asf32()
end


function load_database_f16_(file; key, blocksize=0, normalize=false)::Matrix{Float16} # SISAP25 datasets are already normalized
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

function load_database(file; trainkey="train", testkey="otest/queries") # SISAP25 datasets are already normalized
    load_database_f16_(file; key=trainkey), load_database_f16_(file; key=testkey), nothing, NormalizedCosine_asf32()
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
        A["task"] = meta["task"]
        A["dataset"] = meta["dataset"]
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
        logbase,
        preprocessing,
        maxoutdim,
        neighborhood=Neighborhood(SatNeighborhood(; nndist=0.001); logbase),
    )
    if preprocessing === nothing || preprocessing == "none"
        dist = NormalizedCosine_asf32()  # 1 - dot(·, ·)
        X, Q, _, dist = load_database(benchmark.file)
        maxoutdim = size(X, 1)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=0.001); logbase)
    elseif preprocessing == "pca-f16"
        X, Q, _, dist = load_pca_f16(benchmark.file; maxoutdim)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=2.0); logbase)
    elseif preprocessing == "pca-i8"
        X, Q, _, dist = load_pca_i8(benchmark.file; maxoutdim)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=2.0); logbase)
    else
        error("Unknown preprocessing '$preprocessing'")
    end

    ctx = SearchGraphContext(;
                hyperparameters_callback = OptimizeParameters(optim,
                                                              verbose=false,
                                                              space=BeamSearchSpace(bsize_scale=(s=1.1, p1=0.25, p2=0.5, lower=2, upper=12))
                                                             ), # this should work fine with (relatively) low-cost distance functions, i.e. the structure costs more than the dist. evaluation
                                                             
                neighborhood,
               )

    @info "indexing" 
    #=if  isfile(indexfile)
        @info "loading index"
        G, m_ = loadindex(indexfile, db)
        buildtime = m_.buildtime
    else=#
        #mkpath(dirname(indexfile))
        G = SearchGraph(; db=StrideMatrixDatabase(X), dist)
        buildtime = @elapsed G = index!(G, ctx)
        #@time "saving index" saveindex(indexfile, G; meta=(; buildtime), store_db=false)
    #end
    meta = Dict()
    meta["algo"] = "ABS"
    meta["task"] = "task1"
    meta["dataset"] = let dataset = basename(outdir)
        dataset = replace(dataset, r"-[io]test" => "")
        dataset = replace(dataset, "benchmark-" => "")
        replace(dataset, "dev-" => "")
    end
    meta["buildtime"] = buildtime
    meta["params"] = pretty_params(string("$preprocessing $maxoutdim; ", optim, "; ", neighborhood))
    meta["size"] = size(X, 2)
    run_search(G, ctx, StrideMatrixDatabase(Q), benchmark.k, meta, outdir, optimsearch; maxvisits)
end

function main_task1(;
        k::Int=30, outdir="results",
        optim=MinRecall(0.97), optimsearch=MinRecall(0.75), maxvisits=10^4,
        preprocessing="none", logbase=1.3, maxoutdim=192)
    file="data/benchmark-eval-pubmed23.h5"
    name = replace(basename(file), ".h5" => "")
    indexfile = joinpath(outdir, "index-$(name).jl2")
    benchmark = (; file, k, queries="otest/queries")
    task1(; benchmark, indexfile, outdir=joinpath(outdir, "$name-otest"), optim, optimsearch, maxvisits, preprocessing, logbase, maxoutdim)
end

function task2(;
        optim,
        benchmark,
        outdir,
        indexfile,
        optimsearch,
        preprocessing,
        maxoutdim,
        logbase,
        neighborhood=nothing
    )

    if preprocessing === nothing || preprocessing == "none"
        dist = NormalizedCosine_asf32()  # 1 - dot(·, ·)
        X, _, _, dist = load_database(benchmark.file)
        maxoutdim = size(X, 1)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=0.001); logbase)
    elseif preprocessing == "pca-f16"
        X, _, _, dist = load_pca_f16(benchmark.file; maxoutdim)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=2.0); logbase)
    elseif preprocessing == "pca-i8"
        X, _, _, dist = load_pca_i8(benchmark.file; maxoutdim)
        neighborhood = Neighborhood(SatNeighborhood(; nndist=2.0); logbase)
    else
        error("Unknown preprocessing '$preprocessing'")
    end

    ctx = SearchGraphContext(;
                hyperparameters_callback = OptimizeParameters(optim,
                                                              verbose=true, 
                                                              space=BeamSearchSpace(bsize_scale=(s=1.1, p1=0.25, p2=0.5, lower=2, upper=12)) # this should work fine with (relatively) low-cost distance functions, i.e. the structure costs more than the dist. evaluation
                                                             ),
                parallel_block=12 * Threads.nthreads(),
                neighborhood,
               )

    @info "indexing" 
    #=if isfile(indexfile)
        @info "loading index"
        G, m_ = loadindex(indexfile, db)
        buildtime = m_.buildtime
    else=#
        #mkpath(dirname(indexfile))
        G = SearchGraph(; db=StrideMatrixDatabase(X), dist)
        buildtime = @elapsed G = index!(G, ctx)
    #    @time "saving index" saveindex(indexfile, G; meta=(; buildtime), store_db=false)
    #end

    meta = Dict()
    meta["algo"] = "ABS"
    meta["task"] = "task2"
    meta["dataset"] = let dataset = basename(outdir)
        dataset = replace(dataset, "-allknn" => "")
        dataset = replace(dataset, "benchmark-" => "")
        replace(dataset, "dev-" => "")
    end
    meta["buildtime"] = buildtime
    meta["params"] = pretty_params(string("$preprocessing $maxoutdim; ", optim, "; ", neighborhood))
    meta["size"] = size(X, 2)
    meta["optimtime"] = @elapsed bestlist = optimize_index!(G, ctx, optimsearch, ksearch=benchmark.k+1)
    meta["querytime"] = @elapsed knns, dists = allknn(G, ctx, benchmark.k+1)
    meta["searchparams"] = "allknn algorithm"
    resfile = joinpath(outdir, "ABS " * meta["params"] * ".h5")
    save_results(knns, dists, meta, resfile)
end

function main_task2(;
        k::Int=15,
        outdir="results",
        minrecall=0.95,
        minrecallsearch=0.82,
        preprocessing="none",
        logbase=1.5,
        maxoutdim=128
    )
    file = "data/benchmark-eval-gooaq.h5"
    name = replace(basename(file), ".h5" => "")
    indexfile = joinpath(outdir, "index-$(name).jl2")
    benchmark = (; file, k)
    task2(; benchmark, indexfile,
          outdir=joinpath(outdir, "$name-allknn-$preprocessing"),
          optim=MinRecall(minrecall),
          optimsearch=MinRecall(minrecallsearch),
          preprocessing,
          maxoutdim,
          logbase)
end
