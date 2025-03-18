using HDF5, SimilaritySearch, DataFrames, CSV, Glob, UnicodePlots, Printf, Statistics, StatsBase, Markdown

function read_gold(filename; group, k=30, qlist=0:0.10:1)
    G = h5open(filename) do f
        display(md"## structure of the h5 file")
        display(f)
        (; knns=f["$group/knns"][1:k, :], dists=f["$group/dists"][1:k, :])
    end

    G
end

function distquantiles(name, dists; qlist=0.0:0.1:1.0, klist=[1, 3, 10, 30])
    qlist = 0.0:0.1:1.0
    D = DataFrame(name=[], min=[], q10=[], q20=[], q30=[], q40=[], q50=[], q60=[], q70=[], q80=[], q90=[], max=[])
    for k in klist
        push!(D, ("$(k)nn", quantile(dists[k, :], qlist)...))
    end
    display(md"""## Quantiles for different nearest neighbors""")
    display(name)
    display(D)
end

function eval_task1(; 
        goldfile = "data/benchmark-dev-ccnews-fp16.h5",
        filelist = glob("results/benchmark-dev-ccnews-fp16-otest/*.h5"),
        k=30,
        minrecall = 0.7,
        group = "otest"
    )

    G = read_gold(goldfile; k, group)
    distquantiles(md"gold standard: $goldfile", G.dists)

    # Some statistics about the gold standard 
    display(md"## Result analysis of your algorithm")
    D = DataFrame(dataset=[], task=[], querytime=[], throughput=[], recall=[], buildtime=[], group=[], algo=[], optimtime=[], size=[], file=[])
    numqueries = size(G.knns, 2)
    numinspect = 3

    for (i, file) in enumerate(filelist)
        h5open(file) do f
            i == 1 && display(md"""
            ### result file structure

            $f
            """)
            A = attrs(f)
            knns, dists = f["knns"][1:k, :], f["dists"][1:k, :]
            recall = macrorecall((@view G.knns[1:k, :]), knns)
            if recall >= minrecall && numinspect > 0
                numinspect -= 1
                distquantiles(md"""
                - recall: $recall
                - minrecall: $minrecall
                - file: $file
                """, dists)
            end

            push!(D, (; dataset=A["dataset"], task=A["task"], algo=A["algo"], recall, group, buildtime=A["buildtime"], optimtime=A["optimtime"], querytime=A["querytime"], throughput=numqueries / A["querytime"], size=A["size"], file))
        end
    end

    P = lineplot(title="query throughput vs recall -- $group",
        xlabel="recall", ylabel="q/s",
        height=25, width=60,
        xlim=(0.3, 1.0), ylim=(0.0, maximum(D.throughput) * 1.05),
        border=:ascii
        )
    for g in groupby(D, :algo)
        lineplot!(P, g.recall, g.throughput, name=g.algo[1])
    end

    display(P)
    display(D)

    CSV.write("result-task1.csv", D)
end

function remove_self_loop!(knns, dists)
    for i in axes(knns, 2)
        knns_ = @view knns[:, i]
        p = findfirst(x -> x == i, knns_)

        if p !== nothing
            dists_ = @view dists[:, i]
            knns_[p:end-1] .= knns_[p+1:end]
            dists_[p:end-1] .= dists_[p+1:end]
        end
    end

end

function eval_task2(; 
    goldfile = "data/allknn-benchmark-dev-ccnews.h5",
    resfile = only(glob("results/benchmark-dev-ccnews-fp16-allknn/*.h5")),
    k=15
)

    G = read_gold(goldfile; k=k+1, group="")
    remove_self_loop!(G.knns, G.dists)
    distquantiles(md"gold standard: $goldfile", G.dists, klist=[1, 5, 10, 15])

    # Some statistics about the gold standard 
    display(md"## Result analysis of your _allknn_ algorithm")
    
    R = h5open(resfile) do f
        display(md"""
        ### result file structure
        file: $resfile

        $f
        """)
        A = attrs(f)
        knns, dists = f["knns"][1:k, :], f["dists"][1:k, :]
        remove_self_loop!(knns, dists)
        recall = macrorecall((@view G.knns[1:k, :]), (@view knns[1:k, :]))
        
        distquantiles(md"", dists, klist=[1, 5, 10, 15])
        buildtime = A["buildtime"]
        optimtime = A["optimtime"]
        querytime = A["querytime"]
        task = A["task"]
        dataset = A["dataset"]
        totaltime = buildtime + optimtime + querytime
        (; dataset, task, algo=A["algo"], recall, totaltime, buildtime, optimtime, querytime, size=A["size"], resfile)
    end
    
    D = DataFrame([R])
    display(D)
    CSV.write("result-task2.csv", D)
end

