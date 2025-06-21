using Glob, HDF5, SimilaritySearch, JSON, CSV, DataFrames, Dates

parse_time(d) = DateTime(replace(d, " CEST"  => ""), dateformat"e dd u yyyy HH:MM:SS p")

function report_task1(D, team, path="results-task1/*/*.h5", k=30)
    begins = ""
    ends = ""

    for line in eachline("log-task1.txt")
        m = match(r"^==== RUN BEGINS (.+)", line)
        if m !== nothing
            begins = m.captures[1]
            continue
        end
        m = match(r"^==== RUN ENDS (.+)", line)
        if m !== nothing
            ends = m.captures[1]
            continue
        end
    end

    begins, ends = parse_time(begins), parse_time(ends)
    total_time = (ends - begins).value / 1000

    gold = h5open("/home/sisap23evaluation/data2025/benchmark-eval-pubmed23.h5") do f
        f["otest/knns"][1:k, :]
    end

    for file in glob(path)
        h5open(file) do f
            A = attributes(f)
            let knns=f["knns"][], algo=A["algo"][], task=A["task"][], buildtime=A["buildtime"][], querytime=A["querytime"][], params=A["params"][]
                recall = macrorecall(gold, knns, k)
                push!(D, (; team, algo, task, k, recall, buildtime, querytime, params, begins, ends, total_time))
            end
        end
    end
end

function remove_self_link!(knns, k)
    size(knns, 1) == k && return knns
    for (i, c) in enumerate(eachcol(knns))
        for j in 1:k
            if knns[j, i] == i
                knns[j, i] = knns[k+1, i]
                break
            end
        end 
    end
    @view knns[1:k, :]
end

function report_task2(D, team, filelist=glob("results-task2/*/*.h5"), k=15)
    begins = ""
    ends = ""

    for line in eachline("log-task2.txt")
        m = match(r"^==== RUN BEGINS (.+)", line)
        if m !== nothing
            begins = m.captures[1]
            continue
        end
        m = match(r"^==== RUN ENDS (.+)", line)
        if m !== nothing
            ends = m.captures[1]
            continue
        end
    end

    begins, ends = parse_time(begins), parse_time(ends)
    total_time = (ends - begins).value / 1000
    gold = h5open("/home/sisap23evaluation/data2025/benchmark-eval-gooaq.h5") do f
        f["allknn/knns"][1:k+1, :]
    end
    gold = remove_self_link!(gold, k)
    
    for file in filelist
        h5open(file) do f
            A = attributes(f)
            let knns=f["knns"][], algo=A["algo"][], task=A["task"][], buildtime=A["buildtime"][], querytime=A["querytime"][], params=A["params"][]
                knns = remove_self_link!(knns, k)
                recall = macrorecall(gold, knns)
                push!(D, (; team, algo, task, k, recall, buildtime, querytime, params, begins, ends, total_time))
            end
        end
    end

    CSV.write("results-task2.csv", D)
end

D = DataFrame(; team=[], algo=[], task=[], k=[], recall=[], buildtime=[], querytime=[], params=[], begins=[], ends=[], total_time=[])
report_task1(D, "ABS-BLINE")
sort!(D, :recall)
report_task2(D, "ABS-BLINE")
@info D
CSV.write("results-task12.csv", D)
