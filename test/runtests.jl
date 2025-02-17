using Test

test_files = [
    "tools.jl",
    "learners/static_algorithms.jl",
    "learners/regression.jl",
    "learners/classification.jl",
    "learners/ensembling.jl",
#    "learners/gradient_descent.jl",
    "learners/incremental_algorithms.jl",
    "learners/dimension_reduction.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
