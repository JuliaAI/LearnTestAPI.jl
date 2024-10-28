using Test

test_files = [
    "patterns/static_algorithms.jl",
    "patterns/regression.jl",
    "patterns/ensembling.jl",
#    "patterns/gradient_descent.jl",
    "patterns/incremental_algorithms.jl",
    "patterns/dimension_reduction.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
