using AxisKeys
using Distributions
using Documenter
using KeyedDistributions
using Metrics

makedocs(;
    modules=[Metrics],
    format=Documenter.HTML(
        assets=[
            "assets/invenia.css",
        ],
        prettyurls=false,
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md"
    ],
    repo="https://gitlab.invenia.ca/invenia/Metrics.jl/blob/{commit}{path}#L{line}",
    sitename="Metrics.jl",
    authors="Invenia Technical Computing Corporation",
    strict=true,
    checkdocs=:none,
)
