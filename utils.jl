using NeuralVerification: compute_output
using LinearAlgebra
function sample_based_bounds(network, cell, coefficients, num_samples)
    xs = sample(cell, num_samples)
    min_obj = Inf
    max_obj = -Inf
    for x in xs 
        output = compute_output(network, x)
        obj = dot(output, coefficients)
        min_obj = min(min_output, obj)
        max_obj = max(max_output, obj)
    end
    return [min_obj, max_obj]
end