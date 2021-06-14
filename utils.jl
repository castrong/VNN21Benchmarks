using NeuralVerification: compute_output
function sample_based_bounds(network, cell, coefficients, num_samples)
    xs = sample(cell, num_samples)
    min_output = Inf
    max_output = -Inf
    for x in xs 
        output = compute_output(network, x)
        min_output = min(min_output, output)
        max_output = max(max_output, output)
    end
    return [min_output, max_output]
end