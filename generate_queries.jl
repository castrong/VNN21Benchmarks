using NeuralVerification
using NeuralPriorityOptimizer
using NeuralPriorityOptimizer: cell_to_all_subcells
using LinearAlgebra 
using LazySets 

include(string(@__DIR__, "/utils.jl"))

function generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)
    # Read in the network
    network = read_nnet(network_file)

    # Define the coefficients for a linear objective
    coefficients = [-0.74; -0.44]
    full_input_region = Hyperrectangle(low=lbs, high=ubs)
    cells = cell_to_all_subcells(full_input_region, cells_per_dim)

    # Parameters for the solvers 
    params = PriorityOptimizerParameters(stop_gap=1e-4, stop_frequency=10, max_steps=20000000) # for the priority solver 

    # Now, time the solvers for each cell and record the optimal values 
    times_max = zeros(cells_per_dim...)
    lower_bounds_max = zeros(cells_per_dim...)
    upper_bounds_max = zeros(cells_per_dim...)
    
    times_min = zeros(cells_per_dim...)
    lower_bounds_min = zeros(cells_per_dim...)
    upper_bounds_min = zeros(cells_per_dim...)


    i_linear = 1 # number the queries starting at 1 
    for i in CartesianIndices(cells)
        cell = cells[i]
        # Run the priority optimizer then the mip splitting optimizer
        time_max = @elapsed x_star, lower_bound_max, upper_bound_max, steps_max = optimize_linear(network, cell, coefficients, params; maximize=true)
        times_max[i], lower_bounds_max[i], upper_bounds_max[i] = time_max, lower_bound_max, upper_bound_max
        
        time_min = @elapsed x_star, lower_bound_min, upper_bound_min, steps_min = optimize_linear(network, cell, coefficients, params; maximize=false)
        times_min[i], lower_bounds_min[i], upper_bounds_min[i] = time_min, lower_bound_min, upper_bound_min

	sample_time = @elapsed sample_lower, sample_upper = sample_based_bounds(network, cell, coefficients, 5000) 

        println("Cell ", i)
        println("min and max solve steps: ", [steps_min, steps_max])
        println("lower, upper max: ", [lower_bound_max, upper_bound_max])
        println("lower, upper min: ", [lower_bound_min, upper_bound_min])
	println("so our control interval is: ", [lower_bound_min, upper_bound_max])
	println("compared to from sampling: ", [sample_lower, sample_upper])
        println("Time min: ", time_min, "  time max: ", time_max, " time sample: ", sample_time)

        epsilon = 0.2 # amount off the true maximum or minimum to point the assertion
        # Write out the query, it should cycle between:
        # (i) property has -0.74y0 - 0.44y1 <= minimum - epsilon which will be UNSAT (HOLDS that it's > minimum-epsilon)
        # (ii) property has -0.74y0 - 0.44y1 <= minimum + epsilon which will be SAT (VIOLATED that it's > minimum + epsilon)
        # (iii) property has -0.74y0 - 0.44y1 >= maximum + epsilon which will be UNSAT (HOLDS that it's < maximum + epsilon)
        # (iv) property has -0.74y0 - 0.44y1 >= maximum - epsilon which will be SAT (VIOLATED that it's < maximum - epsilon)
        output_file = string(query_base_name, "_", string(i_linear), ".vnnlib")
	# it shouldn't matter upper vs. lower bound here, since they should all be within 1e-4 of each other
	@assert abs(upper_bound_max - lower_bound_max) <= 1e-4 "gap should be <= 1e-4"
	@assert abs(upper_bound_min - lower_bound_min) <= 1e-4 "gap should be <= 1e-4"
        if i_linear % 4 == 0
            write_query(network, cell, "<=", lower_bound_min - epsilon, coefficients, output_file)
        elseif i_linear % 4 == 1
            write_query(network, cell, "<=", lower_bound_min + epsilon, coefficients, output_file)
        elseif i_linear % 4 == 2
            write_query(network, cell, ">=", upper_bound_max + epsilon, coefficients, output_file)
        elseif i_linear % 4 == 3
            write_query(network, cell, ">=", upper_bound_max - epsilon, coefficients, output_file)
        end
        i_linear = i_linear + 1
    end

    return times_max, lower_bounds_max, upper_bounds_max, times_min, lower_bounds_min, upper_bounds_min
end

function write_query(network, cell, symbol, value, coefficients, output_file)
    open(output_file, "w") do io
        print_var_definition(io, network)
        print_cell(io, cell)
        print_output_constraint(io, coefficients, symbol, value)
    end
end

function print_var_definition(io, network)
    # Define the input variables 
    for i = 1:size(network.layers[1].weights, 2)
        println(io, string("(declare-const X_", i-1, " Real)"))
    end

    # Define the output variables
    for i = 1:size(network.layers[end].weights, 1)
        println(io, string("(declare-const Y_", i-1, " Real)"))
    end
end

function print_cell(io, cell)
    lbs, ubs = low(cell), high(cell)
    for (i, (lb, ub)) in enumerate(zip(lbs, ubs))
        println(io, string("(assert (>= X_", i-1, " ", lb, "))"))
        println(io, string("(assert (<= X_", i-1, " ", ub, "))"))
    end
end

# For now for simplicity we'll assume two-dimensional coefficients 
function print_output_constraint(io, coefficients, symbol, value)
    println(io, string("(assert (", symbol, " (+ (* ", coefficients[1], " Y_0) (* ", coefficients[2], " Y_1)) ", value, "))"))
end

# Small is 0.03, medium is 0.06, large is 0.12
# we will do a Small, Medium, and Large set of queries 
# at the negative edge of the state space and at the center of the state space 
# I expect the center of the state space to be slightly more challenging 

# Use the full best network for all queries  
network_file = string(@__DIR__, "/networks/GANControl/full_mlp_best_conv.nnet")


# Small queries at the negative edge of the state space 
# formed by splitting the state space into 5 cells along each axis
# leading to 25 queries with width [full latent, full latent, 0.03, 0.03]
query_base_name = "queries/negative_edge_small"
lbs = [-0.8, -0.8, -1.0, -1.0]
ubs = [0.8, 0.8, -0.85, -0.85]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Small Edge Region -----------\n")
#generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)

# Medium queries at the negative edge of the state space 
# formed by splitting the state space into 5 cells along each axis
# leading to 25 queries with width [full latent, full latent, 0.06, 0.06]
query_base_name = "queries/negative_edge_medium"
lbs = [-0.8, -0.8, -1.0, -1.0]
ubs = [0.8, 0.8, -0.7, -0.7]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Medium Edge Region -----------\n")
#generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)

# Medium queries at the negative edge of the state space 
# formed by splitting the state space into 5 cells along each axis
# leading to 25 queries with width [full latent, full latent, 0.12, 0.12]
query_base_name = "queries/negative_edge_large"
lbs = [-0.8, -0.8, -1.0, -1.0]
ubs = [0.8, 0.8, -0.4, -0.4]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Large Edge Region -----------\n")
#generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)


# Small queries at the center of the state space 
# formed by splitting the state space into 3 cells along each axis
# leading to 100 queries with width [full latent, full latent, 0.03, 0.03]
query_base_name = "queries/center_small"
lbs = [-0.8, -0.8, -0.075, -0.075]
ubs = [0.8, 0.8, 0.075, 0.075]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Small Center Region -----------\n")
#generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)

# Medium queries at the center of the state space 
# formed by splitting the state space into 3 cells along each axis
# leading to 100 queries with width [full latent, full latent, 0.03, 0.03]
query_base_name = "queries/center_medium"
lbs = [-0.8, -0.8, -0.15, -0.15]
ubs = [0.8, 0.8, 0.15, 0.15]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Medium Center Region -----------\n")
#generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)


# Test small upper right corner of state space 
query_base_name = "queries/positive_edge_small"
lbs = [-0.8, -0.8, 0.85, 0.85]
ubs = [0.8, 0.8, 1.0, 1.0]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Small Positive Edge Region -----------\n")
generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)


# Large queries at the center of the state space 
# formed by splitting the state space into 3 cells along each axis
# leading to 100 queries with width [full latent, full latent, 0.03, 0.03]
query_base_name = "queries/center_large"
lbs = [-0.8, -0.8, -0.3, -0.3]
ubs = [0.8, 0.8, 0.3, 0.3]
cells_per_dim = [1, 1, 5, 5]
println("\n\n------------- Starting Large Center Region -----------\n")
generate_queries_for_region(lbs, ubs, cells_per_dim, network_file, query_base_name)


