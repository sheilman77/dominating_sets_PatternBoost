include("constants.jl")
using JSON
using Polynomials
using Random

# --- CONFIGURATION ---
# N is the number of vertices. 
# WARNING: Since we are using Brute Force O(2^N), N must be small (<= 20).
# N=50 is impossible for brute force. I have set it to 12 for testing.

const alpha = 1/2
const beta  = 1

# The index in the polynomial sequence we want to check for log-concavity.
# Sequence d_0, d_1, d_2... 
# We check a_k^2 - a_{k-1}a_{k+1} >= 0.
const CHECK_INDEX = 2

# Total number of possible edges in an undirected graph of size N
const NUM_POSSIBLE_EDGES = (N * (N - 1)) ÷ 2

# ---------- Helper Functions ----------

"""
    get_closed_neighborhoods(n::Int, bin_vec::Vector{Int}) -> Vector{UInt64}

Converts the binary vector representation of the graph into a list of bitmasks.
Each bitmask represents the closed neighborhood of a vertex (the vertex itself + neighbors).
"""
function get_closed_neighborhoods(n::Int, bin_vec::Vector{Int})
    # Initialize neighborhoods with the vertex itself (closed neighborhood condition)
    # Using UInt64 allows N up to 64.
    neighborhoods = [UInt64(1) << (i-1) for i in 1:n]
    
    edge_idx = 1
    for i in 1:n
        for j in i+1:n
            if bin_vec[edge_idx] == 1
                # Add j to i's neighbor list and i to j's neighbor list
                mask_j = UInt64(1) << (j-1)
                mask_i = UInt64(1) << (i-1)
                
                neighborhoods[i] |= mask_j
                neighborhoods[j] |= mask_i
            end
            edge_idx += 1
        end
    end
    return neighborhoods
end

# ---------- Dominating Polynomial (Brute Force) ----------

"""
    dominating_polynomial(bin_vec::Vector{Int}) -> Vector{Int}

Calculates the dominating polynomial of the graph using brute force search over all 2^N subsets.
Returns the coefficients array [d_0, d_1, ..., d_N].
"""
function dominating_polynomial(bin_vec::Vector{Int})
    neighborhoods = get_closed_neighborhoods(N, bin_vec)
    
    # Coefficients array, indexed by cardinality (size 0 to N)
    coeffs = zeros(Int, N + 1)
    
    # Target mask is all 1s (representing the set of all vertices V)
    full_mask = (UInt64(1) << N) - 1
    
    # Iterate through all 2^N subsets
    # i represents the bitmask of the subset S
    for i in 0:full_mask
        # Check if subset i is a dominating set
        # A subset S is dominating if the union of closed neighborhoods of v \in S equals V
        
        union_mask = UInt64(0)
        temp_i = i
        v_idx = 1
        
        # Iterate bits of i to find union of neighborhoods
        while temp_i > 0
            if (temp_i & 1) == 1
                union_mask |= neighborhoods[v_idx]
            end
            temp_i >>= 1
            v_idx += 1
        end
        
        # If the union covers all vertices, it is a dominating set
        if union_mask == full_mask
            cardinality = count_ones(i)
            coeffs[cardinality + 1] += 1
        end
    end
    
    # Return coefficients. 
    # Trim trailing zeros if necessary, though for d.p. usually d_N=1 (all vertices) is non-zero.
    return coeffs
end

# ---------- Reward / Score Calculation ----------

"""
    reward_calc(obj::Vector{Int}) -> Float64

Calculates the log-concavity score for the graph.
Score = -(a_k^2 - a_{k-1}*a_{k+1})
We want to maximize this (making it less concave/more convex) or simply check the sign.
"""
function reward_calc(obj::Vector{Int})
    coeffs = dominating_polynomial(obj)
    coeffs = [c for c in coeffs if c!=0]
    
    # Ensure array is large enough for CHECK_INDEX
    # CHECK_INDEX is 1-based index for Julia, but math usually is 0-based.
    # Assuming CHECK_INDEX 3 maps to d_2 in mathematical notation (index 3 in julia array).
    
    if length(coeffs) < CHECK_INDEX + 1
        # Pad with zeros if the graph is weird and polynomial is short (unlikely for N=12)
        return -1e9
    end

    a1 = BigInt(coeffs[CHECK_INDEX - 1]) # d_{k-1}
    a2 = BigInt(coeffs[CHECK_INDEX])     # d_k
    a3 = BigInt(coeffs[CHECK_INDEX + 1]) # d_{k+1}

    # Punish the path graph logic (optional, preserved from original intent but hard to detect strictly on generic graphs without iso check)
    # Skipping specific structural punishment for generic graphs to focus on log-concavity.

    # We want to check log-concavity. 
    # Usually log-concave means a_k^2 >= a_{k-1}a_{k+1}.
    # The score returned is -(a_k^2 - a_{k-1}a_{k+1}).
    # If this is positive, it means a_k^2 < a_{k-1}a_{k+1} (not log concave).
    
    return -Float64(a2^2 - a1*a3)
end

# ---------- Local Optimization ----------

"""
    greedy_search_from_startpoint(db, obj::Vector{Int}) -> Vector{Int}

1. Checks if current score is positive. If so, returns immediately.
2. Generates all neighbors (Hamming distance 1).
3. Evaluates all neighbors and the current graph.
4. Returns the single best graph found.
"""
function greedy_search_from_startpoint(db, obj::Vector{Int})
    
    current_score = reward_calc(obj)
    
    # Optimization skip condition
    if current_score > 0
        return obj
    end

    best_obj = copy(obj)
    best_score = current_score
    
    # Try flipping every single edge
    # obj is a binary vector of length NUM_POSSIBLE_EDGES
    for k in 1:length(obj)
        # Flip edge k
        obj[k] = 1 - obj[k]
        
        new_score = reward_calc(obj)
        
        if new_score > best_score
            best_score = new_score
            best_obj = copy(obj)
        end
        
        # Flip back to restore for next iteration
        obj[k] = 1 - obj[k]
    end
    
    return best_obj
end

# ---------- Utilities ----------

function obj_to_json_string(obj::Vector{Int})::String
    return JSON.json(obj)
end

function json_string_to_obj(json_str::String)::Vector{Int}
    return JSON.parse(json_str)
end

"""
    empty_starting_point() -> Vector{Int}

Generates a random binary vector representing a random graph.
"""
function empty_starting_point()::Vector{Int}
    # Random binary vector of length N(N-1)/2
    return rand(0:1, NUM_POSSIBLE_EDGES)
end
