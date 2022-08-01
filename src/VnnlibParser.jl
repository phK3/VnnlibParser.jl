module VnnlibParser



## vnnlib reader
# Julia translation of Stanley Bak's python parser here: https://github.com/stanleybak/nnenum/blob/master/src/nnenum/vnnlib.py


"""
Return a list of srings (statements) from vnnlib file.

Gets rid of comments and blank lines and combines multi-line statements.
"""
function read_statements(vnnlib_file)
    lines = eachline(vnnlib_file)

    open_parantheses = 0
    statements = []
    current_statement = ""
    for line in lines
        comment_idx = findfirst(";", line)
        if ~isnothing(comment_idx)
            line = line[1:comment_idx[1]-1]
        end

        new_open = count("(", line)
        new_close = count(")", line)

        open_parantheses += new_open - new_close

        @assert open_parantheses >= 0 "mismatched parantheses in vnnlib file"

        # add space
        current_statement = isnothing(current_statement) ? "" : string(current_statement, " ")
        current_statement = string(current_statement, line)

        if open_parantheses == 0
            push!(statements, current_statement)
            current_statement = ""
        end
    end

    ~isnothing(current_statement) && push!(statements, current_statement)

    # remove repeated whitespace characters
    statements = [join(split(s), " ") for s in statements]

    # remove space after "("
    statements = [replace(s, "( " => "(") for s in statements]

    # remove space after ")"
    statements = [replace(s, ") " => ")") for s in statements]

    return statements[length.(statements) .> 0]
end


"""
Updates a return value tuple of (lo, hi, A, b) with lower and upper bounds on the
input values and constraints Ax ≤ b with the constraints provided in op, fst, snd

op - (string) operand (i.e. ≤ or ≥)
fst - (string) first operand (either X_n or Y_n or a constant)
snd - (string) second operand (either X_n or Y_n or a constant)
n_in - number of inputs to the NN
n_out - number of outputs to the NN
"""
function update_rv_tuple(rv_tuple, op, fst, snd, n_in, n_out)
    # rv is 4-tuple (lo, hi, A, b)
    # A needs to be list of rows, to be converted to matrix later!
    if startswith(fst, "X_")
        # input constraints
        idx = parse(Int, fst[3:end]) + 1 # julia is 1-indexed
        @assert ~startswith(snd, "X_") && ~startswith(snd, "Y_") "input constraints must be box! ($op, $fst, $snd)"
        @assert 1 <= idx && idx <= n_in "index of variable exceeds number of inputs: n_in = $(n_in), $fst"

        lb = rv_tuple[1][idx]
        ub = rv_tuple[2][idx]

        if op == "<="
            ub = min(parse(Float64, snd), ub)
        else
            lb = max(parse(Float64, snd), lb)
        end

        @assert lb <= ub "$fst : range is empty: ($lb, $ub)"

        rv_tuple[1][idx] = lb
        rv_tuple[2][idx] = ub
    else
        # output constraint
        if op == ">="
            # swap order to get <=
            fst, snd = snd, fst
        end

        row = zeros(n_out)
        rhs = 0.

        # now op is <=
        if startswith(fst, "Y_") && startswith(snd, "Y_")
            idx1 = parse(Int, fst[3:end]) + 1
            idx2 = parse(Int, snd[3:end]) + 1
            # y1 <= y2 <--> y1 - y2 <= 0
            row[idx1] = 1
            row[idx2] = -1
        elseif startswith(fst, "Y_")
            # snd is then constant!
            idx1 = parse(Int, fst[3:end]) + 1
            row[idx1] = 1
            rhs = parse(Float64, snd)
        else
            @assert startswith(snd, "Y_")
            idx2 = parse(Int, snd[3:end]) + 1
            row[idx2] = -1
            rhs = -1 * parse(Float64, fst)
        end

        A, b = rv_tuple[3], rv_tuple[4]
        push!(A, row)
        push!(b, rhs)
    end
end


function get_conjuncts(line)
    tokens = replace(line, "(" => " ")
    tokens = replace(tokens, ")" => " ")
    tokens = split(tokens)
    tokens = tokens[3:end]  # skip "assert" and "or"

    conjuncts = join(tokens, " ")
    conjuncts = split(conjuncts, "and")[2:end]  # skip the "and"
    return conjuncts
end


function parse_conjunct!(rv, rv_tuple, conjunct, n_in, n_out)
    l, u, A, b = rv_tuple
    # in julia you can't copy tuples as they are immutable (i.e. you can't change rv_tuple[1] = l',
    # you CAN however, say rv_tuple[1][1] = -5, which is why we need a COPY)
    rv_tuple_copy = (copy(l), copy(u), copy(A), copy(b))
    push!(rv, rv_tuple_copy)

    c_tokens = [s for s in split(conjunct) if length(s) > 0]
    count = Integer(length(c_tokens) / 3)

    for i in 1:count
        op, fst, snd = c_tokens[3*(i-1)+1:3*i]
        update_rv_tuple(rv_tuple_copy, op, fst, snd, n_in, n_out)
    end
end


function merge_same_input_spaces!(rv)
    merged_dict = Dict()
    for rv_tuple in rv
        l, u, A, b = rv_tuple
        key = string(l, u)  # just match on string representation for now

        if haskey(merged_dict, key)
            _, _, matrhs = merged_dict[key]
            # A... as A is list of rows
            #push!(A1, A...)
            #push!(b1, b...)
            #push!(A1, A)
            #push!(b1, b)
            push!(matrhs, (A, b))
        else
            # merged_dict[key] = rv_tuple
            merged_dict[key] = (l, u, [(A, b)])
        end
    end

    return values(merged_dict)
end


function convert_to_matrices(rv)
    final_rv = []
    for rv_tuple in rv
        #l, u, A, b = rv_tuple
        l, u, matrhs = rv_tuple
        speclist = []
        for (A, b) in matrhs
            A = Matrix(hcat(A...)')
            b = Float64.(b)
            push!(speclist, (A, b))
        end

        push!(final_rv, (l, u, speclist))
    end
    return final_rv
end


"""
Parses property specification in simplified vnnlib format.

vnnlib_file - path of the vnnlib file
n_in - number of inputs of the NN
n_out - number of outputs of the NN

returns result value - list [(l, u, output_specs)] of output specs for common
input constraints defined by [l, u].
An output_specs is a list [(A, b)] of output specs representing constraints Ax ≤ b.

"""
function read_vnnlib_simple(vnnlib_file, n_in, n_out)
    # should match e.g. "(declare const X_0 Real)"
    regex_declare = r"^\(declare-const (X|Y)_(\S+) Real\)$"

    # should match e.g. "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
    regex_comparison = r"\((<=|>=) (\S+) (\S+)\)"

    # should match e.g. "(and (<= Y_0 Y_1)(<= Y_1 Y_2))"
    regex_dnf_clause = r"\(and (\((<=|>=) (\S+) (\S+)\))+\)"

    # example: "(assert (<= Y_0 Y_1))"
    regex_simple_assert = r"^\(assert \((<=|>=) (\S+) (\S+)\)\)$"

    # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
    regex_dnf = r"^\(assert \(or (\(and (\((<=|>=) (\S+) (\S+)\))+\))+\)\)$"

    rv = []  # list of 4-tuples (lo, hi, A, b)
    push!(rv, (fill(-Inf, n_in), fill(Inf, n_in), [], []))

    lines = read_statements(vnnlib_file)

    for line in lines
        # skip variable declarations
        group = match(regex_declare, line)
        ~isnothing(group) && continue

        # parse simple assert
        group = match(regex_simple_assert, line)
        if ~isnothing(group)
            op, fst, snd = group.captures

            for rv_tuple in rv
                update_rv_tuple(rv_tuple, op, fst, snd, n_in, n_out)
            end

            continue
        end

        # parse dnf expression
        group = match(regex_dnf, line)
        @assert ~isnothing(group) "Failed to parse line: $line"

        conjuncts = get_conjuncts(line)

        old_rv = copy(rv)
        rv = []
        for rv_tuple in old_rv
            for c in conjuncts
                parse_conjunct!(rv, rv_tuple, c, n_in, n_out)
            end
        end
    end

    rv = merge_same_input_spaces!(rv)
    final_rv = convert_to_matrices(rv)
    return final_rv
end


export read_vnnlib_simple

end # module
