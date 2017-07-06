
export lbfgsbS

using NLPModels
using Stopping

function lbfgsbS(nlp :: AbstractNLPModel,
                 x::Array;
                 lb = [],
                 ub = [],
                 btype = [],
                 m::Int64 = 5,
                 stp :: TStopping = TStopping(),
                 #maxiter::Int64 = 100,
                 factr::Float64 = 1e7,
                 #pgtol::Float64 = 1e-5,
                 iprint::Int64 = -1 # does not print
                 )

    function _ogFunc!(x, g::Array)
        f, g = objgrad!(nlp, x, g)
        return f
    end
    #ogFunc!(x,g) = objgrad!(nlp,x, g)

    start!(nlp,stp,x)
    
    initial_x = x;
    
    m = [convert(Int32, m)]
    factr = [convert(Float64, factr)];
    #pgtol = [convert(Float64, pgtol)];
    pgtol = [convert(Float64,stp.atol + stp.rtol * stp.optimality0)];
    iprint = [convert(Int32, iprint)];
    
    n = [convert(Int32, length(x))]; # number of variables
    f = [convert(Float64, 0.0)]; # The value of the objective.
    g = [convert(Float64, 0.0) for i=1:(n[1])]; # The value of the gradient.

    if length(lb) == 0
        lb = [-Inf for i=1:(n[1])];
    else
        lb = [convert(Float64, i) for i in lb]
    end

    if length(ub) == 0
        ub = [Inf for i=1:(n[1])];
    else
        ub = [convert(Float64, i) for i in ub]
    end

    if length(btype) == 0
        btype = [convert(Int32, 2) for i=1:(n[1])];
    else
        btype = [convert(Int32, i) for i in btype];
    end

    # structures used by the L-BFGS-B routine.
    wa = [convert(Float64, 0.0) for i = 1:(2*m[1] + 5)*n[1] + 12*m[1]*(m[1] + 1)];
    iwa = [convert(Int32, 0) for i = 1:3*n[1]];
    task = [convert(UInt8, 0) for i =1:60];
    csave = [convert(UInt8, 0) for i =1:60];
    lsave = [convert(Bool, 0) for i=1:4];
    isave = [convert(Int32, 0) for i=1:44];
    dsave = [convert(Float64, 0.0) for i=1:29];

    @callLBFGS "START"

    status = "success";

    t = 0;
    c = 0;

    while true

        if task[1] == UInt32('F')
            fobj, g = objgrad!(nlp, x, g)
            f[1] = convert( Float64, fobj );
            c += 1;
            
        elseif task[1] == UInt32('N')
            t += 1;
            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,t,x,f[1],g)
            #if t >= maxiter # exceed maximum number of iteration
            if tired
                @callLBFGS "STOP"
                break;
            end
        elseif task[1] == UInt32('C') # convergence
            break;
        elseif task[1] == UInt32('A')
            status = "abnormal";
            break;
        elseif task[1] == UInt32('E')
            status = "error";
            break;
        end

        @callLBFGS ""
    end

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,t,x,f[1],g)

    return (f[1], x, t, c, status, optimal, unbounded, tired, elapsed_time)
    

end # function lbfgsbS


