export lbfgsbS

using Stopping

#  Low level interface, direct connexion with the FORTRAN code
#
#  If bounds are present, must be explicitely passed in lb and ub (not extracted from nlp)
#
#  Use preferably the higher level interface LbfgsBS

function lbfgsbS(nlp :: AbstractNLPModel,
                 x::Array;
                 lb = [],
                 ub = [],
                 btype = [],
                 m::Int64 = 5,
                 stp :: AbstractStopping = TStopping(),
                 factr::Float64 = 1e1,
                 iprint::Int64 = -1 # does not print
                 )

    #function _ogFunc!(x, g::Array)
    #    f, g = objgrad!(nlp, x, g)
    #    return f
    #end

    start!(stp,x)
    
    initial_x = x;
    
    m = [convert(Int32, m)]
    factr = [convert(Float64, factr)];
    pgtol = [convert(Float64,max(stp.atol, stp.rtol * stp.optimality0))];
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
            pg = gradproj(ub, lb, g, x)
            stp_part = stop(stp,t,x,f[1],pg)
            #optimal, unbounded, tired, elapsed_time = stop(stp,t,x,f[1],pg)
            #if t >= maxiter # exceed maximum number of iteration
            if stp.tired
                @callLBFGS "STOP"
                status = "Max ressources"
                break;
            end
        elseif task[1] == UInt32('C') # convergence
            status = "convergence"
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


    # Check the stopping tolerances
    pg = gradproj(ub, lb, g, x)
    #optimal, unbounded, tired, elapsed_time = stop(stp,t,x,f[1],pg)
    sf = stop(stp,t,x,f[1],pg)

    return (x, f[1], g, t, c, status, stp.optimal, stp.unbounded, stp.tired, stp.elapsed_time)
    

end # function lbfgsbS


