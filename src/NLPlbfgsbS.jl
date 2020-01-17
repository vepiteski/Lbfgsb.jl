export NLPlbfgsbS

using Stopping
using LinearAlgebra  #  for norm
using NLPModels

#  Low level interface, direct connexion with the FORTRAN code
# the bounds lb and ub, if any, are given in nlp
# no optimization performed for partially bounded models, but
# the btype parameter may be used to specify such per component bound property
# see the L-BFGS-B paper.
#

proj(ub :: Vector, lb :: Vector, x :: Vector) = max.(min.(x,ub),lb)

gradproj(ub :: Vector, lb :: Vector, g::Vector, x :: Vector) =  x - proj(ub, lb, x-g)

function optim_check_bounded(pb    :: AbstractNLPModel,
                             state :: NLPAtX;
                             pnorm :: Float64 = Inf)
    res = norm(gradproj(pb.meta.uvar, pb.meta.lvar, state.gx, state.x), pnorm)

    return res
end


function NLPlbfgsbS(nlp :: AbstractNLPModel,
                    x₀::Array;
                    btype = [],
                    m::Int64 = 5,
                    stp :: AbstractStopping = NLPStopping(nlp,
                                                          (p,s) -> optim_check_bounded(p,s),
                                                          NLPAtX(x₀, [])
                                                          ),
                    factr::Float64 = 1e1,
                    iprint::Int64 = -1 # does not print
                    )


    lb = nlp.meta.lvar
    ub = nlp.meta.uvar
    
    update_and_start!(stp, x= x₀)
    
    x = copy(x₀)
    
    m = [convert(Int32, m)]
    factr = [convert(Float64, factr)];
    pgtol = [convert(Float64,max(stp.meta.atol, stp.meta.rtol * stp.meta.optimality0))];
    iprint = [convert(Int32, iprint)];
    
    n = [convert(Int32, length(x₀))]; # number of variables
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
        btype = [convert(Int32, 2) for i=1:(n[1])]; # by default consider lb <= x <= ub
    else
        btype = [convert(Int32, i) for i in btype]; # 
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

    c = 0;

    while true

        if task[1] == UInt32('F')
            fobj, g = objgrad!(nlp, x, g)
            f[1] = convert( Float64, fobj );
            c += 1;
            
        elseif task[1] == UInt32('N')
            stp_part = update_and_stop!(stp, x=x, fx=f[1], gx=g)
            if stp.meta.tired
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
    sf = stop!(stp)
    

    return stp
    

end # function NLPlbfgsbS


