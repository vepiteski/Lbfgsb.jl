export LbfgsBS


function LbfgsBS(nlp :: AbstractNLPModel;
                 stp :: TStopping = TStopping(),
                 verbose :: Bool=false,
                 m :: Int=5,
                 kwargs...)
    n = nlp.meta.nvar
    x₀ = copy(nlp.meta.x0)
    
    g = Array(Float64,n)
    g₀ = Array(Float64,n)
    g = grad(nlp,x₀)
    grad!(nlp,x₀,g₀)

    #function _ogFunc!(x, g::Array{Float64})
    #    return objgrad!(nlp,x,g);
    #end

    
    tolI = atol + rtol * norm(g₀,Inf) 
    verbose && println("LbfgsB: atol = ",atol," rtol = ",rtol," tolI = ",tolI, " norm(g₀) = ",norm(g₀))

    verblevel = verbose ? 1 : -1
    
    f, x, iterB, callB, status, optimal, unbounded, tired, elapsed_time  =
        lbfgsbS(nlp,
                x₀,
                m=m,
                stp=stp,
                iprint = verblevel,
                factr = 0.0
                )

    #tired = ! ((iterB < max_iter) & (callB < max_eval))

    grad!(nlp,x,g)
    #gNorm = norm(g,Inf)

    #optimal = gNorm <= tolI

    calls = [nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hess, nlp.counters.neval_hprod]  
    if tired status = :UserLimit 
    elseif optimal  status =  :Optimal
    else status =  :SubOptimal
    end

    return (x, f, stp.optimality_residual(g), iterB, optimal, tired, status, elapsed_time)
    
end
