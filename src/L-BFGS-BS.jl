export LbfgsBS


#  High level interface 

function LbfgsBS(nlp :: AbstractNLPModel;
                 x₀=[],
                 lb=[],
                 ub=[],
                 stp :: AbstractStopping = TStoppingB(),
                 verbose :: Bool=false,
                 m :: Int=5,
                 kwargs...)
    n = nlp.meta.nvar
    x₀!=[] || (x₀ = copy(nlp.meta.x0))

    lb!=[] || (lb = nlp.meta.lvar)
    ub!=[] || (ub = nlp.meta.uvar)

    #stp.lb = lb
    #stp.ub = ub


    g = Array{Float64}(undef,n)
    g₀ = Array{Float64}(undef,n)
    grad!(nlp, x₀, g)
    grad!(nlp, x₀, g₀)

    #function _ogFunc!(x, g::Array{Float64})
    #    return objgrad!(nlp,x,g);
    #end

    
    tolI = max(stp.atol , stp.rtol * norm(g₀,Inf)) 
    verbose && println("LbfgsB: atol = ",stp.atol," rtol = ",stp.rtol," tolI = ",tolI, " norm(g₀) = ",norm(g₀))
    
    verblevel = verbose ? 1 : -1

    x, f, g, iterB, callB, status, optimal, unbounded, tired, elapsed_time  =
        lbfgsbS(nlp,
                x₀,
                lb=lb,
                ub=ub,
                m=m,
                stp=stp,
                iprint = verblevel,
                factr = 0.0
                )

    calls = [nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hess, nlp.counters.neval_hprod]  
    if tired status = :UserLimit 
    elseif optimal  status =  :Optimal
    else status =  :SubOptimal
    end

    pg = gradproj(ub, lb, g, x)
    Auϵ = findall((ub - x) .<= 0)
    Alϵ = findall((x - lb) .<= 0)
    Aϵ = (Auϵ ∪ Alϵ)
    Iϵ = setdiff(1:n,  Aϵ)

    println("#Iϵ = ", length(Iϵ), " #Aϵ = ", length(Aϵ)) 

    return (x, f, stp.optimality_residual(pg), iterB, optimal, tired, status, elapsed_time)
    #return (x, f, stp.optimality_residual(stp.state), iterB, optimal, tired, status, elapsed_time)
    
end
proj(ub :: Vector, lb :: Vector, x :: Vector) = max.(min.(x,ub),lb)

gradproj(ub :: Vector, lb :: Vector, g::Vector, x :: Vector) =  x - proj(ub, lb, x-g)
