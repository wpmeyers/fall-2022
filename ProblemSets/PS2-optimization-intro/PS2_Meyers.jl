using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables

function ProblemSet2()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    # Julia's Optim package has a function "minimizer"
    # To find maximum of f(x), need to minimize -f(x)
    #::::::::::::::::::::::::::::::::::::::::::::::::::
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, LBFGS())
    println(result)
    println(result.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #::::::::::::::::::::::::::::::::::::::::::::::::::: 
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    
    bols = inv(X'*X)*X'*y
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println(bols_lm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function logit(alpha, X, y)
        prob1 = exp.(X * alpha) ./ (1 .+ exp.(X * alpha))
        loglike = sum(y .* log.(prob1) .+ (1 .- y) .* log.(1 .- prob1))
        return loglike
    end

    alpha_hat_loglike = optimize(alpha -> -logit(alpha, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(alpha_hat_loglike.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # see Lecture 3 slides for example
    bloglike_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println(bloglike_glm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8,:occupation] .= 7
    df[df.occupation.==9,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X_2 = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1] # 2237x4 Matrix
    y_2 = df.occupation # 2237 vector

    # Normalize with respect to occupation 7
    # We want beta to be [(J-1) x K], but optimization function will only accept a vector 
    # Initialize beta as [(J-1)K x 1] vector and reshape in function

    function mlogit(beta, X, y)
        beta1 = reshape(beta[1:4,1], (4,1))    
        beta2 = reshape(beta[5:8,1], (4,1))
        beta3 = reshape(beta[9:12,1], (4,1))
        beta4 = reshape(beta[13:16,1], (4,1))
        beta5 = reshape(beta[17:20,1], (4,1))
        beta6 = reshape(beta[21:24,1], (4,1))
        
        denominator = sum(1 .+ exp.(X * beta1) .+ exp.(X * beta1).+ exp.(X * beta1).+ exp.(X * beta1).+ exp.(X * beta1).+ exp.(X * beta1))
        
        prob1 = exp.(X * beta1) ./ denominator
        prob2 = exp.(X * beta2) ./ denominator
        prob3 = exp.(X * beta3) ./ denominator
        prob4 = exp.(X * beta4) ./ denominator
        prob5 = exp.(X * beta5) ./ denominator
        prob6 = exp.(X * beta6) ./ denominator

        D = zeros(size(X_2,1),6)
        for i in 1:size(D,1)
            if y[i,1] == 1
                D[i,1] = 1
            elseif y[i,1] == 2
                D[i,2] = 1
            elseif y[i,1] == 3
                D[i,3] = 1
            elseif y[i,1] == 4
                D[i,4] = 1
            elseif y[i,1] == 5
                D[i,5] = 1
            else
                D[i,6] = 1
            end
        end

        loglike = sum((D[:,1] .* log.(prob1)) .+ (D[:,2] .* log.(prob2)) .+ (D[:,3] .* log.(prob3))  .+ (D[:,4] .* log.(prob4))  .+ (D[:,5] .* log.(prob5))  .+ (D[:,6] .* log.(prob6)) )
    
        return loglike
    end

    beta_hat_loglike = optimize(beta -> -mlogit(beta, X_2, y_2), zeros(24,1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(beta_hat_loglike.minimizer)

end

ProblemSet2()


