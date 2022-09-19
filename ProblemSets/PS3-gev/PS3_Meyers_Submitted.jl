using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions

function ProblemSet3()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    # Estimate a multinomial logit (with alternative specific covariates Z)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    freqtable(df, :occupation)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # Difference the Z's in the likelihood function
    # βJ is indexed to 0, so lets Difference Z's for ZJ = 0
    Z_indexed = zeros(size(Z,1),size(Z,2))
    for i in 1:size(Z_indexed,1)
        Z_indexed[i,1] = Z[i,1] - Z[i,8]
        Z_indexed[i,2] = Z[i,2] - Z[i,8]
        Z_indexed[i,3] = Z[i,3] - Z[i,8]
        Z_indexed[i,4] = Z[i,4] - Z[i,8]
        Z_indexed[i,5] = Z[i,5] - Z[i,8]
        Z_indexed[i,6] = Z[i,6] - Z[i,8]
        Z_indexed[i,7] = Z[i,7] - Z[i,8]
        Z_indexed[i,8] = Z[i,8] - Z[i,8]
    end

    function mlogit(coef_vector, X, y)

        # Number of dependent variables in X
        K = size(X,2)

        # Number of choice variables (occupations)
        J = length(unique(y))

        # Number of observations
        N = length(y)

        # Seperate beta matrix and gamma coefficient
        # Reshape beta vector into beta matrix
        gamma = coef_vector[end]
        beta_vector = coef_vector[1: K * (J-1)]
        beta = [reshape(beta_vector, K, J-1) zeros(K)]

        # Create numerator for each occupation
        num1 = exp.(X * beta[:,1] .+ gamma * Z_indexed[:,1])
        num2 = exp.(X * beta[:,2] .+ gamma * Z_indexed[:,2])
        num3 = exp.(X * beta[:,3] .+ gamma * Z_indexed[:,3])
        num4 = exp.(X * beta[:,4] .+ gamma * Z_indexed[:,4])
        num5 = exp.(X * beta[:,5] .+ gamma * Z_indexed[:,5])
        num6 = exp.(X * beta[:,6] .+ gamma * Z_indexed[:,6])
        num7 = exp.(X * beta[:,7] .+ gamma * Z_indexed[:,7])
        num8 = exp.(X * beta[:,8] .+ gamma * Z_indexed[:,8])

        # Create denominator
        den = sum(num1 .+ num2 .+  num3.+ num4 .+ num5 .+ num6 .+ num7 .+ num8)

        # Create probability ratios for each choice (occupation)
        prob1 = num1 ./ den
        prob2 = num2 ./ den
        prob3 = num3 ./ den
        prob4 = num4 ./ den
        prob5 = num5 ./ den
        prob6 = num6 ./ den
        prob7 = num7 ./ den
        prob8 = num8 ./ den

        log.(prob8)
        # Create choice dummy variable matrix used in log likelihood
        D = zeros(N,J)
            for j=1:J
                D[:,j] = y.==j
            end
        D

        loglike = sum((D[:,1] .* log.(prob1)) .+ (D[:,2] .* log.(prob2)) .+ (D[:,3] .* log.(prob3))  .+ (D[:,4] .* log.(prob4))  .+ (D[:,5] .* log.(prob5))  .+ (D[:,6] .* log.(prob6)) .+ (D[:,7] .* log.(prob7)) .+ (D[:,8] .* log.(prob8)))
            
        return loglike
    end

    beta_hat_loglike = optimize(coef_vector -> -mlogit(coef_vector, X, y), zeros(22,1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(beta_hat_loglike.minimizer)


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    # Interpret the estimated coefficient γ(hat)
    # γ(hat) is the impact of differences in wage on the probability of selecting and occupation
    # Positive γ(hat) indicates that a person's likelihood of selecting an occupation increases if the wage he or she could earn in the occupation is higher 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    # Interpret the estimated coefficient γ(hat)
    # γ(hat) is the impact of differences in wage on the probability of selecting an occupation
    # Positive γ(hat) indicates that a person's likelihood of selecting an occupation increases if the wage he or she could earn in the occupation is higher 
    # The coefficients are relative to occupation 8 (other)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # X = Same as in #1 (2237 x 3)
    # Z = Same matrix as #1 (2237 x 8)
    # y = Same vector as #1 (2237 x1)
    # Beta_WC > 3x1
    # Beta_BC > 3x1
    # Lambda_WC > scalar
    # Lambda_BC > scalar
    # Gamma2 > scalar


    function nlogit(coef_vector2, X, y)

        # Number of dependent variables in X
        K = size(X,2)

        # Number of choice variables (occupations)
        # WC includes j's 1 - 3
        # BC includes j's 4 - 7
        # Other includes j 8
        J = length(unique(y))

        # Number of observations
        N = length(y)

        # Reshape coef_vector2
        B_WC = coef_vector2[1:3]
        B_BC = coef_vector2[4:6]
        L_WC = coef_vector2[7]
        L_BC = coef_vector2[8]
        Gamma2 = coef_vector2[9]
        
        # Create dummy variables for each occupation to make match simplier
        dm1 = exp.((X * B_WC .+ Gamma2 * Z_indexed[:,1]) ./ L_WC)
        dm2 = exp.((X * B_WC .+ Gamma2 * Z_indexed[:,2]) ./ L_WC)
        dm3 = exp.((X * B_WC .+ Gamma2 * Z_indexed[:,3]) ./ L_WC)
        dm4 = exp.((X * B_BC .+ Gamma2 * Z_indexed[:,4]) ./ L_BC)
        dm5 = exp.((X * B_BC .+ Gamma2 * Z_indexed[:,5]) ./ L_BC)
        dm6 = exp.((X * B_BC .+ Gamma2 * Z_indexed[:,6]) ./ L_BC)
        dm7 = exp.((X * B_BC .+ Gamma2 * Z_indexed[:,7]) ./ L_BC)
        
        # Create probability ratios for each choice (occupation)
        prob1 = (dm1 .* ((dm1 .+ dm2 .+ dm3) .^ (L_WC - 1))) ./ ((1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC)))
        prob2 = (dm2 .* ((dm1 .+ dm2 .+ dm3) .^ (L_WC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob3 = (dm3 .* ((dm1 .+ dm2 .+ dm3) .^ (L_WC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob4 = (dm4 .* ((dm4 .+ dm5 .+ dm6 .+ dm7) .^ (L_BC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob5 = (dm5 .* ((dm4 .+ dm5 .+ dm6 .+ dm7) .^ (L_BC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob6 = (dm6 .* ((dm4 .+ dm5 .+ dm6 .+ dm7) .^ (L_BC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob7 = (dm7 .* ((dm4 .+ dm5 .+ dm6 .+ dm7) .^ (L_BC - 1))) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))
        prob8 = ( 1 ) ./ (1 .+ (dm1 .+ dm2 .+ dm3) .^ (L_WC) .+ (dm4 .+ dm5 .+ dm6 .+dm7) .^ (L_BC))

        # Create choice dummy variable matrix used in log likelihood
        D2 = zeros(N,J)
            for j=1:J
                D2[:,j] = y.==j
            end
        D2

        loglike = sum((D2[:,1] .* log.(prob1)) .+ (D2[:,2] .* log.(prob2)) .+ (D2[:,3] .* log.(prob3))  .+ (D2[:,4] .* log.(prob4))  .+ (D2[:,5] .* log.(prob5))  .+ (D2[:,6] .* log.(prob6)) .+ (D2[:,7] .* log.(prob7)) .+ (D2[:,8] .* log.(prob8)))
            
        return loglike
    end

    beta_hat_loglike2 = optimize(coef_vector2 -> -nlogit(coef_vector2, X, y), ones(9,1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(beta_hat_loglike2.minimizer)
end

ProblemSet3()