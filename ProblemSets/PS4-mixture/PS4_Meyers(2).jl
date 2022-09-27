using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions

function ProblemSet3()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    # Estimate a multinomial logit (with alternative specific covariates Z) on panel data
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

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

    function mlogit(coef_vector, X, Z_indexed, y)

        # Number of dependent variables in X
        K = size(X,2)

        # Number of choice variables (occupations)
        J = length(unique(y))

        # Number of observations
        N = length(y)

        # Seperate beta matrix and gamma coefficient
        gamma = coef_vector[end]
        beta_vector = coef_vector[1:end-1]
        beta = [reshape(beta_vector, K, J-1) zeros(K)]

        # Create bigY (the Ransom Way)
        bigY = zeros(N,J)
        for j = 1:J
            bigY[:,j] = y.==j
        end

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
        den = (num1 .+ num2 .+  num3.+ num4 .+ num5 .+ num6 .+ num7 .+ num8)

        # Create probability ratios for each choice (occupation)
        prob1 = num1 ./ den
        prob2 = num2 ./ den
        prob3 = num3 ./ den
        prob4 = num4 ./ den
        prob5 = num5 ./ den
        prob6 = num6 ./ den
        prob7 = num7 ./ den
        prob8 = num8 ./ den

        # Create probability matrix
        prob =hcat(prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8)

        loglike = sum(bigY .* log.(prob))
        
        # Old loglikelihood calculation left for reference
        # loglike = sum((D[:,1] .* log.(prob1)) .+ (D[:,2] .* log.(prob2)) .+ (D[:,3] .* log.(prob3))  .+ (D[:,4] .* log.(prob4))  .+ (D[:,5] .* log.(prob5))  .+ (D[:,6] .* log.(prob6)) .+ (D[:,7] .* log.(prob7)) .+ (D[:,8] .* log.(prob8)))
            
        return loglike
    end

    # Run optimizer
    startvals = ones(22,1)
    td = TwiceDifferentiable(coef_vector -> -mlogit(coef_vector, X, Z_indexed, y), startvals; autodiff = :forward)
    beta_hat_loglike = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true, show_every=5))
    beta_hat_mle = beta_hat_loglike.minimizer

    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, beta_hat_mle)
    beta_hat_mle_se = sqrt.(diag(inv(H)))
    println([beta_hat_mle beta_hat_mle_se]) 


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    # Interpret the estimated coefficient γ(hat)
    # Gamma is more difficult to interpret because there is a time element (panel data)
    # From Dr. Ransom's PS3 solutions: gamma represents the change in utility from a 1-unit change in log wages relative to the normalized option "Other"
    # Since we are working with panel data in PS4, I believe gamma is now relative to "Other" wage average over the time period
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 (part a)
    # Practice quadrature with normal distribution N(0,1)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # lgwt.jl is a function that Dr. Ransom gave the class
    # The fucntion returns the ω (omega) and ξ (xi) for a given choice of K

    include("lgwt.jl")

    # Practice using a normal Distributions

    d = Normal(0,1)

    # Get quadrature nodes and weights for 7 grid points
    # For normal distributions, +/- 4σ is sufficient

    nodes, weights = lgwt(7, -4, 4)

    # Compute the integral over the density
    # Should be equal to 1

    integral_density_3a = sum(weights .*pdf.(d,nodes))
    println(integral_density_3a)

    # Compute the expectation
    # Should be equal to 0 (mean of normal distribution)

    expectation_3a = sum(weights .* nodes .* pdf.(d,nodes))
    println(expectation_3a)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 (part b)
    # More Practice w/ quadrature
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Practice using a different normal Distributions

    d = Normal(0,2)

    nodes, weights = lgwt(7, -10, 10)

    integral_density_3b = sum(weights .*pdf.(d,nodes)) # confirm integral is close to 1
    println(integral_density_3b)

    expectation_3b = sum(weights .* (nodes .^ 2) .* pdf.(d,nodes))
    println(expectation_3b)

    # Same as above, but with 10 quadrature points
    # This get us an answer much closer to 4 (4.03898)

    d = Normal(0,2)

    nodes, weights = lgwt(10, -10, 10)

    integral_density_3b2 = sum(weights .*pdf.(d,nodes)) # confirm integral is close to 1
    println(integral_density_3b2)
    
    expectation_3b2 = sum(weights .* (nodes .^ 2) .* pdf.(d,nodes))
    println(integral_density_3b2)

    # Variance is 4 ( =  sigma ^ 2)
    # Quadrature usring 7 points was pretty inaccurate 3.266
    # Quadrature using 10 points was much closer 4.039

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 (part c)
    # Practice with Monte Carlo integration
    # D is the number of random draws
    # Each Xi is drawn from a U[a,b]
    # Notes from Dr. Ransom's office hours
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    sigma = 2
    dist = Normal(0,sigma)
    d = 1_000_000
    a = -5 * sigma
    b = -a
    w = (b - a) / d

    # CDF for integrand x^2 f(x)
    # Returns a number close to 4 (4.00401)
    dummy = 0
    for i = 1:d
        draw = rand(Uniform(a,b))
        integrand = (draw ^ 2) * pdf(dist, draw)
        dummy += integrand
    end
    approx1 = (b - a) / d * dummy
    println(approx1)

    # CDF for integral x f(x)  
    # Returns a number close to 0 (-0.002755)
    dummy = 0
    for i = 1:d
        draw = rand(Uniform(a,b))
        integrand = draw * pdf(dist, draw)
        dummy += integrand
    end
    approx2 = (b - a) / d * dummy
    println(approx2)

    # CDF for integral f(x)  
    # Returns a number close to 1 (1.00141)
    dummy = 0
    for i = 1:d
        draw = rand(Uniform(a,b))
        integrand = pdf(dist, draw)
        dummy += integrand
    end
    approx3 = (b - a) / d * dummy
    println(approx3)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 (part d)
    # Additional notes: similarities between Quadrature and Monte Carlo
    # Quadrature: ω (omega) = quadrature weight; ξ (xi) = quadrature node
    # Comparing Quadrature with Monte Carlo: quadrature weight is analogous to (b-a)/d at each node, and the quadrature node is a U[a,b] random variable
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    # Modify the code from Question 1 to optimize the likelihood function in equation (2)
    # Notes from Dr. Ransom's office hours
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    dist = Normal(0,2)

    # xi (nodes)
    # omega (weights)
    xi, omega = lgwt(7, -10, 10)

    beta_vector = ones(21,1)

    function mixerlogit(X, Z_indexed, y, xi, beta_vector)

        # Number of dependent variables in X
        K = size(X,2)

        # Number of choice variables (occupations)
        J = length(unique(y))

        # Number of observations
        N = length(y)

        beta = [reshape(beta_vector, K, J-1) zeros(K)]

        # Create bigY (the Ransom Way)
        bigY = zeros(N,J)
        for j = 1:J
            bigY[:,j] = y.==j
        end

        # Create numerator for each occupation
        num1 = exp.(X * beta[:,1] .+ xi * Z_indexed[:,1])
        num2 = exp.(X * beta[:,2] .+ xi * Z_indexed[:,2])
        num3 = exp.(X * beta[:,3] .+ xi * Z_indexed[:,3])
        num4 = exp.(X * beta[:,4] .+ xi * Z_indexed[:,4])
        num5 = exp.(X * beta[:,5] .+ xi * Z_indexed[:,5])
        num6 = exp.(X * beta[:,6] .+ xi * Z_indexed[:,6])
        num7 = exp.(X * beta[:,7] .+ xi * Z_indexed[:,7])
        num8 = exp.(X * beta[:,8] .+ xi * Z_indexed[:,8])

        # Create denominator
        den = (num1 .+ num2 .+  num3.+ num4 .+ num5 .+ num6 .+ num7 .+ num8)

        # Create probability ratios for each choice (occupation)
        prob1 = num1 ./ den
        prob2 = num2 ./ den
        prob3 = num3 ./ den
        prob4 = num4 ./ den
        prob5 = num5 ./ den
        prob6 = num6 ./ den
        prob7 = num7 ./ den
        prob8 = num8 ./ den

        # Create probability matrix
        prob =hcat(prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8)

        return prob
    end

    for g = 1:7
        prob = mixerlogit(X, Z_indexed, y, xi[g], beta_vector)
        prob_prod = prod(prob .^ bigY, dims = 2)
        integral .+= omega[g] .* prob_prod .* pdf(dist,xi[g])
    end

    loglike = sum(log.(integral))
    println(loglike)

    # As instructed, I do not solve using the optimizer, but if we did solve we would optimize over beta_vector (21 x 1)
    # The mixerlogit function returns "prob" which is a 28_365 x 8 matrix
    # "prob_prod" returns a 28_365 x 1 vector
    # "integral" returns a 28_365 x 1 vector
    # "loglike" is a scalar 

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    # Modify the code from Question 1 to optimize the likelihood function in equation (1)
    # Use Monte Carlo approximation
    # Notes from Dr. Ransom's office hours
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # dummy gamma and beta_vector to make sure the code works
    gamma = 1
    beta_vector = ones(21,1)

    function mixerlogit_mc(X, Z_indexed, y, gamma, beta_vector)

        # Number of dependent variables in X
        K = size(X,2)

        # Number of choice variables (occupations)
        J = length(unique(y))

        # Number of observations
        N = length(y)

        beta = [reshape(beta_vector, K, J-1) zeros(K)]

        # Create bigY (the Ransom Way)
        bigY = zeros(N,J)
        for j = 1:J
            bigY[:,j] = y.==j
        end

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
        den = (num1 .+ num2 .+  num3.+ num4 .+ num5 .+ num6 .+ num7 .+ num8)

        # Create probability ratios for each choice (occupation)
        prob1 = num1 ./ den
        prob2 = num2 ./ den
        prob3 = num3 ./ den
        prob4 = num4 ./ den
        prob5 = num5 ./ den
        prob6 = num6 ./ den
        prob7 = num7 ./ den
        prob8 = num8 ./ den

        # Create probability matrix
        prob =hcat(prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8)

        return prob
    end

    prob = mixerlogit_mc(X, Z_indexed, y, gamma, beta_vector)
    prob_prod = prod(prob .^ bigY, dims = 2)

    #integral .+= omega[g] .* prob_prod .* pdf(dist,xi[g])
    sigma = 2
    dist = Normal(0,sigma)
    d = 1_000_000
    a = -5 * sigma
    b = -a
    w = (b - a) / d

    integral = zeros(28_365,1)
    for i = 1:d
        draw = rand(Uniform(a,b))
        integrand = prob_prod .* pdf(dist, draw)
        integral_temp .+= integrand
    end

    integral = (b - a) / d * integral_temp

    loglike = sum(log.(integral))
    println(loglike)
end