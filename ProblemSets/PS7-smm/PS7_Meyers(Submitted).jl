using Pkg, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, SMM, FreqTables, Distributions
Random.seed!(1234)

function PS7()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: Estimate the linear regression model from Question 2 of Problem Set 2 by GMM. Write down the moment
    # function as in slide [#10] of the Lecture 9 slide deck and use Optim for estimation. Use the N × N Identity matrix 
    # as your weighting matrix. Check your answer using the closed-form matrix formula for the OLS estimator.
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # from PS2 solutions
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    # X is a 2246 x 4 matrix
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    # y is a 2246 x 1 vector
    y = df.married.==1

    # Estimate OLS using GMM
    function OLS_gmm(beta, X, y)
        g = y .- X*beta
        J = g'*I*g
        return J
    end
    beta_optim = optimize(beta -> OLS_gmm(beta, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
    println(beta_optim.minimizer)

    # Check your answer using the closed-form matrix formula for the OLS estimator
    # from PS2 solutions
    bols = inv(X'*X)*X'*y
    println(bols)
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println(bols_lm)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2:  Estimate the multinomial logit model from Question 5 of Problem Set 2 by the following means:
    # (a) Maximum likelihood (i.e. re-run your code [or mine] from Question 5 of Problem Set 2)
    # (b) GMM with the MLE estimates as starting values. Your g object should be a vector of dimension N ×J where N is 
    # the number of rows of the X matrix and J is the dimension of the choice set. 
    # Each element, g should equal d − P, where d and P are “stacked” vectors of dimension N ×J
    # (c) GMM with random starting values 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Part (A)
    # from PS2 solutions
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    df2 = df
    df2= dropmissing(df2, :occupation)
    df2[df2.occupation.==8 ,:occupation] .= 7
    df2[df2.occupation.==9 ,:occupation] .= 7
    df2[df2.occupation.==10,:occupation] .= 7
    df2[df2.occupation.==11,:occupation] .= 7
    df2[df2.occupation.==12,:occupation] .= 7
    df2[df2.occupation.==13,:occupation] .= 7
    freqtable(df2, :occupation)

    X2 = [ones(size(df2,1),1) df2.age df2.race.==1 df2.collgrad.==1]
    y2 = df2.occupation


    function mlogit(alpha2, X2, y2)
        
        K = size(X2,2)
        J = length(unique(y2))
        N = length(y2)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y2.==j
        end
        bigAlpha = [reshape(alpha2,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X2*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    alpha2_hat_optim = optimize(alpha2 -> mlogit(alpha2, X2, y2), zeros(24,1), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha2_hat_mle = alpha2_hat_optim.minimizer
    println(alpha2_hat_mle)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Part (B)
    # Starting code from pg 10 of lecture 9
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function mlogit_gmm(beta, X2, y2)
        
        # declare dimensions
        K = size(X2,2)
        J = length(unique(y2))
        N = size(X2,1)
        
        # Initialize g
        # 15,659 x 1 matrix
        g = zeros(N*J,1)
        
        # Initialize d_matrix
        # 2237 (N) x 7 (J)
        d_matrix = zeros(N,J)
        for j=1:J
            d_matrix[:,j] = y2.==j
        end

        # Convert to stacked column vector
        d = reshape(d_matrix,(N*J,1))

        # Initialize P
        # P is a 2237 (N) x 7 (J) matrix
        bigbeta = [reshape(beta,K,J-1) zeros(K)]
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X2*bigbeta[:,j])
            dem .+= num[:,j]
        end
        
        # 2237 (N) x 7 (J)
        p_matrix = num./repeat(dem,1,J)    
        p = reshape(p_matrix,(N*J,1))

        g = d .- p
        J = (g'*I*g)[1] # (g'*I*g) produces a 1x1 matrix, which causes an error in the optimizer.  Adding the index[1] converts to scalar

        return J
    end

    beta_optim = optimize(beta -> mlogit_gmm(beta, X2, y2), alpha2_hat_mle, LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim.minimizer)
    #[0.16344358202031442; -0.02847067257863155; 0.3428123623592821; 0.4575875761020685; -0.3283729860219569; -0.02888741676115209; 
    #1.1472729594055515; -0.4140438150245014; 0.7052005132389539; -0.011523952948614788; 0.5376404608807288; -1.5050611881071523; 
    #-2.0402331018212427; -0.009607148561548219; 1.3049307035168156; -1.062240018647667; -1.2195697348833112; -0.018204331048083888; 
    #-0.14761166436872336; -1.139433083201305; 0.37553877670919417; -0.01021091823043359; -0.5519221629135022; -2.8553632107323423;;]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Part (C)
    # GMM with random starting values
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # The objective function is not globally concave.  The starting values of the optimization function have a meaningful
    # impact on the coefficient estimates

    beta_optim_cr = optimize(beta -> mlogit_gmm(beta, X2, y2), rand(Uniform(-1,1),24,1), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim_cr.minimizer)
    #[0.17354816245925314; -0.02793972422063178; 0.36005000474266213; 0.41383556100142194; -0.3751159656857626; -0.02678596468279491; 
    #1.1359424947383747; -0.41836860676873744; 0.7064328752111895; -0.011648520761012415; 0.5162936911711911; -1.4640025677631847; 
    #0.29425250324652324; -0.6339466231518581; -0.387873076395313; -0.10388720237798604; -1.1612414284450152; -0.017672854393156197; 
    #0.09991237974282322; -1.1537259917084952; 0.3860431055252124; -0.010479607876127736; -0.5084016234085749; -2.6499159154768144;;]


    beta_optim_c1 = optimize(beta -> mlogit_gmm(beta, X2, y2), ones(24,1), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim_c1.minimizer)
    #[0.16966428098976288; -0.028015162012436843; 0.33987142843041995; 0.43782724045116994; -0.28092732851051516; -0.028477298723676846; 
    #1.0926552165118797; -0.4163741650177999; 0.6881138656036029; -0.011284823536533567; 0.5349555141411668; -1.4861162810458262; 
    #-1.80065206821418; -0.010194460381658532; 1.128200473449433; -1.038627789779327; 0.20093091817554845; -0.5633564052030727; 
    #0.5857773436485283; 0.6635031119183512; 0.36329467860003634; -0.009937507554745912; -0.5340518606200493; -2.733333295407395;;]

    beta_optim_c0 = optimize(beta -> mlogit_gmm(beta, X2, y2), zeros(24,1), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim_c0.minimizer)
    #[0.16966371597656218; -0.02801514195558005; 0.3398710132599369; 0.437827237367406; -0.28093126671283997; -0.028477209717338397; 
    #1.0926555583299469; -0.4163742215976618; 0.6881090377070289; -0.011284687893600175; 0.5349547853102142; -1.4861156508734594; 
    #-1.8006450962051248; -0.01019437951958711; 1.1281899142343617; -1.0386324144006946; -0.021309485569257592; -0.7315126745786614; 
    #-0.023674862297566338; -0.00521853554121755; 0.36329681977544964; -0.009937558742990878; -0.5340517226797885; -2.73333700888174;;]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3: (i) Simulate a data set from a multinomial logit model, (ii) estimate its parameter values, 
    # (iii) verifiy that the estimates are close to the parameter values
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Part (i) Simulate a data set for a multinomial logit model
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    N3 = 10000 # Sample size
    J3 = 4 # Number of unique choices
    K3 = 3 # Number of independent variable in x

    # Generate X using a random number generator
    X3 = rand(N3,K3)

    # Set values for β such that conformability with X and J is satisfied
    # 3 x 4 matrix (last column is a vector of zeros)
    beta3 = hcat(rand(K3, J3-1), zeros(K3,1)) 

    # Generate the choice probabilities P 
    num = zeros(N3,J3)
    dem = zeros(N3)
    for j=1:J3
        num[:,j] = exp.(X3*beta3[:,j])
        dem .+= num[:,j]
    end
    P_matrix = num./repeat(dem,1,J3)    

    # Draw Preference shocks ϵ
    epsilon = rand(Uniform(0,1),N3,1)

    # Generate Y3
    Y3 = zeros(N3,1)

    for i in 1:N3
        for j in 1:J3
            Y3[i,1] += sum(P_matrix[i,j:end]) > epsilon[i,1]
        end
    end
    Y3

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Part (ii) & (iii) estimate its parameter values & verifiy that the estimates are close to the parameter values
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # The estimates for the parameter values are consistent when solving using mlogit (2.a) and mlogit_gmm (2.b), but
    # the estimates are not close t0 randomly generated parameter values

    beta_optim3 = optimize(beta -> mlogit(beta, X3, Y3), ones(9,1), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim3.minimizer)
    #[0.5472792246953092; 0.5529653020454329; 0.5359005958944228; 0.015836118333616443; 0.6951338495785081; -0.07357586200589679; 0.5233159484095276; 0.5751157549202104; 0.3148504859293158;;]

    beta_optim3 = optimize(beta -> mlogit_gmm(beta, X3, Y3), ones(9,1), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=50))
    println(beta_optim3.minimizer)
    #[0.5466284517390931; 0.5536371795675903; 0.5349656115325827; 0.012189009012934729; 0.6947233588392075; -0.07025976657183138; 0.521106209042795; 0.5721678179551871; 0.31838854742251904;;]

    println(beta3)
    #[0.6210835656757471 0.11099585187164784 0.6761044412800106 0.0; 0.40128036641417464 0.5401882664078723 0.4434353198525429 0.0; 0.6931686970545191 0.052834520612428326 0.3575158352142701 0.0]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5: Re-estimate the multinomial logit model from Question 2 using SMM
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #X5 = X2
    #y5 = y2
    #beta5 = beta_optim.minimizer
    #draws = 1000

    function mlogit_smm(beta5, X5, y5, draws)
        Random.seed!(1234)

        # declare dimensions
        K5 = size(X5,2)
        J5 = length(unique(y5))
        N5 = size(y5,1)

        # initialize g
        g5 = zeros(N5*J5,1)

        # Initialize d_matrix
        # 2237 (N) x 7 (J)
        d_matrix5 = zeros(N5,J5)
        for j=1:J5
            d_matrix5[:,j] = y5.==j
        end

        # convert to stacked column vector
        d5 = reshape(d_matrix5,(N5*J5,1))

        # simulated model moments
        bigbeta5 = [reshape(beta5,K5,J5-1) zeros(K5)]
        num5 = zeros(N5,J5)
        dem5 = zeros(N5)
        for j=1:J5
            num5[:,j] = exp.(X5*bigbeta5[:,j])
            dem5 .+= num5[:,j]
        end
        P_matrix = num5./repeat(dem5,1,J5)    

        # Generate Y3
        Yssm = zeros(N5,draws)

        for d=1:draws
            # Draw Preference shocks ϵ
            epsilon5 = rand(Uniform(0,1),N5,1)
            for i in 1:N5
                for j in 1:J5
                    Yssm[i,d] += sum(P_matrix[i,j:end]) > epsilon5[i,1]
                end
            end
        end

        # criterion function
        err = vec(y5 .- mean(Yssm; dims=2))
        # weighting matrix is the identity matrix
        # minimize weighted difference between data and moments
        J5 = err'*I*err
        return J5
    end

    beta5_optim = optimize(beta -> mlogit_smm(beta, X2, y2, 1000), beta_optim.minimizer, LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000, show_trace=true, show_every=1))
    println(beta5_optim.minimizer)
end

PS7()