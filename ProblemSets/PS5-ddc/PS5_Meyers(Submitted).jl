using Pkg
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

function ProblemSet5()
    # Read in function to create state transitions for dynamic model
    include("create_grids.jl")

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: reshaping the data
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Load in the data
    # df is a 1000 x 63 dataframe
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)


    # Create bus id variable
    # "@transform" adds additional columns
    # df is a 1000 x 64 dataframe
    # Running the code more than once does not continue to add additional rows
    df = @transform(df, bus_id = 1:size(df,1))

    # Examine data
    # CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\fall-2022\ProblemSets\PS5-ddc\\temp_df.csv", df)
    # bus_id assigns 1:1000 to each consecutive row

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Reshape from wide to long (must do this twice because DataFrames.stack() requires doing it one variable at a time)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # First reshape the decision variable
    # dfy becomes a 1000 x 23 dataframe
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    # dfy_long becomes a 20000 x 5 dataframe
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    # Adds a time column. dfy_long becomes a 20000 x 6 dataframe
    dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
    # Drops variable column. dfy_long becomes a 20000 x 5 dataframe
    select!(dfy_long, Not(:variable))

    # Next reshape the odometer variable
    # Creates dfx which is a 1000 x 21 dataframe
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    # Creates dfx_long which is a 20000 x 3 dataframe
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    # Renames value column header as Odometer
    rename!(dfx_long, :value => :Odometer)
    # Adds a time column. dfx_long becomes a 20000 x 4 dataframe
    dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
    # Drops variable column
    select!(dfx_long, Not(:variable))

    # Join reshaped df's back together
    # Creates df_long which is a 20000 x 6 dataframe
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    # First sorts by bus_id then my time
    sort!(df_long,[:bus_id,:time])

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: estimate a static version of the model
    # Y is the choice set where 0 denotes replace and 1 denotes keep
    # Odometer is the mileage (in 10,000s of miles)
    # Branded is a dummy variable meaning the manufacturer is high-end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    u_formula = @formula(Y ~ Odometer + Branded)
    u_logit = glm(u_formula, df_long, Binomial(), LogitLink())
    println(u_logit)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: read in data for dynamic model
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Load data
    # df2 is a 1000 x 63 dataframe
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df2 = CSV.read(HTTP.get(url).body, DataFrame)

    # Examine dataset
    #CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\fall-2022\ProblemSets\PS5-ddc\\temp_df2.csv", df2)

    # Keep data in wide format
    # Convert columns :Y1 through :Y20 into an array labeled Y 
    # Creates Y which is a 1000 x 20 matrix
    # Remember, Y is the decision to replace (0) or keep (1)
    Y = Matrix(df2[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

    # Create M (mileage) which is a 1000 x 20 matrix
    M = Matrix(df2[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])

    # Create X (transition state) which is a 1000 x 20 matrix
    X = Matrix(df2[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])

    # Create R (route usage) which is a 1000 x 1 vector
    R = @select(df2, :RouteUsage)

    # Create B (branded) which is a 1000 x 1 vector
    B = @select(df2, :Branded)

    # Create Z (utilization intensity) which is a 1000 x 1 vector
    Z = @select(df2, :Zst)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: generate state transition matrices
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # zval is a 101 x 1 vector
    # xval is a 201 x 1 vector
    # xtran is a (zbin * xbin) x xbin Markov transition matrix
    zval,zbin,xval,xbin,xtran = create_grids()

    # Examine Markov transition matrix
    # xtran_df = DataFrame(xtran, :auto)
    # CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\fall-2022\ProblemSets\PS5-ddc\\temp_MTM.csv", xtran_df)
    # xval[2]

    # function used in optimizer. Below section include part 3c and 3d
    @views @inbounds function loglike_function(theta, Y, M, X, B, Z, zbin, xbin, xtran)


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3c: compute the future value terms 
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        # Initializes future value array.
        FV = zeros(size(xtran,1), 2, 21)

        # Initiialize thetas using values from part 2
        #theta0 = 1.92596
        #theta1 = -0.148154
        #theta2 = 1.05919

        # Seperate thetas passed from optimizer
        theta0 = theta[1,1]
        theta1 = theta[2,1]
        theta2 = theta[3,1]

        # Input value for beta
        beta = 0.9

        for t in 20:-1:1 # Reverse loops time
            for b in 0:1 # Loops brand states
                for z in 1:zbin # Loops possible permanent route usage states
                    for x in 1:xbin # Loops possible odometer states
                        xtran_row = x + (z-1)*xbin
                        # Conditional value function for keep (dont replace)
                        V1 = theta0 + (theta1 * xval[x]) + (theta2 * b) + xtran[xtran_row,:]' * FV[((z-1)*xbin)+1: z*xbin, b+1, t+1]
                        # Conditional value function for replace
                        V0 = xtran[1+(z-1)*xbin,:]' * FV[((z-1)*xbin)+1: z*xbin, b+1, t+1]
                        FV[xtran_row,b+1,t] = beta * log(exp(V0) + exp(V1))
                    end
                end
            end
        end

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3d: construct the log likelihood
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        # construct log likelihood using FV and observes state in the data
        # (for) loop over buses (1000) and time periods (20 years)

        # Initialize the log likelihood value to be zero
        loglike = 0

        # Initialize probability matrix
        prob_keep = zeros(size(X,1), size(X,2))

        for bus in 1:size(X,1) 
            for t in 1:size(X,2)
                replaced_row = 1+(Z[bus,1]-1)*xbin # (Y = 0)
                keep_row = X[bus,t] + (Z[bus,1]-1)*xbin # (Y = 1)
                flow_utility = theta0 + (theta1 * M[bus,t]) +(theta2 * B[bus,1])
                # discounted future value component of utility (below)
                DFV = (xtran[keep_row,:] .- xtran[replaced_row,:])' * FV[replaced_row:(replaced_row + xbin -1), B[bus,1] + 1, t+1]
                diff_CVF = flow_utility + DFV # differenced conditional value function (v1t - v0t)
                # populate prob_keep matrix which is 1000 x 20
                prob_keep[bus,t] = exp(diff_CVF) / (1 + exp(diff_CVF))      
            end
        end   

        # Y is a 1000 x 20 matrix (1 = keep, 0 = replace)
        # prob_keep is a 1000 x 20 matrix
        # generates a scalar
        loglike = sum(Y .* log.(prob_keep))

        return -loglike
    end

    # start with theta values calculated in part 1
    startvals = [1.92596, -0.148154, 1.05919]

    # Confirm that function works
    #loglike_function(startvals, Y, M, X, B, Z, zbin, xbin, xtran)

    # Optimize!
    theta_hat_loglike = optimize(theta -> loglike_function(theta, Y, M, X, B, Z, zbin, xbin, xtran), startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_every = 10))
    println(theta_hat_loglike)
end