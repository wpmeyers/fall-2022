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

# Remember to keep this outside of the functions
include("create_grids.jl")

function PS6()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: Read in the data and reshape to "long" panel format
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Load data
    # df is a 1_000 x 63 dataframe
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create bus id variable
    # "@transform" adds additional columns
    # df is a 1_000 x 64 dataframe
    # bus_id assigns 1:1_000 to each consecutive row
    # Running the code more than once does not continue to add additional rows
    df = @transform(df, bus_id = 1:size(df,1))

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Reshape from wide to long (must do this twice because DataFrames.stack() requires doing it one variable at a time)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # First reshape the decision variable
    # dfy becomes a 1_000 x 23 dataframe
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    # dfy_long becomes a 20_000 x 5 dataframe
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    # Adds a time column. dfy_long becomes a 20_000 x 6 dataframe
    dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
    # Drops variable column. dfy_long becomes a 20_000 x 5 dataframe
    select!(dfy_long, Not(:variable))

    # Next reshape the odometer variable
    # Creates dfx which is a 1_000 x 21 dataframe
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    # Creates dfx_long which is a 20_000 x 3 dataframe
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    # Renames value column header as Odometer
    rename!(dfx_long, :value => :Odometer)
    # Adds a time column. dfx_long becomes a 20_000 x 4 dataframe
    dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
    # Drops variable column
    select!(dfx_long, Not(:variable))

    # Join reshaped df's back together
    # Creates df_long which is a 20_000 x 6 dataframe
    # 20,000 rows because 1_000 busses x 20 years
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    # First sorts by bus_id then my time
    sort!(df_long,[:bus_id,:time])

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: estimate a flexible logit model where the dependent variable is the replacement decision and the RHS 
    # is a fully interacted set of variables
    # Y is the choice set where 0 denotes replace and 1 denotes keep
    # Odometer is the mileage (in 10_,000s of miles)
    # Branded is a dummy variable meaning the manufacturer is high-end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    u_formula = @formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time)
    u_logit = glm(u_formula, df_long, Binomial(), LogitLink()) 
    # println(u_logit)

    # 54 x 1 vector
    u_logit_coef = coef(u_logit)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: generate state transition matrices
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Keep data in wide format
    # Convert columns :Y1 through :Y20 into an array labeled Y 
    # Creates Y which is a 1000 x 20 matrix
    # Remember, Y is the decision to replace (0) or keep (1)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

    # Create M (mileage) which is a 1000 x 20 matrix
    M = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])

    # Create X (transition state) which is a 1000 x 20 matrix
    X = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])

    # Create R (route usage) which is a 1000 x 1 vector
    R = @select(df, :RouteUsage)

    # Create B (branded) which is a 1000 x 1 vector
    B = @select(df, :Branded)

    # Create Z (utilization intensity) which is a 1000 x 1 vector
    Z = @select(df, :Zst)

    # zval is a 101 x 1 vector.  Values range from [0.25, 1.25] in 0.01 increments
    # xval is a 201 x 1 vector.  Values range from [0.000, 25.000] in 0.125 increments
    # xtran is a (zbin * xbin) x xbin Markov transition matrix
    zval,zbin,xval,xbin,xtran = create_grids()

    # Examine Markov transition matrix
    xtran_df = DataFrame(xtran, :auto)
    # CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\fall-2022\ProblemSets\PS6-ccp\\temp_MTM.csv", xtran_df)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: compute the future value terms
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Create a dataframe that contains all possible values for the observed variables 
    df_OR = DataFrame(Odometer = kron(ones(zbin), xval))
    df_RU = DataFrame(RouteUsage = kron(ones(xbin), zval))    
    df_B = DataFrame(Branded = zeros(size(df_OR,1)))
    df_T = DataFrame(time = zeros(size(df_OR,1)))

    df_state = hcat(df_OR, df_RU, df_B, df_T) 

    # Initialize beta
    beta = 0.9

    function CCP_function(df_long, X, Z, df_state, xtran, xbin, u_logit, beta)

        # Initialize future value array
        # 20_301 x 2 x 21 matrix
        FV = zeros(size(xtran,1), 2, 21)

        for t in 2:20 # Loops time
            for b in 0:1 # Loops brand states
                df_state.time .= t
                df_state.Branded .= b
                p0 = predict(u_logit, df_state)   # 20_301 x 1 vector
                FV[:,b+1,t] = -beta .* log.(p0)
            end
        end

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Multiply the state transitions (xtran) by the future value term.
        # This maps the CCP's from the each-possible-state-is-a-row data to the actual data frame we used to estimate the
        # flexible logit in question 2 (df_long)
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        # Initialize FVT1 which is a 1_000 x 20 matrix
        FVT1 = zeros(size(X,1), size(X,2))

        for bus in 1:size(X,1) 
            for t in 1:size(X,2)
                row1 = X[bus,t] + (Z[bus,1]-1)*xbin # Keep engine
                row0 = 1+(Z[bus,1]-1)*xbin # Replaced engine
                FVT1[bus,t] = (xtran[row1,:] .- xtran[row0,:])' * FV[row0:(row0 + xbin -1), B[bus,1] + 1, t+1]     
            end
        end

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Convert FVT1 into "long panel" format
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        # Initialize FVT1_long (20_000 x 1 vector)
        # Confirmed bus and time indexing matches the original long dataset
        FVT1_long = FVT1'[:]

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3C: Estimate the structural parameters
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        df_long = @transform(df_long, FV = FVT1_long)

        theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset=df_long.FV)

        return theta_hat_ccp_glm

    end

    CCP_function(df_long, X, Z, df_state, xtran, xbin, u_logit, beta)

end

@time PS6()