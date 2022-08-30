#1. Initializing variables and practice with basic matrix operations
using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
Random.seed!(1234)

function q1()
    #i. A10×7 - random numbers distributed U [−5,10]
    A=rand(Uniform(-5,10),10,7)

    #ii. B10×7 - random numbers distributed N (−2,15) [st dev is 15]
    B=rand(Normal(-2,15),10,7)

    #iii. C5×7 - the first 5 rows and first 5 columns of A and the last 
    #two columns and first 5 rows of B
    Aintermediate=A[1:5,1:5]
    Bintermediate=B[1:5,6:7]
    C=hcat(Aintermediate,Bintermediate)

    #iv. D10×7 - where Di,j = Ai,j if Ai,j ≤ 0, or 0 otherwise
    Dintermediate=collect(A.<0)
    D=Dintermediate.*A 

    #(b) Use a built-in Julia function to list the number of elements of A
    length(A)

    #(c) Use a series of built-in Julia functions to list the number of 
    #unique elements of D
    length(unique(D))

    #(d) Using the reshape() function, create a new matrix called E which 
    #is the ‘vec’ operator applied to B. Can you find an easier way to 
    #accomplish this?
    E = reshape(B,(70,1))

    #vectorization converts a matrix into a column vector
    #The "vec" function is an alternative (easier) method "E=vec(B)"

    #(e) Create a new array called F which is 3-dimensional and contains 
    #A in the first column of the third dimension and B in the second 
    #column of the third dimension
    F=[A;;;B]

    #(f) Use the permutedims() function to twist F so that it is now 
    #F2×10×7 instead of F10×7×2. Save this new matrix as F.
    F=permutedims(F,[3,1,2])

    #(g) Create a matrix G which is equal to B⊗C (the Kronecker product 
    #of B and C). 
    G=kron(B, C)

    #What happens when you try C⊗F?
    #G=kron(C, F)
    #The user receives an error

    #(h) Save the matrices A, B, C, D, E, F and G as a .jld file named 
    #matrixpractice.
    jldsave(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\Problem Sets\\matrixpractice.jld"; A, B, C, D, E, F, G)

    #(i) Save only the matrices A, B, C, and D as a .jld file called 
    #firstmatrix
    jldsave(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\Problem Sets\\firstmatrix.jld"; A, B, C, D,)

    #(j) Export C as a .csv file called Cmatrix. You will first need to 
    #transform C into a DataFrame.
    C_DataFrame = DataFrame(C, :auto)
    CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\Problem Sets\\Cmatrix.csv", C_DataFrame)

    #(k) Export D as a tab-delimited .dat file called Dmatrix. You will 
    #first need to transform D into a DataFrame.
    D_DataFrame = DataFrame(D, :auto)
    CSV.write(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\Problem Sets\\Dmatrix.dat", D_DataFrame)

    return A, B, C, D
end

A, B, C, D = q1()

#2. Practice with loops and comprehensions

function q2(A, B, C)
    #(a) Write a loop or use a comprehension that computes the 
    #element-by-element product of A and B. Name the new matrix AB. 
    AB = similar(A)
    for i in 1:size(A,1)
        for j in 1:size(A,2)
            AB[i,j] = A[i,j]*B[i,j]
        end
    end

    #Create a matrix called AB2 that accomplishes this task without a 
    #loop or comprehension.
    AB2=A.*B

    #(b) Write a loop that creates a column vector called Cprime which 
    #contains only the elements of C that are between -5 and 5 (inclusive). 
    Cprime_intermediate = similar(C)
    for i in 1:size(C,1)
        for j in 1:size(C,2)
            if -5 <= C[i,j] <= 5
                Cprime_intermediate[i,j] = C[i,j]
            else
                Cprime_intermediate[i,j] = 0
            end
        end
    end

    Cprime_intermediate = vec(Cprime_intermediate)
    Cprime = filter(x-> x!=0, Cprime_intermediate)

    #Create a vector called Cprime2 which does this calculation without 
    #a loop
    Cprime2_intermediate = vec(C)
    Cprime2 = filter(x-> 5>=x>=-5, Cprime2_intermediate)

    #(c) Using loops or comprehensions, create a 3-dimensional array 
    #called X that is of dimension N × K × T where N = 15,169, K = 6, 
    #and T = 5. For all t, the columns of X should be (in order):
    X = zeros(15_169, 6)

    for t in 1:5

        K1 = ones(15_169,1)

        K2 = ones(15_169,1)
        K2_int = rand(Uniform(0,1),15_169,1)
        for i in 1:size(K2_int,1)
            if K2_int[i,1] <= .75*(6-t)/5
                K2[i,1] = 1
            else
                K2[i,1] = 0
            end
        end

        K3 = rand(Normal(15+t-1,5t-5),15_169,1)

        K4 = rand(Normal(π*(6-t)/3,1/ℯ),15_169,1)

        K5 =  rand(Binomial(20,0.6),15_169,1)

        K6 = rand(Binomial(20,0.5),15_169,1)

        T = hcat(K1, K2, K3, K4, K5, K6)
        
        if t == 1
            global X
            X = T
        else
            X = cat(X,T;dims=3)
        end

    end

    #(d) Use comprehensions to create a matrix β which is K × T and whose 
    #elements evolve across time in the following fashion:
    R1_int = [.75 + t/4 for t in 1:5]
    R1 = reshape(R1_int, (1, 5))

    R2_int = [log(t) for t in 1:5]
    R2 = reshape(R2_int, (1, 5))

    R3_int = [-sqrt(t) for t in 1:5]
    R3 = reshape(R3_int, (1, 5))

    R4_int = [(ℯ^t)-ℯ^(t+1) for t in 1:5]
    R4 = reshape(R4_int, (1, 5))

    R5_int = [t for t in 1:5]
    R5 = reshape(R5_int, (1, 5))

    R6_int = [t/3 for t in 1:5]
    R6 = reshape(R6_int, (1, 5))

    β = cat(R1, R2, R3, R4, R5, R6; dims = 1)

    #(e) Use comprehensions to create a matrix Y which is N × T defined 
    #by Yt = Xtβt + εt where εt iid∼ N (0,σ = .36)
    Y = [X[:,:,t] * β[:,t] .+ rand(Normal(0,.36)) for t in 1:5]

end

q2(A, B, C)

#3. Reading in Data and calculating summary statistics
function q3()

    #(a) Clear the workspace and import the file nlsw88.csv into Julia 
    #as a DataFrame. Make sure you appropriately convert missing values 
    #and variable names. Save the result as nlsw88.jld.

    nlsw88 = CSV.read(raw"C:\Users\Otto\OneDrive\Desktop\ECON 6343\Problem Sets\\nlsw88.csv", DataFrame)
    jldsave("nlsw88.jld2"; nlsw88)

    #(b) What percentage of the sample has never been married? 
    table_never_married = freqtable(nlsw88.never_married)
    prop(table_never_married)

    #What percentage are college graduates?
    table_collgrad = freqtable(nlsw88.collgrad)
    prop(table_collgrad)

    #(c) Use the freqtable() function to report what percentage of the 
    #sample is in each race category
    table_race = freqtable(nlsw88.race)
    prop(table_race)

    #(d) Use the describe() function to create a matrix called 
    #summarystats which lists the mean, median, standard deviation, min, 
    #max, number of unique elements, and interquartile range (75th 
    #percentile minus 25th percentile) of the data frame. How many grade 
    #observations are missing?
    summarystats = describe(nlsw88)
    summarystats
    #Two (2) grade observations are missing

    #(e) Show the joint distribution of industry and occupation using a cross-tabulation.
    freqtable(nlsw88.industry, nlsw88.occupation)

    #(f) Tabulate the mean wage over industry and occupation categories. Hint: you should first
    #subset the data frame to only include the columns industry, occupation and wage. You
    #should then follow the “split-apply-combine” directions here.
    new_table = select(nlsw88, 11, 12, 14)
    gdf = groupby(new_table, :industry)

    #combine(gdf, :)
    #gdf1 = groupby(gdf[1], :occupation)

end

q3()


#4. Practice with functions
function q4()
    #(a) Load firstmatrix.jld.
    jldsave("firstmatrix.jld"; A, B, C, D)
    load("firstmatrix.jld")

    #(b) Write a function called matrixops that takes as inputs the 
    #matrices A and B from question (a) of problem 1 and has three 
    #outputs: 
    #(i) the element-by-element product of the inputs, 
    #(ii) the product A'B, and 
    #(iii) the sum of all the elements of A+B.
    function matrixops(x,y)
    #matrixops (i) does element-by-element product of two matrices (ii) does
    #element by element multiplication of transpose of first matrix by second
    #matrix (iii) addition of two matrices
        if size(x) == size(y)
            return .*(x,y)
            return x'*y
            return x+y
        else
            println("Inputs must have the same size !!!!!!!!!!!!")
        end
    end


    #(c) Starting on line 2 of the function, write a comment that 
    #explains what matrixops does.

    #(d) Evaluate matrixops() using A and B from question (a) of 
    #problem 1
    matrixops(A, B)

    #(f) Evaluate matrixops using C and D from question (a) of problem 1. 
    #What happens?
    matrixops(C, D)

    #(g) Now evaluate matrixops.m using ttl_exp and wage from nlsw88.jld. Hint: before doing this, you will need to convert the data frame columns to Arrays. e.g.
    #convert(Array,nlsw88.ttl_exp), depending on what you called the data frame
    #object [I called it nlsw88].
    convert(Array,nlsw88.ttl_exp)
    convert(Array,nlsw88.wage)
    matrixops(nlsw88.ttl_exp,nlsw88.wage)
    
    #(h) Wrap a function definition around all of the code for question 4. Call the function q4().
    #The function should have no inputs or outputs. At the very bottom of your script you
    #should add the code q4().

end

q4()