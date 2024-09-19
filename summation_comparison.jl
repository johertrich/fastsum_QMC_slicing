# reproduce the results from Section 4.3 (and corresponding experiments from the appendix) of the paper

using Distributions
using HDF5
using LinearAlgebra
using Sobol
using MLDatasets: MNIST, FashionMNIST

include("src/fastsum.jl")
include("src/basis_functions.jl")
include("src/utils.jl")


# read console arguments
kernel_nr=0
dataset_nr=0
if size(ARGS,1)>=2
    kernel_nr=parse(Int64,ARGS[2])
end
if size(ARGS,1)>=1
    dataset_nr=parse(Int64,ARGS[1])
end


# load dataset
if dataset_nr==0
    d=16
    N=20000
    fid = h5open("datasets/letter.h5","r")
    x=reshape(collect(fid["data"]),d,N)'
elseif dataset_nr==1
    d=20
    N=60000
    # load MNIST dataset
    dataset=MNIST()
    X=dataset.features
    X=reshape(X,:,N)'
    X=X.-mean(X,dims=1) # center

    # PCA
    U,s,V=svd(X')
    mat=U[:,1:d]
    x=X*mat
elseif dataset_nr==2
    d=30
    N=60000
    # load FashionMNIST dataset
    dataset=FashionMNIST()
    X=dataset.features
    X=reshape(X,:,N)'
    X=X.-mean(X,dims=1) # center

    # PCA
    U,s,V=svd(X')
    mat=U[:,1:d]
    x=X*mat
elseif dataset_nr==3
    d=784
    N=60000

    # load MNIST dataset
    dataset=MNIST()
    X=dataset.features
    X=reshape(X,:,N)'
    X=X.-mean(X,dims=1) # center
elseif dataset_nr==4
    d=784
    N=60000

    # load FashionMNIST dataset
    dataset=FashionMNIST()
    X=dataset.features
    X=reshape(X,:,N)'
    X=X.-mean(X,dims=1) # center
end

x=convert(Array{Float64},x)
y=copy(x)
x_weights=ones(N)
soboleng=SobolSeq(d)


# if ground truth is already computed dont compute it again...
if !isdir("kernel_sums")
    mkdir("kernel_sums")
end
load=false
if isfile("kernel_sums/sum_for_ds_"*string(dataset_nr)*"_kernel_"*string(kernel_nr)*".h5")
    fid = h5open("kernel_sums/sum_for_ds_"*string(dataset_nr)*"_kernel_"*string(kernel_nr)*".h5","r")
    s_true=reshape(collect(fid["s_true"]),N)
    med=collect(fid["med"])
    load=true
else
    # choose kernel parameter by median rule. med=sigma=beta=1/alpha for Gauss/Matern/Laplace
    med=median_distance(x,y,1000)
end


# set kernel-specific parameters
if kernel_nr==0
    kernel(x,y)=dropdims(Gauss(reshape(x-y,1,:),med),dims=1)
    fourier_fun(h,scale)=Gaussian_kernel_fun_ft(h,d,scale^2)
    n_ft=128
    x_range=0.3
elseif kernel_nr==1
    n_ft=512
    nu=3.5
    kernel(x,y)=dropdims(Matern(reshape(x-y,1,:),med,3),dims=1)
    fourier_fun(h,scale)=Matern_kernel_fun_ft(h,d,scale,nu)
    x_range=0.2
elseif kernel_nr==2
    n_ft=512
    nu=1.5
    kernel(x,y)=dropdims(Matern(reshape(x-y,1,:),med,1),dims=1)
    fourier_fun(h,scale)=Matern_kernel_fun_ft(h,d,scale,nu)
    x_range=0.2
elseif kernel_nr==3
    n_ft=1024
    kernel(x,y)=dropdims(Laplace(reshape(x-y,1,:),1/med),dims=1)
    fourier_fun(h,scale)=Matern_kernel_fun_ft(h,d,scale,0.5)
    x_range=0.1
elseif kernel_nr==4
    kernel(x,y)=dropdims(Riesz(reshape(x-y,1,:)),dims=1)
    sliced_factor=compute_sliced_factor(d)
end

if kernel_nr<=3 && d>=100
    n_ft=2*n_ft
end


# compute ground truth if not loaded
if !load
    # compile naive_kernel_sum
    @time naive_kernel_sum(x[1:5,:],x_weights[1:5],y[1:5,:],kernel)

    # for Gauss sigma=med, for Laplace alpha=1/med, for Matern beta=med
    s_true = @time naive_kernel_sum(x,x_weights,y,kernel)
    fid = h5open("kernel_sums/sum_for_ds_"*string(dataset_nr)*"_kernel_"*string(kernel_nr)*".h5","w")
    fid["s_true"]=s_true
    fid["med"]=med
end

# initialize samplers
normal_distr=Normal()
mvt7=MvTDist(7.,zeros(d),Matrix(I,d,d))
mvt3=MvTDist(3.,zeros(d),Matrix(I,d,d))
mvt1=MvTDist(1.,zeros(d),Matrix(I,d,d))

# take rff_factor*P Fourier features to obtain a similar computation time for RFF and slicing
rff_factor=2


function all_errors(P,trials=10)
    println("")
    println("")
    println("P=",P)
    println("")
    fid = h5open("distance_MMD_projs/d"*string(d)*"/P_sym"*string(P)*".h5","r")
    xis_mmd=reshape(collect(fid["xis"]),d,P)'
    errors_dict=Dict()
    times_dict=Dict()
    if kernel_nr <= 3
        # RFF computations
        name="RFF"
        println("RFF")
        errors=zeros(trials)
        times=zeros(trials)
        for trial=1:trials
            if kernel_nr==0
                xis=rand(normal_distr,P*rff_factor,d)
            elseif kernel_nr==1
                xis=rand(mvt7,P*rff_factor)'
            elseif kernel_nr==2
                xis=rand(mvt3,P*rff_factor)'
            elseif kernel_nr==3
                xis=rand(mvt1,P*rff_factor)'
            end
            tic=time()
            s=RFF(x,y,x_weights,med,xis)
            toc=time()-tic
            err=sum(abs.(s-s_true))/sum(abs.(s_true))
            errors[trial]=err
            times[trial]=toc
        end
        errors_dict[name]=errors
        times_dict[name]=times
    
        # ORF computations
        println("ORF")
        name="ORF"
        errors=zeros(trials)
        times=zeros(trials)
        for trial=1:trials
            xis_orth=zeros(0,d)
            for i=1:ceil(Int64,P*rff_factor/d)
                rand_mat=rand(normal_distr,d,d)
                Q,R=qr(rand_mat)
                xis_orth=vcat(xis_orth,Q)
            end
            xis_orth=xis_orth[1:P*rff_factor,:]
            if kernel_nr==0
                xis_s=rand(normal_distr,P*rff_factor,d)
            elseif kernel_nr==1
                xis_s=rand(mvt7,P*rff_factor)'
            elseif kernel_nr==2
                xis_s=rand(mvt3,P*rff_factor)'
            elseif kernel_nr==3
                xis_s=rand(mvt1,P*rff_factor)'
            end
            if kernel_nr <= 3
                xis_scale=sqrt.(sum(xis_s.^2,dims=2))
                xis=xis_orth.*xis_scale
                tic=time()
                s= RFF(x,y,x_weights,med,xis)
                toc=time()-tic
                err=sum(abs.(s-s_true))/sum(abs.(s_true))
                errors[trial]=err
                times[trial]=toc
            end
        end
        errors_dict[name]=errors
        times_dict[name]=times
        
        # QMC slicing with RFF to compute the 1D kernel problems
        println("QMRFFS")
        name="QMRFFS"
        P_h=convert(Int64,round(.5*P))
        fid = h5open("distance_MMD_projs/d"*string(d)*"/P_sym"*string(P_h)*".h5","r")
        xis_mmd_h=reshape(collect(fid["xis"]),d,P_h)'
        errors=zeros(trials)
        times=zeros(trials)
        k=10
        for trial=1:trials
            # get unbiased estimator for random rotation
            rot=rand(normal_distr,d,d)
            rot,R=qr(rot)
            xis=xis_mmd_h*rot
            if kernel_nr==0
                xis_s=rand(normal_distr,k*P_h,d)
            elseif kernel_nr==1
                xis_s=rand(mvt7,k*P_h)'
            elseif kernel_nr==2
                xis_s=rand(mvt3,k*P_h)'
            elseif kernel_nr==3
                xis_s=rand(mvt1,k*P_h)'
            end
            xis_scale=reshape(sqrt.(sum(xis_s.^2,dims=2)),P_h,:)
            tic=time()
            s= slicing_RFF(x,y,x_weights,med,xis,xis_scale)
            toc=time()-tic
            err=sum(abs.(s-s_true))/sum(abs.(s_true))
            errors[trial]=err
            times[trial]=toc
        end    
        errors_dict[name]=errors
        times_dict[name]=times
        
        if kernel_nr==0
            # QMC RFF, only available for Gaussian
            name="SOBOLRFF"
            println("SOBOLRFF")
            errors=zeros(trials)
            times=zeros(trials)

            P_sobol=2^convert(Int64,ceil(log2(rff_factor*P))) # sobol sequnces work best for powers of 2
            
            xis_01=reduce(hcat, next!(soboleng) for i =1:P_sobol)'
            xis_base=sqrt(2)*erfinv.(2*xis_01.-1)
            
            for trial=1:trials
                rot=rand(normal_distr,d,d)
                rot,R=qr(rot)
                xis=xis_base*rot
            
                tic=time()
                s= RFF(x,y,x_weights,med,xis)
                toc=time()-tic
                err=sum(abs.(s-s_true))/sum(abs.(s_true))
                errors[trial]=err
                times[trial]=toc
            end
            errors_dict[name]=errors
            times_dict[name]=times 
        end
        
    end
    
    # Slicing with fast Fourier summation and iid directions
    println("MCS")
    name="MCS"
    errors=zeros(trials)
    times=zeros(trials)
    for trial=1:trials
        xis=rand(normal_distr,P,d)
        xis=xis./sqrt.(sum(xis.^2,dims=2))
        tic=time()
        if kernel_nr<=3
            s= fastsum_fft(x,y,x_weights,med,n_ft,x_range,fourier_fun,xis)
        elseif kernel_nr==4
            s= fastsum_energy(x,y,x_weights,sliced_factor,xis)
        end
        toc=time()-tic
        err=sum(abs.(s-s_true))/sum(abs.(s_true))
        errors[trial]=err
        times[trial]=toc
    end
    errors_dict[name]=errors
    times_dict[name]=times
    
    
    # Slicing with fast Fourier summation and QMC directions
    println("QMCFS")
    name="QMCFS"
    errors=zeros(trials)
    times=zeros(trials)
    for trial=1:trials
        # get unbiased estimator for random rotation
        rot=rand(normal_distr,d,d)
        rot,R=qr(rot)
        xis=xis_mmd*rot
        tic=time()
        if kernel_nr<=3
            s= fastsum_fft(x,y,x_weights,med,n_ft,x_range,fourier_fun,xis)
        elseif kernel_nr==4
            s= fastsum_energy(x,y,x_weights,sliced_factor,xis)
        end
        toc=time()-tic
        err=sum(abs.(s-s_true))/sum(abs.(s_true))
        errors[trial]=err
        times[trial]=toc
    end   
    errors_dict[name]=errors
    times_dict[name]=times 
    
    return errors_dict, times_dict
end

# small run for compiling all functions
all_errors(10)

if !isdir("results")
    mkdir("results")
end
tic=time()
errors=Dict()
times=Dict()

# simulate all cases
for P=[10*2^k for k=0:9]
    errors_dict,times_dict=all_errors(P)
    for (key,value) in errors_dict
        if haskey(errors,key)
            errors[key]=vcat(errors[key],reshape(value,1,:))
        else
            errors[key]=reshape(value,1,:)
        end
        if haskey(times,key)
            times[key]=vcat(times[key],reshape(times_dict[key],1,:))
        else
            times[key]=reshape(times_dict[key],1,:)
        end
        println(key,": Error: ",mean(value),"+-",std(value),", Time: ",mean(times_dict[key]))
    end
    toc_main=time()-tic
    println("After ",P," ",toc_main)
    flush(stdout)
end

# save results

fid = h5open("results/results_ds_"*string(dataset_nr)*"_kernel_"*string(kernel_nr)*"_errors.h5","w")
for (key,value) in errors
    fid[key]=value
end
fid = h5open("results/results_ds_"*string(dataset_nr)*"_kernel_"*string(kernel_nr)*"_times.h5","w")
for (key,value) in times
    fid[key]=value
end

