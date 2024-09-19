# reproduce the results from Section 4.2 (and corresponding experiments from the Appendix) from the paper

using Distributions
using Sobol
using LinearAlgebra
using HDF5

include("src/basis_functions.jl")

# read kernel factor and dimension from console
kernel_factor=1.0
d=3
if size(ARGS,1)>=2
    kernel_factor=parse(Float64,ARGS[2])
end
if size(ARGS,1)>=1
    d=parse(Int64,ARGS[1])
end

# number of test points
n_points=1000

# number of runs
n_runs=50

# factor for the negative distance kernel
sliced_factor=compute_sliced_factor(d)

# initialize distributions
normal_distr=Normal()
mvt7=MvTDist(7.,zeros(d),Matrix(I,d,d))
mvt3=MvTDist(3.,zeros(d),Matrix(I,d,d))
mvt1=MvTDist(1.,zeros(d),Matrix(I,d,d))
soboleng=SobolSeq(d)

# number of projections
Ps=[10*2^k for k=0:9]
Ps_sd=Vector{Int64}(undef,0)
for P=1:5100 # spherical designs are only available for special values of P
    if isfile("distance_MMD_projs/d3/spherical_designs/P"*string(P)*".h5")
        push!(Ps_sd,P)
    end
end
Ps_sobol=[2^k for k=4:12] # sobol sequences work best for powers of 2

# generate MMD slicing directions
function get_MMD_xis(P)
    fid = h5open("distance_MMD_projs/d"*string(d)*"/P_sym"*string(P)*".h5","r")
    return reshape(collect(fid["xis"]),d,P)'
end

# generate orthogonal slicing directions
function get_orth_xis(P)
    xis_orth=zeros(0,d)
    for i=1:ceil(Int64,P/d)
        rand_mat=rand(normal_distr,d,d)
        Q,R=qr(rand_mat)
        xis_orth=vcat(xis_orth,Q)
    end
    return xis_orth[1:P,:]
end

# generate iid slicing directions
function get_MC_xis(P)
    xis=rand(normal_distr,P,d)
    return xis./sqrt.(sum(xis.^2,dims=2))
end

# generate random features for Gauss
function get_RFF_Gauss_xis(P)
    return rand(normal_distr,P,d)
end

# generate random features for Matern-3/2
function get_RFF_Matern1_xis(P)
    return rand(mvt3,P)'
end

# generate random features for Matern-7/2
function get_RFF_Matern3_xis(P)
    return rand(mvt7,P)'
end

# generate random features for Laplace
function get_RFF_Laplace_xis(P)
    return rand(mvt1,P)'
end

# generate orthogonal random features for Gauss
function get_ORF_Gauss_xis(P)
    xis_orth=get_orth_xis(P)
    xis_s=rand(normal_distr,P,d)
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    return xis_orth.*xis_scale
end

# generate orthogonal random features for Matern 3/2
function get_ORF_Matern1_xis(P)
    xis_orth=get_orth_xis(P)
    xis_s=rand(mvt3,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    return xis_orth.*xis_scale
end

# generate orthogonal random features for Matern 5/2
function get_ORF_Matern3_xis(P)
    xis_orth=get_orth_xis(P)
    xis_s=rand(mvt7,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    return xis_orth.*xis_scale
end

# generate orthogonal random features for Laplace
function get_ORF_Laplace_xis(P)
    xis_orth=get_orth_xis(P)
    xis_s=rand(mvt1,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    return xis_orth.*xis_scale
end

# generate Sobol slicing directions
function get_Sobol_xis(P)
    soboleng=SobolSeq(d)
    xis_01=reduce(hcat, next!(soboleng) for i =1:P)'
    xis_base=sqrt(2)*erfinv.(2*xis_01.-1)
    zero_inds=dropdims(sum(xis_base.^2,dims=2),dims=2).==0
    xis_base[zero_inds,:]=rand(normal_distr,sum(zero_inds),d) # treat zero correctly
    return xis_base./sqrt.(sum(xis_base.^2,dims=2))
end

# generate sobol RFF directions for Gauss
function get_Sobol_RFF_xis(P)
    soboleng=SobolSeq(d)
    xis_01=reduce(hcat, next!(soboleng) for i =1:P)'
    xis_base=sqrt(2)*erfinv.(2*xis_01.-1)
    return xis_base
end

# load spherical designs
function get_SD_xis(P)
    # load spherical design direction. The sources of that are taken from 
    # https://www-user.tu-chemnitz.de/~potts/workgroup/graef/quadrature/index.php.en
    fid = h5open("distance_MMD_projs/d3/spherical_designs/P"*string(P)*".h5","r")
    xis_sd=reshape(collect(fid["xis"]),d,P)'
    return xis_sd
end

# create dict with directions
xis_dict=Dict("MMD" => get_MMD_xis, "orth" => get_orth_xis, "MC" => get_MC_xis, 
        "RFF-Gauss" => get_RFF_Gauss_xis, "RFF-Matern1" => get_RFF_Matern1_xis, 
        "RFF-Matern3" => get_ORF_Matern3_xis, "RFF-Laplace" => get_RFF_Laplace_xis, 
        "ORF-Gauss" => get_ORF_Gauss_xis, "ORF-Matern1" => get_ORF_Matern1_xis, 
        "ORF-Matern3" => get_ORF_Matern3_xis, "ORF-Laplace" => get_ORF_Laplace_xis, 
        "Sobol" => get_Sobol_xis,  "Sobol-RFF" => get_Sobol_RFF_xis, "SD" => get_SD_xis)
        
# dict indicating the numbers P of directions for each method.
Ps_dict=Dict("MMD" => Ps, "orth" => Ps, "MC" => Ps, "RFF-Gauss" => Ps, "RFF-Matern1" => Ps, 
        "RFF-Matern3" => Ps, "RFF-Laplace" => Ps, "ORF-Gauss" => Ps, "ORF-Matern1" => Ps, 
        "ORF-Matern3" => Ps, "ORF-Laplace" => Ps, "Sobol" => Ps_sobol,  "Sobol-RFF" => Ps_sobol, 
        "SD" => Ps_sd)


function compute_RFF_error(x,gt,scale,xis)
    # compute the mean error of the kernel approximation with RFF
    x_proj=xis*x'/scale
    cos_x=cos.(x_proj)
    cos_x_sum=dropdims(mean(cos_x,dims=1),dims=1)
    mean_error=mean(abs.(gt-cos_x_sum))
    return mean_error
end

function compute_slicing_error(x,gt,xis,basis_f)
    # compute the mean error of the kernel approximation with slicing
    x_proj=xis*x'
    f_vals=basis_f(abs.(x_proj))
    f_val_mean=dropdims(mean(f_vals,dims=1),dims=1)
    mean_error=mean(abs.(gt-f_val_mean))
    return mean_error
end



function compute_errors(x,kernel_nr,scale)
    # set kernel specific parameters and select methods
    if kernel_nr==0 # Gauss
        gt=Gauss(x,scale)
        basis_f=x -> Gauss_f(x,scale,d)
        RFF_labels=["RFF-Gauss","ORF-Gauss","Sobol-RFF"]
    elseif kernel_nr==1 # Matern 7/2
        gt=Matern(x,scale,3)
        basis_f=x -> Matern_f(x,scale,3,d)
        RFF_labels=["RFF-Matern3","ORF-Matern3"]
    elseif kernel_nr==2 # Matern 3/2
        gt=Matern(x,scale,1)
        basis_f=x -> Matern_f(x,scale,1,d)
        RFF_labels=["RFF-Matern1","ORF-Matern1"]
    elseif kernel_nr==3 # Matern 3/2
        gt=Laplace(x,1/scale)
        basis_f=x -> Laplace_f(x,1/scale,d)
        RFF_labels=["RFF-Laplace","ORF-Laplace"]
    elseif kernel_nr==4 # negative distance
        gt=Riesz(x)
        basis_f=x -> Riesz_f(x,sliced_factor)
        RFF_labels=[]
    end
    errors_dict=Dict()
    if d==3
        labels=["MMD", "Sobol", "orth", "SD", "MC"]
    else
        labels=["MMD", "Sobol", "orth", "MC"]
    end
    
    # compute errors
    for label in labels
        errors_array=[]
        for run in 1:n_runs
            errors=[compute_slicing_error(x,gt,xis_dict[label](P),basis_f) for P in Ps_dict[label]]
            push!(errors_array,errors)
        end
        errors_dict[label]=[mean([errors_array[i][j] for i in 1:n_runs]) for j in 1:size(errors_array[1],1)]
        println("\n\nErrors for ",label," and kernel ",kernel_nr," for d=",d,":")
        for i in 1:size(Ps_dict[label],1)
            println("Error for P=",Ps_dict[label][i],": ",errors_dict[label][i])
        end
    end    
    for label in RFF_labels
        errors_array=[]
        for run in 1:n_runs
            errors=[compute_RFF_error(x,gt,scale,xis_dict[label](P)) for P in Ps_dict[label]]
            push!(errors_array,errors)
        end
        errors_dict[label]=[mean([errors_array[i][j] for i in 1:n_runs]) for j in 1:size(errors_array[1],1)]
        println("\n\nErrors for ",label," and kernel ",kernel_nr," for d=",d,":")
        for i in 1:size(Ps_dict[label],1)
            println("Error for P=",Ps_dict[label][i],": ",errors_dict[label][i])
        end
    end
    return errors_dict
end

# create test data
x=0.1*rand(normal_distr,n_points,d)
x_norm=sqrt.(sum(x.^2,dims=2))
med=median(x_norm)


if !isdir("qmc_comp")
    mkdir("qmc_comp")
end

for kernel_nr = 0:4
    if kernel_nr==4 && kernel_factor!=1.0
        continue
    end
    errors_dict=compute_errors(x,kernel_nr,med)
    # save results
    fid = h5open("qmc_comp/qmc_comp_dim_"*string(d)*"_kernel_"*string(kernel_nr)*"_kernel_factor_"*string(kernel_factor)*".h5","w")
    for (key,value) in errors_dict
        fid[key]=value
    end
    fid["Ps"]=Ps
    fid["Ps-sobol"]=Ps_sobol
    fid["Ps-sd"]=Ps_sd
    fid["scale"]=med
end
