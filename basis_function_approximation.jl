# reproduce the results from Section 4.2 (and Appendix G.1) from the paper

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

# create dict with directions
xis_dict=Dict("MMD" => [], "orth" => [], "MC" => [], "RFF-Gauss" => [], "RFF-Matern1" => [], "RFF-Matern3" => [],
        "RFF-Laplace" => [], "ORF-Gauss" => [], "ORF-Matern1" => [], "ORF-Matern3" => [], "ORF-Laplace" => [], 
        "Sobol" => [],  "Sobol-RFF" => [], "SD" =>[])
for P in Ps
    # load MMD directions (computed a priori, see helpers directory)
    fid = h5open("distance_MMD_projs/d"*string(d)*"/P_sym"*string(P)*".h5","r")
    xis_mmd=reshape(collect(fid["xis"]),d,P)'
    push!(xis_dict["MMD"],xis_mmd)
    
    # Compute features for RFF with Gauss/Student-t/Cauchy distribution
    xis_rff_gauss=rand(normal_distr,P,d)
    push!(xis_dict["RFF-Gauss"],xis_rff_gauss)
    xis_rff_matern=rand(mvt1,P)'
    push!(xis_dict["RFF-Laplace"],xis_rff_matern)
    xis_rff_matern=rand(mvt3,P)'
    push!(xis_dict["RFF-Matern1"],xis_rff_matern)
    xis_rff_matern=rand(mvt7,P)'
    push!(xis_dict["RFF-Matern3"],xis_rff_matern)
    
    # Orthogonal features for slicing
    xis_orth=zeros(0,d)
    for i=1:ceil(Int64,P/d)
        rand_mat=rand(normal_distr,d,d)
        Q,R=qr(rand_mat)
        xis_orth=vcat(xis_orth,Q)
    end
    xis_orth=xis_orth[1:P,:]
    push!(xis_dict["orth"],xis_orth)
    
    # Rescale orthogonal features for ORF
    xis_s=rand(normal_distr,P,d)
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    xis=xis_orth.*xis_scale
    push!(xis_dict["ORF-Gauss"],xis)
    
    xis_s=xis_s=rand(mvt1,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    xis=xis_orth.*xis_scale
    push!(xis_dict["ORF-Laplace"],xis)
    
    xis_s=xis_s=rand(mvt3,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    xis=xis_orth.*xis_scale
    push!(xis_dict["ORF-Matern1"],xis)
    
    xis_s=xis_s=rand(mvt7,P)'
    xis_scale=sqrt.(sum(xis_s.^2,dims=2))
    xis=xis_orth.*xis_scale
    push!(xis_dict["ORF-Matern3"],xis)

    # iid directions on the sphere for slicing
    xis=rand(normal_distr,P,d)
    xis=xis./sqrt.(sum(xis.^2,dims=2))
    push!(xis_dict["MC"],xis)
end

for P in Ps_sobol
    # Sobol RFF by putting the Sobol sequence into the inverse CDF of the Gaussian for each direction
    xis_01=reduce(hcat, next!(soboleng) for i =1:P)'
    xis_base=sqrt(2)*erfinv.(2*xis_01.-1)
    push!(xis_dict["Sobol-RFF"],xis_base)
    
    
    # project Sobol Features on the sphere for Sobol slicing
    xis_base=deepcopy(xis_base)
    zero_inds=dropdims(sum(xis_base.^2,dims=2),dims=2).==0
    xis_base[zero_inds,:]=rand(normal_distr,sum(zero_inds),d) # treat zero correctly
    xis=xis_base./sqrt.(sum(xis_base.^2,dims=2))
    push!(xis_dict["Sobol"],xis)
end

if d==3
    for P in Ps_sd
        # load spherical design direction. The sources of that are taken from 
        # https://www-user.tu-chemnitz.de/~potts/workgroup/graef/quadrature/index.php.en
        fid = h5open("distance_MMD_projs/d3/spherical_designs/P"*string(P)*".h5","r")
        xis_sd=reshape(collect(fid["xis"]),d,P)'
        push!(xis_dict["SD"],xis_sd)
    end
end


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
    labels=["MMD", "Sobol", "orth", "SD", "MC"]
    # compute errors
    for label in labels
        if size(xis_dict[label],1)==0
            continue
        end
        errors_dict[label]=[compute_slicing_error(x,gt,xis,basis_f) for xis in xis_dict[label]]
    end    
    for label in RFF_labels
        if size(xis_dict[label],1)==0
            continue
        end
        errors_dict[label]=[compute_RFF_error(x,gt,scale,xis) for xis in xis_dict[label]]
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
