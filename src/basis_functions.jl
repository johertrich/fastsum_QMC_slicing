using HypergeometricFunctions
using SpecialFunctions
using FFTW
using Plots


function Gauss(x,sigma)
    # x -> F(||x||) for basis function F of the Gauss kernel
    return dropdims(exp.(-.5/sigma^2*sum(x.^2,dims=2)),dims=2)
end

function Gauss_f(x,sigma,d)
    # x -> f(|x|) for one-dimensional basis function f for the Gauss kernel
    return map(t -> pFq((d/2, ), (1/2,),-0.5*t^2/sigma^2),x)
end

function Riesz(x,r=1)
    # x -> F(||x||) for basis function F of the Riesz kernel
    return -dropdims(sqrt.(sum(x.^2,dims=2)).^r,dims=2)
end

function Riesz_f(x,sliced_factor,r=1)
    # x -> f(|x|) for one-dimensional basis function f for the Riesz kernel
    return -sliced_factor*abs.(x).^r
end

function Laplace(x,alpha)
    # x -> F(||x||) for basis function F of the Laplace kernel
    return exp.(-alpha*sqrt.(sum(x.^2,dims=2)))
end

function thin_plate(x,scale)
    diff_sq= sum(x.^2,dims=2)
    out = diff_sq/scale^2 .*log.(sqrt.(diff_sq)/scale)
    out[diff_sq.==0].=0
    return out
end

function thin_plate_f(x,scale,C,d)
    out = d*(x/scale).^2 .*log.(x/scale)-C*(x/scale).^2
    out[x.==0].=0
    return out
end

function thin_plate_fun_ft(grid1d,d,scale,C)
    n_ft=size(grid1d,1)
    vect=thin_plate_f(abs.(grid1d/n_ft),scale,C,d)
    vect_perm=ifftshift(vect)
    kernel_ft=1/n_ft * fftshift(fft(vect_perm))
    return kernel_ft
end

function Gaussian_kernel_fun_ft(grid1d,d,sigma_sq)
    # Fourier trafo of the 1D counterpart of the Gauss kernel
    k=grid1d
    args=2 .* pi^2 .* sigma_sq .* k.^2
    log_args=log.(args)
    log_args[args.==0].=0
    factor=d*pi*sqrt(.5*sigma_sq)
    log_out=log_args*.5*(d-1).-args.-loggamma(.5*(d+2))
    out=exp.(log_out)
    if d>1
        out[args.==0].=0
    else
        out[args.==0].=1/gamma(.5*(d+2))
    end
    return out*factor
end


function Laplace_f(x,alpha,d)
    # x -> f(|x|) for one-dimensional basis function f for the Laplace kernel
    first_term=map(t -> pFq((d/2,), (1/2,1/2),alpha^2*t^2/4),x)
    second_term=map(t -> pFq(((d+1)/2,), (1,3/2),alpha^2*t^2/4),x)
    factor=exp(loggamma((d+1)/2)-loggamma(d/2))*sqrt(pi)*alpha*x
    return first_term-factor.*second_term
end

function Laplace_kernel_fun_ft(grid1d,d,alpha)
    # Laplace kernel is the Matern kernel with nu=0.5
    return Matern_kernel_fun_ft(grid1d,d,1/alpha,0.5)
end


function Matern(x,beta,p)
    # x -> F(||x||) for basis function F of the Matern kernel for smoothness nu=p+1/2
    x_norm = dropdims(sqrt.(sum(x.^2,dims=2)),dims=2)/beta
    summands=zeros(size(x_norm))
    for n=0:p
        add=factorial(p+n)/(factorial(n)*factorial(p-n))*(2*sqrt(2*p+1.)*x_norm).^(p-n)
        summands=summands+add
    end
    prefactor=factorial(p)/factorial(2*p)*exp.(-sqrt(2*p+1)*x_norm)
    return prefactor.*summands
end

function Matern_basis(x,beta,p)
    # Matern basis function (redundant!)
    x_norm = abs.(x)/beta
    summands=0
    for n=0:p
        add=factorial(p+n)/(factorial(n)*factorial(p-n))*(2*sqrt(2*p+1.)*x_norm).^(p-n)
        summands=summands+add
    end
    prefactor=exp.(-sqrt(2*p+1)*x_norm)*factorial(p)/factorial(2*p)
    return prefactor*summands
end


function Matern_f(x,beta,p,d)
    # x -> f(|x|) for one-dimensional basis function f for the Matern kernel for smoothness nu=p+1/2
    nu=p+1/2
    first_term=map(t -> pFq((d/2,), (1/2,1-nu),nu*t^2/(2*beta^2)),x)
    second_term=map(t -> pFq((nu+d/2,), (nu+1/2,nu+1),nu*t^2/(2*beta^2)),x)
    factor=(gamma(1-nu)*exp(loggamma(nu+d/2)-loggamma(d/2)-loggamma(2*nu+1))*(2*nu)^nu/beta^(2*nu)) * x.^(2*nu)
    return first_term-factor.*second_term
end

function Matern_kernel_fun_ft(grid1d,d,beta,nu)
    # Fourier trafo of the 1D counterpart of the Matern kernel
    k=grid1d
    args=k.^2
    log_args = log.(args)
    log_factor=loggamma(nu + .5*d)+d*log(pi)+d*log(beta)+.5*d*log(2.)-loggamma(.5*d)-loggamma(nu)-.5*d*log(nu)
    log_out=.5*(d-1)*log_args .- (nu+ .5*d) .* log.(1 .+(2*pi^2*beta^2 /nu)*args)
    out=exp.(log_out.+log_factor)
    if d>1
        out[args.==0].=0
    else
        out[args.==0].=exp(log_factor)
    end
    return out
end

function compute_sliced_factor(d)
    # Compute the slicing constant within the negative distance kernel
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0
        for j in 1:k
            fac=2*fac*j/(2*j-1)
        end
    else
        for j in 1:k
            fac=fac*(2*j+1)/(2*j)
        end
        fac=fac*pi/2
    end
    return fac
end

function compute_thin_plate_constant(d)
    if d%2==0
        mysum=0.
        for k in 1:d/2
            mysum+=1/k
        end
    else
        grid=0:.001:(1-0.001)
        integrands=(1 .- grid.^(d/2))./(1 .- grid)
        mysum=sum(integrands)/size(grid,1)
    end
    return -d/2 * (mysum-2+log(4))
end
