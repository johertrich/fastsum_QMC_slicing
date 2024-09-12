using NFFT

function naive_kernel_sum(x,x_weights,y,kernel)
    # Naive kernel summation via the kernel matrix
    d=size(x,2)
    N=size(x,1)
    M=size(y,1)
    naive_sum=zeros(M)
    # batching for fitting memory constraints
    for i in 1:N
        for j in 1:M
            naive_sum[j]+=x_weights[i]*kernel(x[i,:],y[j,:])
        end
    end
    return naive_sum
end


function fastsum_fft(x,y,x_weights,scale,n_ft,x_range,fourier_fun,xis)
    # slicing with fft
    P=size(xis,1)
    d=size(x,2)    
    if size(x,2)==1
        P=1
    end
    
    # rescaling procedure
    x_norm=dropdims(sqrt.(sum(x.^2,dims=2)),dims=2)
    y_norm=dropdims(sqrt.(sum(y.^2,dims=2)),dims=2)
    max_norm=maximum([x_norm y_norm])
    
    scale_factor=Float32(0.25)*x_range/max_norm
    scale_max=0.1
    if scale*scale_factor>scale_max
        scale_factor=scale_max/scale
    end
    scale_real=scale*scale_factor
    x=x.*scale_factor
    y=y.*scale_factor
    
    # compute kernel on regular grid via Fourier trafo on R (use Poissons sampling theorem)
    h=collect(floor(.5*(-n_ft+1)):1:floor(.5*(n_ft-1)))
    kernel_ft=fourier_fun(h,scale_real)
    plan_x=plan_nfft(x[:,1],size(kernel_ft,1),reltol=1e-9)
    plan_y=plan_nfft(y[:,1],size(kernel_ft,1),reltol=1e-9)
    out=zeros(size(y,1))
    
    for p in 1:P
        xi = xis[p,:]
        x_proj=x*xi
        y_proj=y*xi
        nodes!(plan_x,reshape(-x_proj,1,:))
        nodes!(plan_y,reshape(-y_proj,1,:))
        a=adjoint(plan_x)*x_weights
        d=kernel_ft.*a
        out+=real.(plan_y*d)
    end
    return out./P
end

function RFF(x,y,x_weights,scale,xis)
    # Evaluate RFF for feature set xis
    P=size(xis,1)
    out=zeros(size(y,1))
    for p in 1:P
        xi = xis[p,:]
        x_proj=x*xi./scale
        y_proj=y*xi./scale
        cos_x=cos.(x_proj)
        cos_y=cos.(y_proj)
        cos_x_sum=sum(cos_x.*x_weights)
        sin_x=sin.(x_proj)
        sin_y=sin.(y_proj)
        sin_x_sum=sum(sin_x.*x_weights)
        res=cos_y.*cos_x_sum+sin_y.*sin_x_sum
        out=out+res
    end
    return out./P
end


function slicing_RFF(x,y,x_weights,scale,xis,one_d_xis)
    # slice with xis and evaluate 1D kernel sums via RFF
    P=size(xis,1)
    out=zeros(size(y,1))
    for p in 1:P
        xi = xis[p,:]
        x_proj=x*xi/scale
        y_proj=y*xi/scale
        x_proj=reshape(x_proj,:,1) .* reshape(one_d_xis[p,:],1,:)
        y_proj=reshape(y_proj,:,1) .* reshape(one_d_xis[p,:],1,:)
        cos_x=cos.(x_proj)
        cos_y=cos.(y_proj)
        cos_x_sum=sum(cos_x.*reshape(x_weights,:,1),dims=1)
        sin_x=sin.(x_proj)
        sin_y=sin.(y_proj)
        sin_x_sum=sum(sin_x.*reshape(x_weights,:,1),dims=1)
        res=cos_y.*cos_x_sum+sin_y.*sin_x_sum
        res=sum(res,dims=2)/size(one_d_xis,2)
        out=out+dropdims(res,dims=2)
    end
    return out./P
end

function fastsum_energy_1d(x,x_weights,y)
    # Sorting algorithm for fast sumation with negative distance (energy) kernel
    N=size(x,1)
    M=size(y,1)
    # Potential Energy
    yx=[y; x]
    inds_yx=sortperm(yx,rev=false)
    sorted_yx=yx[inds_yx]
    yx_weights=[zeros(M);x_weights]
    weights_sorted=yx_weights[inds_yx]
    pot0=sum(weights_sorted.*(sorted_yx.-sorted_yx[1]))
    yx_diffs=sorted_yx[2:M+N]-sorted_yx[1:M+N-1]
    # Mults from cumsums shifted by 1
    mults_short=sum(x_weights).-2*cumsum(weights_sorted)[1:M+N-1]
    mults=zeros(M+N)
    mults[2:N+M]=mults_short
    
    potential=zeros(M+N)
    potential[2:M+N]=yx_diffs
    potential=pot0.-cumsum(potential.*mults)
    out1=zeros(M+N)
    out1[inds_yx]=potential
    out1=out1[1:M]
    return out1
end

function fastsum_energy(x,y,x_weights,sliced_factor,xis)
    # slicing for energy kernel
    d=size(x,2)
    if d==1
        P=1
    end
    P=size(xis,1)
    out=zeros(size(y,1))
    for p in 1:P
    
        xi = xis[p,:]
        x_proj=x*xi
        y_proj=y*xi
        fastsum_energy=fastsum_energy_1d(x_proj,x_weights,y_proj)
        out=out+fastsum_energy
    end
    return -sliced_factor/P * out
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
