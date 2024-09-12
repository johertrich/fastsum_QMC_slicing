using Random

function median_distance(x,y,subsample=nothing)
    # compute the median of the pairwise distances of x[i] and y[j]
    # to speed it up take only the number of samples defined by subsample
    # nothing for taking all samples
    
    if isnothing(subsample)
        x_short=x
        y_short=y
    else
        inds1=randperm(N)[1:subsample]
        inds2=randperm(N)[1:subsample]
    
        x_short=x[inds1,:]
        y_short=y[inds2,:]
    end

    dists=zeros(size(x_short,1)*size(y_short,1))
    for i=1:size(x_short,1)
        for j=1:size(y_short,1)
            dists[(i-1)*size(y_short,1)+j]=sqrt(sum((x_short[i,:]-y[j,:]).^2))  
        end
    end
    med=median(dists)
    return med
end
