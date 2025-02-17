function [predY] = sv_fcm(X,cls,anchor_num,dim)
%ACA_cmean, according to fuzzy cmean 
view_num= length(X);
iter = 0;
thrsh = 0.00001;
IsConverge = 0;
max_iter = 100;
A=cell(view_num,1);
U=cell(view_num,1);
V=cell(view_num,1);
d=zeros(view_num,1);
r=2;
m=anchor_num;
for v = 1 : view_num
   d(v)=size(X{v},1);
   A{v}=rand(d(v),m);
end

n=size(X{1},2);
for v = 1 : view_num
P{v}=zeros(m,n);
P{v}(:,1:m)=eye(m);
G{v} = P{v}.^r;
end

while (IsConverge == 0&&iter<max_iter+1)
    iter = iter + 1;
    
    for v = 1 : view_num
       XG{v} = X{v} * G{v}';
       A{v} = XG{v} ./ repmat(sum(G{v},2)',d(v),1);
    end
    %XG = temp * G';
    %aaa = sum(G,2)';
            
    for v = 1 : view_num
        dist{v} = L2_distance(X{v},A{v})+eps; 
        di = dist{v}.^(2/(1-r));
        P{v} = di./repmat(sum(di,2),1,m); 
        P{v} = P{v}';    
        G{v} = P{v}.^r;
    end
    obj(iter) = 0;
    for v = 1 : view_num
        obj(iter) = obj(iter) + sum(sum(dist{v}' .* G{v}));
    end
    
    
    if iter>2&&abs(obj(iter)-obj(iter-1))/obj(iter)<thrsh
        IsConverge = 1;
    end
end

F=0;
for v = 1 : view_num
    F=F+P{v}';
end

[~, predY] = max(F, [], 2);

[UU,~,~] = svd(P{1}','econ');
UU = UU ./ repmat(sqrt(sum(UU.^2, 2)), 1,size(UU,2));
Y=litekmeans(UU, cls, 'MaxIter', 100,'Replicates',1);

end

