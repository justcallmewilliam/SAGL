function [Y] = SAGL(X,cls,anchor_num,dim,lambda)%%%%||x-wa||^2_2*p+FLF
%OPACA_cmean, according to fuzzy cmean_ 
view_num= length(X);
iter = 0;
thrsh = 0.00001;
IsConverge = 0;
max_iter = 50;
XPtAt=cell(view_num,1);
W=cell(view_num,1);
WtX=cell(view_num,1);
WvA=cell(view_num,1);
distXA=cell(view_num,1);
U=cell(view_num,1);
V=cell(view_num,1);
d=zeros(view_num,1);
r=1;
alpha = ones(view_num,1)/view_num;


for v = 1 : view_num
   d(v)=size(X{v},1);
end
m=anchor_num;
n=size(X{1},2);
P=rand(m,n);
col_sum = sum(P);
P = P./col_sum(ones(m, 1), :);

A=rand(dim,m);
while (IsConverge == 0&&iter<max_iter+1)
    iter = iter + 1;
    AP = A * P;
    for v = 1 : view_num
       XPtAt{v}= alpha(v)^r*X{v} * AP';
       %XPtAt{v}= X{v} * AP';
                [U{v},~,V{v}] = svd(XPtAt{v},'econ');
       W{v} = U{v} * V{v}';
    end%%%%%%%%%%%%checked   
    
    for v = 1 : view_num
       WtX{v} = W{v}'*X{v};
    end
    temp = 0;
    for v = 1 : view_num
       temp = temp + alpha(v)^r*WtX{v};
    end
    
    WtXP = temp * P';
    A = WtXP ./ ((repmat(sum(P,2)',dim,1)+eps).*sum(alpha.^r));

    dist=0;
    for v = 1 : view_num
        WvA{v} = W{v}*A;
        distXA{v} = pdist2(X{v}',WvA{v}');       
        dist = dist + alpha(v)^r*distXA{v};
        alpha(v) = 0.5/sqrt(sum(sum( distXA{v}.*P'))); 
    end
    %alpha = updatealpha(P,distXA,r,view_num);
    [Y,~,Pt,~,~,term2] = my_gamma_coclustering_bipartite_fast1(-0.5.*dist, cls, lambda,50,0);
    Pt = full(Pt);
%         if n == 2000
%         if iter ==1
%            save('Z1.mat','Pt'); 
%         end
% 
%         if iter ==3
%            save('Z3.mat','Pt'); 
%         end
% 
%         if iter ==5
%            save('Z5.mat','Pt'); 
%         end
% 
%         if iter ==10
%            save('Z10.mat','Pt'); 
%         end
% 
%         if iter ==20
%            save('Z20.mat','Pt'); 
%         end
% 
%         if iter ==50
%            save('Z50.mat','Pt'); 
%         end
%     
%     end
    P = Pt';    
    
    obj(iter) = sum(sum(dist' .* P))+lambda*norm(P,'fro')^2+term2;
    %obj(iter) = sum(sum(dist' .* P));
    if iter>2&&abs(obj(iter-1)-obj(iter))/obj(iter-1)<thrsh
        IsConverge = 1;
    end
end

end

