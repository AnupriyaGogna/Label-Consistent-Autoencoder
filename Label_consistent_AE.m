
function [we, wd,M]=Label_consistent_AE(x,c,node,max_iter,lambda)

% This code is for Label Consistent AE paper submitted to IEEE TIP

%Function to implement Label Consistent Autoencoder
%||x-wd*we*x||_F+lambda||x-M*we*x||_F

% x is input sample x
% c is label of input sample Q
% node = number of nodes in hidden layer h_n
% max_iter is maximum number of iterations
% lambda is regularization paramter
% we is encoder weight
% wd is decoder weight
% M is linear map

x=x';
C=c';

%Initilaize variables
W=mldivide(x,x);
L=mldivide(x,C);
[U,~,V]=svd(W);
wet=U(:,1:node); % wet = we';
wdt=V(1:node,:); % wdt = wd';
alpha=1.1*max(eig(x'*x));

for i=1:max_iter

   B1=W+(1/alpha)*x'*(x-x*W);
   B2=L+(1/alpha)*x'*(C-x*L);
   wdt=mldivide(wet,B1);
   Mt=mldivide(wet,B2);
    
   aoper=wdt*wdt'+lambda*(Mt*Mt');
   bvec=B1*wdt'+lambda*B2*Mt';
   wet=mrdivide(bvec,aoper);
  
   W=wet*wdt;
   L=wet*Mt;
 
end
we=wet';
wd=wdt';
M=Mt';


