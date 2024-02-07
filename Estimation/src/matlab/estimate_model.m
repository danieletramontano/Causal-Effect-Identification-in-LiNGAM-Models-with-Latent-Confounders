function [Bhnorm]=estimate_model(V, q, po, train_size, n)

%overcomplete ICA
C = cov(V);
[U,S,Ut] = svd(C);
Vo = V*U*inv(S.^(0.5)); %prewhitening
if train_size == n
    train_index = randsrc(1,train_size, [1:n;ones(1,n)/n]);
else
    train_index = randsample(n,train_size);
end
Mdl = rica(Vo(train_index,:),q,'NonGaussianityIndicator', 1*ones(q,1));%
W = Mdl.TransformWeights;

R = (U*inv(S.^0.5)*W);
Bhat = (R'*(R*R')^-1)';

[~, ids] = max(abs(Bhat));
B_max = zeros(1, q);
for i=1:q
    B_max(1, i) = Bhat(ids(i), i);
end

Bhnorm = Bhat./ (ones(po,1)*B_max); 
P = zeros(q, q);

for i=1:q
    P(i, i) = 1;
end

[~, id] = max(abs(Bhnorm(po-1,:)));
P(q-1, q-1) = 0;
P(id, id) = 0;
P(id, q-1) = 1;
P(q-1, id) = 1;
Bhnorm = Bhnorm * P;