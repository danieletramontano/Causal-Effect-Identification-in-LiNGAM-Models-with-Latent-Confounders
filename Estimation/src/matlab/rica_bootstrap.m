function [B_pred] = rica_bootstrap(V, q, po, n, train_size, bootn)
    [Bhbase] = estimate_model(V, q, po, n, n);
    Btot(:, :, 1) = Bhbase;
    if bootn>0
        for k=1:bootn
            [Bhnorm] = estimate_model(Vo, q, po, train_size, n);
            for i=1:q
                [~,temp]=min(sum((Bhbase-Bhnorm(:,i)).^2));
                I(i)=temp;
            end
        
            P=zeros(q,q);
            for i=1:q
                P(i,I(i))=1;
            end
            Bhnorm = Bhnorm * P; 
            Btot(:,:,k+1) = Bhnorm;
        end
    end
    B_pred =  mean(Btot, 3);
