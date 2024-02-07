filename_V = "../../data/data_stored.csv";
V = csvread(filename_V);

filename_samples_list = "../../data/sample_list.csv";
samples_list = csvread(filename_samples_list);

n = size(V, 1);
po = size(V, 2);

bootn = 0;
for i=1:length(samples_list)
    samples_n = samples_list(i);
    train_size = samples_n;
    V_sampled = V(1:samples_n, :);
    B_pred = rica_bootstrap(V_sampled, q, po, samples_n, train_size, bootn);
    predictions(i) = B_pred(po, q-1);
end
writematrix(predictions, "../../data/pred.txt");
disp("Rica finished successfully");