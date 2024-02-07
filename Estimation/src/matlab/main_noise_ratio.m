filename_V = "../../data/data.csv";
V = csvread(filename_V);

n = size(V, 1);
po = size(V, 2);

bootn = 0;
B_pred = rica_bootstrap(V, q, po, n, n, bootn);
prediction = B_pred(po, q-1);

save("../../data/out.txt", "prediction", "-ascii")
disp("Rica finished successfully");