function [tr_norm] = znorm(tr)

tr_n = size(tr,1);
tr_mean = mean(tr,1);
tr_std = std(tr,1);
tr_norm = (tr-repmat(tr_mean,tr_n,1))./repmat(tr_std,tr_n,1);