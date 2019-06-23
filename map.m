function [I2Tseen,T2Iseen,I2Tunseen,T2Iunseen]= map_zero(qIfile,qTfile,dIfile,dTfile,dataList,queryList,unseenPer)

qIFea = importdata(qIfile);
qTFea = importdata(qTfile);
dIFea = importdata(dIfile);
dTFea = importdata(dTfile);

Lab_tr = importdata(dataList);
Lab_te = importdata(queryList); 
Lab_tr = Lab_tr.data;
Lab_te = Lab_te.data;
Lab_tr_tmp = Lab_tr-1;
Lab_te_tmp = Lab_te-1;

if unseenPer == 0.5
seenDataIndex = find(rem(Lab_tr_tmp,2) ~= 0);
unseenDataIndex = find(rem(Lab_tr_tmp,2) == 0);
seenQueryIndex = find(rem(Lab_te_tmp,2) ~= 0);
unseenQueryIndex = find(rem(Lab_te_tmp,2) == 0);
elseif unseenPer == 0.1
seenDataIndex = find(rem(Lab_tr_tmp,10) ~= 0);
unseenDataIndex = find(rem(Lab_tr_tmp,10) == 0);
seenQueryIndex = find(rem(Lab_te_tmp,10) ~= 0);
unseenQueryIndex = find(rem(Lab_te_tmp,10) == 0);
elseif unseenPer == 0.3
seenDataIndex = find(rem(Lab_tr_tmp,3) ~= 0);
unseenDataIndex = find(rem(Lab_tr_tmp,3) == 0);
seenQueryIndex = find(rem(Lab_te_tmp,3) ~= 0);
unseenQueryIndex = find(rem(Lab_te_tmp,3) == 0);
elseif unseenPer == 0.7
seenDataIndex = find(rem(Lab_tr_tmp,3) == 0);
unseenDataIndex = find(rem(Lab_tr_tmp,3) ~= 0);
seenQueryIndex = find(rem(Lab_te_tmp,3) == 0);
unseenQueryIndex = find(rem(Lab_te_tmp,3) ~= 0);
elseif unseenPer == 0.9
seenDataIndex = find(rem(Lab_tr_tmp,10) == 0);
unseenDataIndex = find(rem(Lab_tr_tmp,10) ~= 0);
seenQueryIndex = find(rem(Lab_te_tmp,10) == 0);
unseenQueryIndex = find(rem(Lab_te_tmp,10) ~= 0);
end

Lab_data_seen = Lab_tr(seenDataIndex,:);
Lab_query_seen = Lab_te(seenQueryIndex,:);
Lab_data_unseen = Lab_tr(unseenDataIndex,:);
Lab_query_unseen = Lab_te(unseenQueryIndex,:);

I_query_seen_re = qIFea(seenQueryIndex,:);
T_query_seen_re = qTFea(seenQueryIndex,:);
I_data_seen_re = dIFea(seenDataIndex,:);
T_data_seen_re = dTFea(seenDataIndex,:);

I_query_unseen_re = qIFea(unseenQueryIndex,:);
T_query_unseen_re = qTFea(unseenQueryIndex,:);
I_data_unseen_re = dIFea(unseenDataIndex,:);
T_data_unseen_re = dTFea(unseenDataIndex,:);

len_data_seen = size(Lab_data_seen,1);
len_query_seen = size(Lab_query_seen,1);
len_data_unseen = size(Lab_data_unseen,1);
len_query_unseen = size(Lab_query_unseen,1);

I_query_seen_re = znorm(I_query_seen_re);
T_query_seen_re = znorm(T_query_seen_re);
I_data_seen_re = znorm(I_data_seen_re);
T_data_seen_re = znorm(T_data_seen_re);
I_query_unseen_re = znorm(I_query_unseen_re);
T_query_unseen_re = znorm(T_query_unseen_re);
I_data_unseen_re = znorm(I_data_unseen_re);
T_data_unseen_re = znorm(T_data_unseen_re);


D = pdist([I_query_seen_re; T_data_seen_re],'cosine');
Z = squareform(D);
W = 1 - Z;
W_red = W(1:len_query_seen, len_query_seen+1:end);
[I2Tseen,~] = QryonTestBi(W_red, Lab_query_seen, Lab_data_seen);

D = pdist([T_query_seen_re; I_data_seen_re],'cosine');
Z = squareform(D);
W = 1 - Z;
W_red = W(1:len_query_seen, len_query_seen+1:end);
[T2Iseen,~] = QryonTestBi(W_red, Lab_query_seen, Lab_data_seen);

D = pdist([I_query_unseen_re; T_data_unseen_re],'cosine');
Z = squareform(D);
W = 1 - Z;
W_red = W(1:len_query_unseen, len_query_unseen+1:end);
[I2Tunseen,~] = QryonTestBi(W_red, Lab_query_unseen, Lab_data_unseen);

D = pdist([T_query_unseen_re; I_data_unseen_re],'cosine');
Z = squareform(D);
W = 1 - Z;
W_red = W(1:len_query_unseen, len_query_unseen+1:end);
[T2Iunseen,~] = QryonTestBi(W_red, Lab_query_unseen, Lab_data_unseen);
