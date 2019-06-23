function [ mapI_TI, mapICategory] = QryonTestBi( W, testImgCat, testTxtCat)

query = W;
ImgQuery = query;
TxtQuery = query';

[Y,ImgQuery] = sort(ImgQuery,2,'descend');
[Y,TxtQuery] = sort(TxtQuery,2,'descend');


%% ----------evaluation-------------
% image query text
catImgNum = zeros(length(testImgCat),1);
for i = 1:length(testImgCat)
    catImgNum(i) = sum(testTxtCat==testImgCat(i));
end
% text query image
catTxtNum = zeros(length(testTxtCat),1);
for i = 1:length(testTxtCat)
    catTxtNum(i) = sum(testImgCat==testTxtCat(i));
end

[mapI_TI,mapICategory] = evaluateMAP_fast_general(ImgQuery,testImgCat,testTxtCat,catImgNum);
