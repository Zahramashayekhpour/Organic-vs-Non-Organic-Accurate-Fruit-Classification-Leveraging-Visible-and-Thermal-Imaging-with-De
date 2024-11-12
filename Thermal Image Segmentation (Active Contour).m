 clc,clear,close all;
%images{k} = imread(fullfile('D:\nit\BNUT\MSc thesis\dataset\new dataset - changed name\organic\Mushroom\thermal\part1', file(k).name));
I1=imread('D:\nit\BNUT\MSc thesis\dataset\new dataset - changed name\organic\Mushroom\thermal\part1\102.jpg');
I2=im2gray(I);
figure,subplot(1,5,1),imshow(I),title('Original Image');
mask = zeros(size(I));
mask(100:end-200,100:end-100) = 1;
subplot(1,5,3),imshow(mask),title('Initial Contour Location');
bw = activecontour(I,mask);
subplot(1,5,4),imshow(bw),title('Segmented Image, 100 Iterations');
bw = activecontour(I,mask,700);
subplot(1,5,5),imshow(bw)
title('Segmented Image, 250 Iterations');
segmented_image = I.*uint8(bw);
subplot(1,5,5),imshow(segmented_image);
%imwrite(segmented_image,'102.jpg')

