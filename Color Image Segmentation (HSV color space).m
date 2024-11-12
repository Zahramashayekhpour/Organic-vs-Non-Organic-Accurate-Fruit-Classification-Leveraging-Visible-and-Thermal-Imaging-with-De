clc,clear,close all;

file = dir('D:\nit\BNUT\MSc thesis\dataset\new dataset - changed name\Non-organic\Apple\RGB\*.jpeg'); % Directory where images
NF = length(file); % Count of Images in the directory
writetable(struct2table(file), 'file.csv')
images = cell(NF,1); % creating a variable to hold martix of images
for k = 1 : NF  
images{k} = imread(fullfile('D:\nit\BNUT\MSc thesis\dataset\new dataset - changed name\Non-organic\Apple\RGB', file(k).name));
% Define thresholds for channel 1 based on histogram settings
I = rgb2hsv(images{k});
channel1Min = 0.000;
channel1Max = 1.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.114;
channel2Max = 1.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.000;
channel3Max = 1.000;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;
subplot(2,3,2),imshow(BW),title('Original Image');

se = strel('disk',40);
OPEN_IMG = imopen(BW,se);
subplot(2,3,3),imshow(OPEN_IMG);title('open image1');

se = strel('disk',65);
CLOSE_IMG = imclose(OPEN_IMG,se);
subplot(2,3,4),imshow(CLOSE_IMG);title('close image1');
segmented_image = images{k}.*uint8(CLOSE_IMG);
subplot(2,3,5),imshow(segmented_image);title('segmented_image');

imwrite(segmented_image,sprintf('%d.jpg',k))
end