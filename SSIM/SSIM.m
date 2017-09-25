clear
clc
close all


ref = imread('I:\Users\Andersen\Documents\Github\ParallelProgramming\SSIM\SSIM\cinque_terre_small.jpg');
H = fspecial('Gaussian',[11 11], 2);
A = imfilter(ref,H,'replicate');
ref = rgb2gray(ref);
A = rgb2gray(A);

realssim = ssim(A, ref);
ssim_index_val = ssim_index(ref, A);

img1 =double(ref);
img2 = double(A);

window = fspecial('gaussian', 11, 1.5);
K(1) = 0.01;	      
K(2) = 0.03;								      
L = 255;  

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');

mu1_test = zeros(11); 
for i = 1:11
    for j = 1:11
    
        mu1_test(i, j) =window(i, j)*img1(i+1,j+1);
        
    end
end

sum(sum(mu1_test));
0.000001057565598153262*255

mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
mssim = mean2(ssim_map);
