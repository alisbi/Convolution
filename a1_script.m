fprintf("Question 1 is starting ...\n");
fprintf("Part 1.1: \n");

fprintf("0  5  0  0  0  0  0  0  0  0\n");
fprintf("0  0  0  0  0  0  0  0  0  0\n");
fprintf("0 -7  2  8 -1 -2 -1 -3  0  0\n");
fprintf("0  0  1  1  0  0 -1 -1  0  0\n");
fprintf("0  0  3  1 -2  4 -1 -5  0  0\n");
fprintf("0  0 -1 -1  0  0  1  1  0  0\n");
fprintf("0  0  1  2  2  2 -3 -4  0  0\n");
fprintf("0  0  0  0  0  0  0  0  0  0\n");
fprintf("0  0  0  0  0  0  0  0  0  0\n");
pause;

fprintf("Part 1.2: \n");
fprintf("The gradient magnitude at pixel [2,3] is sqrt(65) = %d\n", sqrt(1^2 + 8^2));
fprintf("The gradient magnitude at pixel [4,3] is sqrt(5) = %d\n", sqrt(2^2 + 1^2));
fprintf("The gradient magnitude at pixel [4,6] is sqrt(5) = %d\n", sqrt(2^2 + 1^2));
pause;

fprintf("Part 1.3: \n");
I = imread('house.tiff');
fil = fspecial('gaussian', [13 13], 2);
ker = [1,0,-1];
myC = MyConv(I, fil);
myC = uint8(myC);
imshow(myC);
pause;

fprintf("Part 1.4: \n");
I_gray = rgb2gray(I);
fil = fspecial('gaussian', [13, 13], 2);
conv_filt = imfilter(I_gray, fil, 'conv');
imshow(conv_filt);
pause;
imshow(conv_filt - myC);
fprintf("The resulting image is black since there is no difference between the pictures.\n");
clear;
pause;

fprintf("Part 1.5: \n");
fprintf("Execution time for 2D: \n");
pic = imread('tiger.jpg');
pic_gray = rgb2gray(pic);
tic
ga = fspecial('gaussian', [5, 5], 8);
conv2d = imfilter(pic_gray, ga, 'conv');
imshow(conv2d);
toc
pause;

fprintf("Execution time for two 1D: \n");
tic
ga = fspecial('gaussian', [1, 5], 8);
conv1d_1st = imfilter(pic_gray, ga, 'conv');
conv1d_2nd = imfilter(conv1d_1st, ga', 'conv');
imshow(conv1d_2nd);
toc
fprintf("The execution time for 2 1D is less than 2D\n");
clear;
pause;

fprintf("Question 2 is starting ...\n");
fprintf("Part 2.1: \n");
img_f = imread('bowl-of-fruit.jpg');
img_h = imread('house.tiff');
img_f_canny = MyCanny(img_f);
img_h_canny = MyCanny(img_h);
imshow(img_f_canny);
pause;
imshow(img_h_canny);
pause;
%imshow(edge(rgb2gray(img_f), 'canny'));


fprintf("Part 2.2: \n");


fprintf("Question 3 is starting ...\n");
I2 = imread('ryerson.jpg');
%I2 = imread('house.tiff');
an = MySeamCarving(I2, 320, 720);
an = uint8(an);
%imwrite(an,'C:\Users\alisa\Desktop\School\winter2021\compvision\A1\house_500X500.tiff', 'tiff');
imshow(an);



function sub_img = MySeamCarving(img, row, col)
     I = img;
     sub_img = I;
     [r, c, co] = size(img);
     cond_r = r - row;
     cond_c = c - col;
   
     for u = 1 : cond_c
         R = sub_img(:,:,1);
         G = sub_img(:,:,2);
         B = sub_img(:,:,3);
         h = fspecial('sobel');
         R_im_dy = imfilter(double(R), h, 'conv');
         R_im_dx = imfilter(double(R), h', 'conv');
         R_mag = uint8(sqrt(R_im_dx.^2 + R_im_dy.^2));

         G_im_dy = imfilter(double(G), h, 'conv');
         G_im_dx = imfilter(double(G), h', 'conv');
         G_mag = uint8(sqrt(G_im_dx.^2 + G_im_dy.^2));

         B_im_dy = imfilter(double(B), h, 'conv');
         B_im_dx = imfilter(double(B), h', 'conv');
         B_mag = uint8(sqrt(B_im_dx.^2 + B_im_dy.^2));
         E = R_mag + G_mag + B_mag;


         [x, y] = size(E);
         M = zeros(x, y);
         M(1,:) = E(1,:);

         for i=2 : x-1
             for j=2 : y
                 M(i,j) = E(i,j) + min([M(i-1,j-1),M(i,j-1),M(i+1,j-1)]);
             end
         end
         sub_img = CarvingHelper(M,sub_img);
            
     end
     
     sub_img = permute(sub_img,[2 1 3]);
     
     
     for v = 1 : cond_r
         R = sub_img(:,:,1);
         G = sub_img(:,:,2);
         B = sub_img(:,:,3);
         h = fspecial('sobel');
         R_im_dy = imfilter(double(R), h, 'conv');
         R_im_dx = imfilter(double(R), h', 'conv');
         R_mag = uint8(sqrt(R_im_dx.^2 + R_im_dy.^2));

         G_im_dy = imfilter(double(G), h, 'conv');
         G_im_dx = imfilter(double(G), h', 'conv');
         G_mag = uint8(sqrt(G_im_dx.^2 + G_im_dy.^2));

         B_im_dy = imfilter(double(B), h, 'conv');
         B_im_dx = imfilter(double(B), h', 'conv');
         B_mag = uint8(sqrt(B_im_dx.^2 + B_im_dy.^2));
         E = R_mag + G_mag + B_mag;


         [x, y] = size(E);
         M = zeros(x, y);
         M(1,:) = E(1,:);

         for i=2 : x-1
             for j=2 : y
                 M(i,j) = E(i,j) + min([M(i-1,j-1),M(i,j-1),M(i+1,j-1)]);
             end
         end
         sub_img = CarvingHelper(M, sub_img);
         
     end
     
     sub_img = permute(sub_img,[2 1 3]);
     
end

function Res_img = CarvingHelper(A, im)
     [r,c] = size(A);
     [min_v, index] = min([A(r,:)]);
     cnt_pos = zeros(1,r);
     %cnt_pos(1,1) = min_v;
     Res_img = zeros(r,c-1,3);
     Res_img(r,1:index-1,:) = im(r,1:index-1,:);
     Res_img(r,index:c-1,:) = im(r,index+1:c,:);
     
     
     for i=r-1 : -1 : 2
         for j=2 : c-1
             [min_v, index] = min([A(i-1,j),A(i-1,j+1),A(i-1,j-1)]);
             Res_img(i,1:index-1,:) = im(i,1:index-1,:);
             Res_img(i,index:c-1,:) = im(i,index+1:c,:);
             
         end
     end
     
    
end



function can = MyCanny(img)
    img_f_gray = rgb2gray(img);
    [m, n] = size(img_f_gray);
    can = zeros(m, n);
    h = fspecial('sobel');
    im_dy = imfilter(double(img_f_gray), h, 'conv');
    im_dx = imfilter(double(img_f_gray), h', 'conv');
    grad_mag = sqrt(im_dx.^2 + im_dy.^2);
    grad_or = atan2(im_dy, im_dx);
    [r, c] = size(grad_mag);
    final = zeros(r, c);
    
    for i=1 : r
        for j=1 : c
            if ((grad_or(i, j) >= 0) && (grad_or(i, j) < 22.5) || (grad_or(i, j) >= 157.5) && (grad_or(i, j) < 202.5) || (grad_or(i, j) >= 337.5) && (grad_or(i, j) <= 360))
                grad_or(i, j) = 0;
            end
            if ((grad_or(i, j) >= 22.5) && (grad_or(i, j) < 67.5) || (grad_or(i, j) >= 202.5) && (grad_or(i, j) < 247.5))
                grad_or(i, j) = 45;
            end
            if ((grad_or(i, j) >= 67.5) && (grad_or(i, j) < 112.5) || (grad_or(i, j) >= 274.5) && (grad_or(i, j) < 292.5))
                grad_or(i, j) = 90;
            end
            if ((grad_or(i, j) >= 112.5) && (grad_or(i, j) < 157.5) || (grad_or(i, j) >= 292.5) && (grad_or(i, j) < 337.5))
                grad_or(i, j) = 135;
            end
        end
    end
    
    
    for i=2 : r-1
        for j=2 : c-1
            switch grad_or(i,j)
                case 0          
                    final(i,j) = (grad_mag(i,j) == max([grad_mag(i,j), grad_mag(i,j+1), grad_mag(i,j-1)]));
                case 45
                    final(i,j) = (grad_mag(i,j) == max([grad_mag(i,j), grad_mag(i+1,j-1), grad_mag(i-1,j+1)]));
                case 90
                    final(i,j) = (grad_mag(i,j) == max([grad_mag(i,j), grad_mag(i+1,j), grad_mag(i-1,j)]));
                case 135
                    final(i,j) = (grad_mag(i,j) == max([grad_mag(i,j), grad_mag(i+1,j+1), grad_mag(i-1,j-1)]));
            end
        end
    end
    final = final.*grad_mag;
    can = final > 38;
end


function con = MyConv(img, kernal)
    img_gray = rgb2gray(img);
    img_gray = double(img_gray);
    [r,c] = size(img_gray);
    [m,n] = size(kernal);
    img_gray_padded = padarray(img_gray,[m,n]);
    ker_flip = rot90(kernal, 2);
    [pr,pc] = size(img_gray_padded);
    con = zeros(r,c);
    for x = (1 + m):(pr - m)
        for y = (1 + n):(pc - n)  
            sub = img_gray_padded((x-floor(m/2)):(x+floor(m/2)),( y-floor(n/2)): (y+floor(n/2)));
            acc = 0;
            for i = 1 : m
                for j = 1 : n   
                    temp = sub(i, j) * ker_flip(i, j);
                    acc = acc + temp;
                    
                end
            end
            con(x,y) = acc;
        end
    end
    con = con(1+m:pr-m, 1+n:pc-n);
end




