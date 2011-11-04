
% SCRIPT: To open the AVI file and store the individual frames in a 
% 4-way matrix.  Ways 1 and 2 and 3 are the images (R,G,B), the 4th way is each image in time sequence
% KGD: 17 July 2003
% KGD: updated on 3 November 2011

X = aviread('flame.avi');                                       % Load all the frames into a large structured array
nFrames = size(X,2);                                            % Get the number of frames in the AVI file

TestImage = getfield(X(1),'cdata');                             % Get the first image to determine the dimensions 
nRows     = size(TestImage,1);
nCols     = size(TestImage,2);
nLayers   = size(TestImage,3);
Frames    = uint8(zeros(nRows,nCols,nLayers,1));                % Initialize the storage array with just one frame

for k = 1:nFrames                                               % Store the individual frames in the storage array
    Frames(1:nRows,1:nCols,1:nLayers,k) = getfield(X(k),'cdata');
end;

clear X TestImage
pack

nPixels = nRows * nCols;
XTX = zeros(3,3);
minT = zeros(nFrames,3);
maxT = zeros(nFrames,3);
scaleT = zeros(nFrames,3);
for k = 1:nFrames                                               % Store the individual frames in the storage array
    k
    [a,b,c,d,e,f,g] = libmia('imagepca', Frames(:,:,:,k), 3);
    XTX = XTX + a / nPixels;
    minT(k,:) = e;
    maxT(k,:) = f;
    scaleT(k,:) = g;
end;

minT_avg = mean(minT);
maxT_avg = mean(maxT);
scaleT_avg = mean(scaleT);

[P, val] = eig(XTX);
[val, idx] = sort(diag(val), 'descend');
P = P(:,idx);
% For this data set only:
P(:,2) = -P(:,2);
temp = -maxT_avg(1);
maxT_avg(1) = -minT_avg(1);
minT_avg(1) = temp;

temp = -maxT_avg(2);
maxT_avg(2) = -minT_avg(2);
minT_avg(2) = temp;

scoreplots = uint8(zeros(256,256,nFrames));
for k = 1:nFrames                                               % Store the individual frames in the storage array
    k
    X_unfolded = Frames(:,:,:,k);
    X_unfolded = reshape(X_unfolded, nRows*nCols, 3);
    T = double(X_unfolded) * P;
    T = 255 *(T - repmat(minT_avg, nPixels, 1)) ./ repmat(maxT_avg-minT_avg, nPixels, 1);
    T = uint8(reshape(T, nRows, nCols, 3));
    scoreplots(:, :, k) = libmia('ScorePlot', T(:,:,1), T(:,:,2));
 end;
 
hF = figure;
hIm = subplot(1,2,1);
hSc = subplot(1,2,2);
XLim = [0 nCols] + 0.5;
YLim = [0 nRows] + 0.5;
XTick      = [];                                                          % The rest of the defaults
YTick      = [];
XTickLabel = [];
YTickLabel = [];
XGrid      = 'off';
YGrid      = 'off';
ScoreSpaceColourMap = [                                                 % The colour map used in the score space
    0.01041666666667 0 0; 0.79166666666667 0 0; 1 0.05208333333333 0; 1 0.31250000000000 0; 1 0.31250000000000 0; ...
    1 0.31250000000000 0; 1 0.31250000000000 0; 1 0.57291666666667 0; 1 0.57291666666667 0; 1 0.57291666666667 0; ...
    1 0.57291666666667 0; 1 0.57291666666667 0; 1 0.57291666666667 0; 1 0.57291666666667 0; 1 0.57291666666667 0; ...
    1 0.57291666666667 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; ...
    1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; ...
    1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; ...
    1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 0.83333333333333 0; 1 1 0.14062500000000; ...
    1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; ...
    1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; ...
    1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; ...
    1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; ...
    1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; 1 1 0.140625; ...
    1 1 0.140625; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 0.531250; 1 1 0.531250; 1 1 0.531250; 1 1 0.531250; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; ...
    1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1; 0 1 1];
set(hF, 'colormap', ScoreSpaceColourMap);
for k = 1:nFrames                                               % Store the individual frames in the storage array
    k
    image(XLim, YLim, Frames(:,:,:,k), 'Parent', hIm);
    set(hIm,'XLim',XLim,'YLim',YLim,'XTick',XTick,'YTick',YTick, ...
        'XTickLabel',XTickLabel,'YTickLabel',YTickLabel, ...
        'XGrid',XGrid,'YGrid',YGrid,'DataAspectRatio', [1 1 1],'Visible','off');
     
    image([0.5 255.5], [0.5 255.5], scoreplots(:,:,k), 'Parent', hSc);
    set(hSc,'XLim',XLim,'YLim',YLim,'XTick',XTick,'YTick',YTick, ...
        'XTickLabel',XTickLabel,'YTickLabel',YTickLabel, ...
        'XGrid',XGrid,'YGrid',YGrid,'DataAspectRatio', [1 1 1],'Visible','off');
    
    print('-dpng', '-r200', [sprintf('flame-%0.3d', k), '.png'] )
end
 



