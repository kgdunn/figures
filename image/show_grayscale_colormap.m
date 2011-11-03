X=imread('Grayscale_8bits_palette_sample_image.png');
Y(:,:,1) = X;
Y(:,:,2) = X;
Y(:,:,3) = X;
[r,c] = size(X);


XLim = [0 c] + 0.5;
YLim = [0 r] + 0.5;
hG = subplot(1,3,1);
hC = subplot(1,3,2);
hB = subplot(1,3,3);

image(XLim, YLim, Y, 'Parent', hG);
XTick      = [];                                                          % The rest of the defaults
YTick      = [];
XTickLabel = [];
YTickLabel = [];
XGrid      = 'off';
YGrid      = 'off';
set(hG,'XLim',XLim,'YLim',YLim,'XTick',XTick,'YTick',YTick,'XTickLabel',XTickLabel,'YTickLabel',YTickLabel, ...
                'XGrid',XGrid,'YGrid',YGrid,'DataAspectRatio', [1 1 1],'Visible','off');
     


image(XLim, YLim, X, 'Parent', hC);

XTick      = [];                                                          % The rest of the defaults
YTick      = [];
XTickLabel = [];
YTickLabel = [];
XGrid      = 'off';
YGrid      = 'off';

              
set(hC,'XLim',XLim,'YLim',YLim,'XTick',XTick,'YTick',YTick,'XTickLabel',XTickLabel,'YTickLabel',YTickLabel, ...
                'XGrid',XGrid,'YGrid',YGrid,'DataAspectRatio', [1 1 1],'Visible','off');
     
colormap(jet(255))
axes(hB)
colorbar(hB)

