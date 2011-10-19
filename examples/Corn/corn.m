addpath('C:\kgd61600\Multiblock-code\class')

data = load('corn.mat');
%     m5spec: [80x700 dataset] Spectra on instrument m5   
%        mp5spec: [80x700 dataset] Spectra on instrument mp5  
%        mp6spec: [80x700 dataset] Spectra on instrument mp6  
%       propvals: [80x4   dataset] Property values for samples
%          m5nbs: [ 3x700 dataset] NBS glass stds on m5       
%         mp5nbs: [ 4x700 dataset] NBS glass stds on mp5      
%         mp6nbs: [ 4x700 dataset] NBS glass stds on mp6    
        
X = data.m5spec.data;
csvwrite('corn.csv', [X, data.propvals.data])

Y = block(data.propvals.data);
Y.add_labels(2, {'Moisture', 'Oil', 'Protein', 'Starch'})
plot(Y)

pcaY = lvm({'Quality', Y},2);
A = 5;
pls = lvm({'X', X, 'Y', Y}, A);
plot(pls)


R = pls.W{1}*inv(pls.P{1}'*pls.W{1}); 
C = pls.super.C;
beta = R*C';

figure
plot(data.m5spec.axisscale{2}, beta(:,1:4), 'linewidth', 2)
title(['Coefficients using A=', num2str(A)])
hold on
w = data.m5spec.axisscale{2};
plot([w(1) w(end)], [0, 0], 'k', 'linewidth', 3) 
legend({'Moisture', 'Oil', 'Protein', 'Starch'})
axis tight
grid