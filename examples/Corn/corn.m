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
pls = lvm({'X', X, 'Y', Y}, 5);
plot(pls)