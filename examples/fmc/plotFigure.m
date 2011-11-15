data = xlsread('scores-model-M15.xls');
data(1,:) = [];
data(:,2) = [];

class1 = data(:,2) == 1;
class2 = data(:,2) > 1;     % both class 2 and 3 are merged

hF = figure;
hA = axes;
set(hA,'FontWeight','bold','FontSize',14)
p1=plot(data(class1,3), data(class1,4),'kp','MarkerSize',2,'LineWidth',5);       % black squares
hold on
p2=plot(data(class2,3), data(class2,4),'ro','MarkerSize',2,'LineWidth',5);       % red circles
grid on
xlim([-80 80])
ylim([-80 80])
axis square

T2_limit = 8.60497;%6.42855;
s_i = 23.4108;
s_j = 25.6924;
nEllipsePoints = 300;
constant_X = sqrt(T2_limit) * s_i;
constant_Y = sqrt(T2_limit) * s_j;
delta      = (2*pi-0)/(nEllipsePoints);
ellipsePoints = zeros(nEllipsePoints,2);
for j=1:nEllipsePoints
    ellipsePoints(j,:) = [constant_X * cos(j*delta) constant_Y * sin(j*delta)];
end
ellipsePoints(end+1,:) = ellipsePoints(1,:);
plot(ellipsePoints(:,1),ellipsePoints(:,2),'k');

point1 = [-20  -40];
point2 = [2.60476	-1.44764];
lineSlope = (point2(2) - point1(2)) / (point2(1) - point1(1));
lineIntercept = point1(2) - lineSlope*point1(1);
xLimits = xlim;
plot(xLimits,xLimits.*lineSlope + lineIntercept,'Color',[0.47 0.06 0.9],'LineWidth',3)

offsetx = 2.5;
offsety = 0;
for k = 1:numel(class1)
    if class1(k)
        text(data(k,3)+offsetx,data(k,4)+offsety,num2str(data(k,1)),'Color',[0 0 0],'FontWeight','bold','FontSize',10)
    end
    if class2(k)
        text(data(k,3)+offsetx,data(k,4)+offsety,num2str(data(k,1)),'Color',[1 0 0],'FontWeight','bold','FontSize',10)
    end
end

xlabel('t_1','FontWeight','bold','FontSize',14)
ylabel('t_2','FontWeight','bold','FontSize',14)

