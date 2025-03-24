clear
clc

%%
% type_Mat = ['AA','BB','BF','CC','DD','DU','FB','FF','FL','FR','LL','LR','RL','RR','UD','UU'];
% type_Mat = ['AA','BB','BF','CC'];
% type_Mat = {'back','front','left','right'};
type_Mat = {'up','down','clockwise','counterclockwise'};
input_mat = ['ATM','DTM','RTM','RDM'];
for pp = 1:length(type_Mat)
    % type = type_Mat(pp*2-1 : pp*2);
    type = type_Mat{pp};
    % path = ['D:\radar-data\adc_data_bin\',type ,'\'];
    path = ['D:\matlab\mydir\radar_gesture\processing\',type ,'\'];
    
    pathAT_FW = ['D:\matlab\mydir\radar_gesture\processing\',type ,'\at_azimuth\'];
    pathAT_FY = ['D:\matlab\mydir\radar_gesture\processing\',type ,'\at_elevation\'];
    pathVT = ['D:\matlab\mydir\radar_gesture\processing\',type ,'\dt\'];
    pathRT = ['D:\matlab\mydir\radar_gesture\processing\',type ,'\rt\'];
    % pathRDM = ['G:\2022\radar-data\output_imgs\',type ,'\RDM\'];
    mkdir(pathAT_FW);
    mkdir(pathAT_FY);
    mkdir(pathVT);
    mkdir(pathRT);
    % mkdir(pathRDM);
    
    list = dir([path,'*.bin']);
    k = length(list);
    nm = 1;
    
    for cir=1:50
        num = int2str(nm);
        fname = list(cir).name;
        adcData = bin2mat([path,fname]);
        
        [AT_FW, AT_FY] = AT_process(adcData);
        [RT, VT, R_axis, V_axis] = RD_process(adcData);
        RT = [RT; zeros(32-size(RT,1),32)];
        
        figure
        subplot(2,2,1)
        imagesc(1:32, R_axis, RT)
        xlabel('帧数');ylabel('距离/m');title('RT图');
        colormap(gray(64))
        subplot(2,2,2)
        imagesc(1:32, V_axis, VT)
        xlabel('帧数');ylabel('速度/m/s');title('VT图');
        colormap(gray(64))

        subplot(2,2,3)
        imagesc(AT_FW)
        xlabel('帧数');ylabel('方位角/度');title('水平方向AT图');
        colormap(gray(64))
        subplot(2,2,4)
        imagesc(AT_FY)
        xlabel('帧数');ylabel('俯仰角/度');title('竖直方向AT图');
        colormap(gray(64))
        
        % imwrite(RT,strcat(pathRT,type,'_',num,'RT.jpg'));
        % imwrite(VT,strcat(pathVT,type,'_',num,'VT.jpg'));
        % imwrite(AT_FW,strcat(pathAT_FW,type,'_',num,'AT_FW.jpg'));
        % imwrite(AT_FY,strcat(pathAT_FY,type,'_',num,'AT_FY.jpg'));

        imwrite(RT,strcat(pathRT,num,'.jpg'));
        imwrite(VT,strcat(pathVT,num,'.jpg'));
        imwrite(AT_FW,strcat(pathAT_FW,num,'.jpg'));
        imwrite(AT_FY,strcat(pathAT_FY,num,'.jpg'));

        fprintf(type);
        fprintf(' %.2f %%\n',cir/k*100);
        nm = nm+1;
    end
end