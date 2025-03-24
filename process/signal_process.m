function [RT,VT,AT_FW,AT_FY] = signal_process(path)
    % adcData = bin2mat('raw_data\zuoyi\adc_data_Raw_60.bin');
    % adcData = bin2mat('up\adc_data_Raw_1.bin');
    
    adcData = bin2mat(path);
    
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
    % try
    %     imwrite(RT,strcat('\predict_data','\rt','\1.jpg'));
    % catch exception
    %     disp(['Error occurred: ', exception.message]);
    % end
    % imwrite(RT,strcat('\predict_data','\rt','\1.jpg'));
    % imwrite(VT,strcat('\predict_data','\dt','\1.jpg'));
    % imwrite(AT_FW,strcat('\predict_data','\at_azimuth','\1.jpg'));
    % imwrite(AT_FY,strcat('\predict_data','\at_elevation','\1.jpg'));
end