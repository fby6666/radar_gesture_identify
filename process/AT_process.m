function [AT_FW,AT_FY] = AT_process(adcData)

c = 3e8;
f0 = 77e9;
lambda = c / f0;

num_ADCSamples = 256;
num_chirps = 64;
num_frame  = 32;
num_RX = 8;

data = zeros(num_RX, num_ADCSamples, num_chirps, num_frame);
for nn = 1 : num_RX
    index = 0;
    for ii = 1 : num_frame
        for jj =1 : num_chirps
            data(nn,:,jj,ii) = adcData(nn, (index * num_ADCSamples + 1):(index + 1) * num_ADCSamples);
            index = index+1;
        end
    end
end

interval = 3;
num_MTI  = num_chirps - interval; %双脉冲对消间隔2
data_MTI = zeros(num_RX, num_ADCSamples, num_MTI, num_frame);
for nn = 1 : num_RX
    for ii = 1 : num_frame
        for jj = 1 : num_MTI
            data_MTI(nn,:,jj,ii) = data(nn,:,jj,ii) - data(nn,:,jj+interval,ii);
%             data_MTI(nn,:,jj,ii) = data(nn,:,jj + 2,ii) - 2 * data(nn,:,jj + 1,ii) + data(nn,:,jj,ii);
        end
    end
end

d_base = 0.5 * lambda;  %基线长度

%% 方位角
d = 0 : d_base : 2*d_base;
space_num = 101;
angle = linspace(-50, 50, space_num);  %用于存放幅度-角度曲线横轴
Pmusic1 = zeros(1, space_num);          % 用于存放幅度-角度曲线
Pmusic2 = zeros(1, space_num);          % 用于存放幅度-角度曲线
Pmusic_mn = zeros(space_num, num_frame * num_MTI); % 用于存放AT图

index = 0;%脉冲计数
for ii = 1:num_frame %遍历32帧，分别求取每帧中间一个PRT的AT图
    for jj = 1 : num_MTI
        Rxx = data_MTI(3:2:7, :, jj, ii) * data_MTI(3:2:7, :, jj, ii)' / num_ADCSamples;
        % 特征值分解
        [EV,D] = eig(Rxx);                  % 特征值分解
        EVA = diag(D)';                     % 将特征值矩阵对角线提取并转为一行
        [~,I] = sort(EVA);                % 将特征值排序 从小到大
        EV = fliplr(EV(:,I));               % 对应特征矢量排序
        % 遍历每个角度，计算空间谱
        for iang = 1 : space_num
            phim = deg2rad( angle(iang) );
            a = exp(-1i*2*pi*d/lambda*sin(phim)).';
            En = EV(:, 2 : end);                   % 取矩阵的第M+1到N列组成噪声子空间
            Pmusic1(iang) = 1 / (a' * (En * En') * a);
        end
        
        Rxx = data_MTI(4:2:8, :, jj, ii) * data_MTI(4:2:8, :, jj, ii)' / num_ADCSamples;
        % 特征值分解
        [EV,D] = eig(Rxx);                  % 特征值分解
        EVA = diag(D)';                     % 将特征值矩阵对角线提取并转为一行
        [~,I] = sort(EVA);                % 将特征值排序 从小到大
        EV = fliplr(EV(:,I));               % 对应特征矢量排序
        % 遍历每个角度，计算空间谱
        for iang = 1 : space_num
            phim = deg2rad( angle(iang) );
            a = exp(-1i*2*pi*d/lambda*sin(phim)).';
            En = EV(:, 2 : end);                   % 取矩阵的第M+1到N列组成噪声子空间
            Pmusic2(iang) = 1 / (a' * (En * En') * a);
        end
        
        index = index + 1;
        Pmusic_abs = abs(Pmusic1 + Pmusic2);        
        Pmmax = max(Pmusic_abs);
        Pmusic_mn(:, index) = ((Pmusic_abs / Pmmax));  % 归一化处理
    end
end

AT_FW = (imresize(Pmusic_mn, [32 32]));

% AT_FW = zeros(space_num,32);
% for k = 1 : 32
%     AT_FW(:,k) = Pmusic_mn(:, 1 + (k - 1) * 63);
% end

%% 俯仰角
d = 0 : d_base : 3*d_base;
space_num = 91;
angle = linspace(-45, 45, space_num);%用于存放幅度-角度曲线横轴
Pmusic = zeros(1, space_num);         % 用于存放幅度-角度曲线
Pmusic_mn = zeros(space_num, num_frame * num_MTI); % 用于存放AT图

index = 0;%脉冲计数
for ii = 1 : num_frame %遍历32帧，分别求取每帧中间一个PRT的AT图
    for jj = 1 : num_MTI
        Rxx = data_MTI(1 : 4, :, jj, ii) * data_MTI(1 : 4, :, jj, ii)' / num_ADCSamples;
        % 特征值分解
        [EV,D] = eig(Rxx);                  % 特征值分解
        EVA = diag(D)';                     % 将特征值矩阵对角线提取并转为一行
        [EVA,I] = sort(EVA);                % 将特征值排序 从小到大
        EV = fliplr(EV(:,I));               % 对应特征矢量排序
        % 遍历每个角度，计算空间谱
        for iang = 1 : space_num
            phim = deg2rad( angle(iang) );
            a = exp(-1i * 2 * pi * d / lambda * sin(phim)).';
            En = EV(:, 2 : num_RX - 4);                   % 取矩阵的第M+1到N列组成噪声子空间
            Pmusic(iang) = 1 / (a' * (En * En') * a);
        end
        index = index + 1;
        Pmusic_abs = abs(Pmusic);
        Pmmax = max(Pmusic_abs);
        Pmusic_mn(:, index) = Pmusic_abs / Pmmax;            % 归一化处理
    end
end

AT_FY = (imresize(Pmusic_mn, [32 32]));

% AT_FY = zeros(space_num,32);
% for k = 1 : 32
%     AT_FY(:, k) = Pmusic_mn(:, 1 + (k - 1) * 63);
% end

end