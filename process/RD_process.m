function [RT, VT, R_axis, V_axis] = RD_process(adcData)

num_ADCSamples = 256;
num_chirps = 64;
num_frame  = 32;

adcData_T0 = adcData(1 : 1 : 8, :);
% adcData_T0 = adcData(1 : 2 : 7, :);
% adcData_T1 = adcData(2 : 2 : 8, :);

data = zeros(1, num_chirps * num_ADCSamples * num_frame);
for k = 1 : 4
    data = data + adcData_T0(k, :);
end

data_frame = zeros(num_chirps, num_ADCSamples, num_frame);
for k = 1 : num_frame
    for m = 1 : num_chirps
        data_frame(m,:,k) = data(num_chirps*num_ADCSamples*(k-1) + (m-1)*num_ADCSamples + 1 : num_chirps*num_ADCSamples*(k-1) + m*num_ADCSamples);
    end
end

%% MTI
num_mti = num_chirps - 2;
data_mti = zeros(num_mti, num_ADCSamples, num_frame);
for k = 1 : num_frame
    for m = 1 : num_mti
        data_mti(m, :, k) = data_frame(m, :, k) - data_frame(m+2, :, k);
    end
end

%% 二维FFT
data_fft = zeros(num_mti, num_ADCSamples, num_frame);
for k = 1 : num_frame
    for m = 1 : num_mti
        data_fft(:, :, k) = fftshift(fft2(data_mti(:, :, k)));
    end
end

%% CFAR
num_R = 18;
num_V = 32;
TH_cfar = zeros(num_mti, num_ADCSamples, num_frame);
data_RD = zeros(num_V, num_R, num_frame);
RT = zeros(num_R, num_frame);
VT = zeros(num_V, num_frame);

for k = 1 : num_frame
    for m = 1 : num_mti
        TH_cfar(m, :, k) = cfar(abs(data_fft(m, :, k)));
    end
    data_cfar = (TH_cfar < abs(data_fft)) .* abs(data_fft);
    
    data_RD(:, :, k) = data_cfar(32-num_V/2+1 : 32+num_V/2, 128 : 128+num_R-1, k);
    [x0, y0] = find(data_RD(:, :, k) == max(max(data_RD(:, :, k))));
    
    tmp = data_RD(x0(1), :, k).';
    tmp = tmp / (max(tmp) + eps);
    RT(:, k) = tmp;
    
    tmp = data_RD(:, y0(1), k).';
    tmp = tmp / (max(tmp) + eps);
    VT(:, k) = tmp;
    
    c   = 3e8;
    f0  = 77e9;
    fs  = 10e6;
    Kr  = 105e12;
    PRT = 138e-6;
    lambda = c / f0;
    
    r_axis = linspace(-fs / 2, fs / 2, num_ADCSamples) / Kr * c / 2;
    v_axis = linspace(-1 / (2 * PRT), 1 / (2 * PRT), num_chirps) * lambda / 2;
    
    R_axis = r_axis(128 : 128+32-1);
    V_axis = v_axis(32-num_V/2+1 : 32+num_V/2);
    
%     imagesc(r_axis(128 : 128+num_R-1), v_axis(32-num_V/2+1 : 32+num_V/2), abs(data_RD(:, :, k)));
%     pause(0.2)
end

end