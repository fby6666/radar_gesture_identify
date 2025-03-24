function [adcData] = bin2mat(file_name)

num_ADCSamples = 256;
num_RX         = 4;
% num_TX         = 2;

fid = fopen(file_name,'r');
adcData = fread(fid, 'int16');
fclose(fid);
filesize = size(adcData, 1);

num_chirps = filesize / (2 * num_ADCSamples * num_RX);
tmp = zeros(1, filesize / 2);

counter = 1;
for ii = 1:4:filesize-3
    tmp(1, counter)     = adcData(ii)   + 1j * adcData(ii+2);
    tmp(1, counter + 1) = adcData(ii+1) + 1j * adcData(ii+3);
    counter = counter + 2;
end

tmp = reshape(tmp, num_ADCSamples * num_RX, num_chirps);
tmp = tmp.';

adcData = zeros(num_RX, num_chirps * num_ADCSamples);
for row = 1 : num_RX
    for ii = 1 : num_chirps
        adcData(row, (ii-1)*num_ADCSamples+1 : ii*num_ADCSamples) = tmp(ii, (row-1)*num_ADCSamples+1 : row*num_ADCSamples);
    end
end

tmp = adcData.';
adcData = zeros(8, num_chirps / 2 * num_ADCSamples);
for ii = 1 : 4
    RxTx = reshape(tmp(:, ii), num_ADCSamples, num_chirps);
    RxT1 = RxTx(:, 1 : 2 : num_chirps - 1);
    RxT2 = RxTx(:, 2 : 2 : num_chirps);
    RxT1 = reshape(RxT1, 1, num_chirps / 2 * num_ADCSamples);
    RxT2 = reshape(RxT2, 1, num_chirps / 2 * num_ADCSamples);
    adcData(2*ii-1:2*ii, :) = [RxT1; RxT2];
end

end