function signal_cfar = cfar(signal)

N = length(signal);
signal_cfar = zeros(1, N);
Pfa = 1e-10;
n_train = 10; %һ��Ĳο���Ԫ����
n_guard = 5; %һ��ı�����Ԫ�ĳ���

for i = 1 : N
    if i > n_train + n_guard && i < N - n_train - n_guard
        a = 2 * n_train * (Pfa^(-1 / (2 * n_train)) - 1); %��������
        nsum = sum(abs(signal(i - n_guard - n_train : i - n_guard - 1)).^2)+sum(abs(signal(i + n_guard + 1 : i + n_train + n_guard)).^2);
        nsum = nsum / (2 * n_train);
        nsum = sqrt(a * nsum);
    elseif i < n_train + n_guard + 1    %������Ԫ���ο���Ԫ�����ݲ���n_train��
        nsum = sum(abs(signal(i + n_guard + 1 : i + n_train + n_guard)).^2);
        n_tt = 0;
        a = n_train * (Pfa^(-1 / n_train) - 1); %��������
        if i > n_guard + 1   %����if��ʾ���Ե�Ԫ������ݸ�������n_guard����С��n_guard+n_train��
            nsum = nsum + sum(abs(signal(1 : i - n_guard - 1)));
            n_tt = i - n_guard - 1;
            a = (n_train + n_tt) * (Pfa^(-1 / (n_train+n_tt))-1);
        end
        nsum = nsum / (n_train + n_tt);
        nsum = sqrt(a * nsum);
    elseif i > N-n_train-n_guard - 1  %������Ԫ�Ҳ�ο���Ԫ�����ݲ���n_train��
        nsum = sum(abs(signal(i - n_guard - n_train : i - n_guard - 1)).^2);
        n_tt = 0;
        a = n_train*(Pfa^(-1/n_train)-1); %��������
        if i+ +n_guard<N+1  %����if��ʾ���Ե�Ԫ�Ҳ����ݸ�������n_guard����С��n_guard+n_train��
            nsum = nsum+sum(abs(signal(i+ +n_guard:N)));
            n_tt = N-i-n_guard;
            a = (n_train+n_tt)*(Pfa^(-1/(n_train + n_tt))-1);
        end
        nsum = nsum/n_train+n_tt;
        nsum = sqrt(a*nsum);
    end
    
    signal_cfar(i) = nsum;
end

end

