%% 1. Загрузка и подготовка
files = {'channel_0_merged.wav','channel_1_merged.wav', ...
         'channel_2_merged.wav','channel_3_merged.wav'};
N = numel(files);
refCh = 1; % канал 0 — опорный

data = []; fs = [];
for i = 1:N
    [x, fs_i] = audioread(files{i});
    if size(x,2) > 1, x = mean(x,2); end
    if i == 1
        fs = fs_i;
        data = x;
    else
        if fs_i ~= fs
            x = resample(x, fs, fs_i);
        end
        L = min(length(data), length(x));
        data = data(1:L);
        x = x(1:L);
        data = [data, x];
    end
end
data = data - mean(data,1); % убрать DC
L = size(data,1);

%% 2. Параметры сегментации
frameSec   = 1.0;   % длина окна, с
overlapSec = 0.5;   % перекрытие, с
frameLen   = round(frameSec * fs);
overlap    = round(overlapSec * fs);
step       = frameLen - overlap;
numFrames  = floor((L - overlap) / step);
t          = ((0:numFrames-1)*step + frameLen/2) / fs;

maxDelaySec = 0.05; % искать задержку в пределах ±50 мс
freqsOfInterest = [100 1000 10000]; % Гц

%% 3. Память под результаты
rmsVal   = zeros(numFrames, N);
peakVal  = zeros(numFrames, N);
delayMs  = zeros(numFrames, N-1);
polarity = zeros(numFrames, N-1);
cohMed   = zeros(numFrames, N-1);
magAt    = zeros(numFrames, N-1, numel(freqsOfInterest));
phaseAt  = zeros(numFrames, N-1, numel(freqsOfInterest));

% Тепловая карта фазы (для пары refCh <-> канал 2)
storePhase = true;
if storePhase
    nfft = 2^nextpow2(frameLen);
    F = (0:nfft/2)' * fs / nfft;
    phaseMat = nan(numFrames, numel(F));
end

win       = hann(frameLen);
noverlap  = round(frameLen*0.5);
nfft      = 2^nextpow2(frameLen);

%% 4. Основной цикл
for k = 1:numFrames
    idx = (k-1)*step + (1:frameLen);
    seg = data(idx,:);
    
    % пропуск тишины (порог -80 dBFS, при необходимости изменить)
    if max(sqrt(mean(seg.^2,1))) < 1e-4
        continue;
    end
    
    rmsVal(k,:)  = sqrt(mean(seg.^2,1));
    peakVal(k,:) = max(abs(seg),[],1);
    
    x = seg(:,refCh);
    for ch = 1:N
        if ch == refCh, continue; end
        y = seg(:,ch);
        j = ch - (ch > refCh);
        
        % --- Кросс-корреляция: задержка и полярность ---
        maxLag = round(maxDelaySec * fs);
        [c, lags] = xcorr(x, y, maxLag, 'coeff');
        [~, imax] = max(abs(c));
        delayMs(k,j)  = lags(imax) / fs * 1000; % мс
        polarity(k,j) = sign(c(imax));          % -1 = инверсия
        
        % --- Спектральные оценки ---
        [Pxy, F] = cpsd(x, y, win, noverlap, nfft, fs);
        [Pxx, ~] = cpsd(x, x, win, noverlap, nfft, fs);
        H = Pxy ./ Pxx;
        magRatio  = abs(H);
        phaseDiff = angle(H);
        [coh, ~]  = mscohere(x, y, win, noverlap, nfft, fs);
        
        valid = F > 20 & F < fs/2*0.95;
        cohMed(k,j) = median(coh(valid));
        
        for fi = 1:numel(freqsOfInterest)
            [~, idxF] = min(abs(F - freqsOfInterest(fi)));
            magAt(k,j,fi)   = magRatio(idxF);
            phaseAt(k,j,fi) = phaseDiff(idxF);
        end
        
        if storePhase && ch == 2
            phaseMat(k,:) = phaseDiff;
        end
    end
end

%% 5. Визуализация
figure('Name','RMS по каналам');
plot(t, 20*log10(rmsVal));
legend('ch0','ch1','ch2','ch3');
xlabel('Время, с'); ylabel('RMS, dBFS'); title('Уровень RMS'); grid on;

figure('Name','Задержка');
plot(t, delayMs);
legend('ch1-ch0','ch2-ch0','ch3-ch0');
xlabel('Время, с'); ylabel('Задержка, мс'); title('Задержка относительно ref'); grid on;

figure('Name','Когерентность');
plot(t, cohMed);
legend('ch1-ch0','ch2-ch0','ch3-ch0');
xlabel('Время, с'); ylabel('Медианная когерентность');
ylim([0 1]); grid on;

if storePhase
    figure('Name','Разность фаз ch1-ch0');
    imagesc(t, F, phaseMat.');
    axis xy; colorbar;
    xlabel('Время, с'); ylabel('Частота, Гц');
    title('Разность фаз, рад');
    ylim([20 fs/2*0.95]);
end

% Пример вектоскопа для среднего сегмента
k = round(numFrames/2);
idx = (k-1)*step + (1:frameLen);
figure('Name','Вектоскоп ch0-ch1');
plot(data(idx,refCh), data(idx,2));
axis equal; grid on;
xlabel('ch0'); ylabel('ch1');
title(sprintf('Вектоскоп, сегмент %d, t=%.1f c', k, t(k)));

% АЧХ/ФЧХ на выбранных частотах
figure('Name','АЧХ и ФЧХ на выбранных частотах');
for fi = 1:numel(freqsOfInterest)
    subplot(2,1,1);
    hold on;
    plot(t, 20*log10(magAt(:,1,fi)), ...
        'DisplayName', sprintf('ch1-ch0, %d Hz', freqsOfInterest(fi)));
    ylabel('Отношение амплитуд, дБ'); grid on; legend show;
    
    subplot(2,1,2);
    hold on;
    plot(t, phaseAt(:,1,fi), ...
        'DisplayName', sprintf('ch1-ch0, %d Hz', freqsOfInterest(fi)));
    ylabel('Разность фаз, рад'); xlabel('Время, с'); grid on; legend show;
end