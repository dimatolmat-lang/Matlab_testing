%% =====================================================================
%  АНАЛИЗ 4-МИКРОФОННОЙ ЗАПИСИ ТЕСТОВОГО СИГНАЛА
%  -------------------------------------------------------------------
%  Что изменилось по сравнению со старой версией:
%   * задержка: GCC-PHAT + параболическая интерполяция (точность ~мкс),
%     окно поиска сужено и настраивается (maxDelaySec);
%   * когерентность считается корректно (в старой версии окно на весь
%     кадр давало 1 усреднение => когерентность всегда ~1) и накапли-
%     вается по частоте за всю запись;
%   * тишина не портит графики (NaN вместо нулей и -Inf);
%   * БЛОК A: анализ тонов — частота, чувствительность каналов, THD,
%     линейность (наклон дБ/дБ), полярность после компенсации задержки;
%   * БЛОК B: шумовой пол из пауз, SNR;
%   * БЛОК C: сохранение результатов (tones.csv + analysis_results.mat);
%   * БЛОК D: оценка направления на источник (DOA) по задержкам;
%   * итоговая сводка в командное окно.
%
%  КОНВЕНЦИЯ ЗНАКА ЗАДЕРЖКИ (важно!):
%   delayMs > 0  =>  сигнал канала приходит РАНЬШЕ опорного (refCh);
%   delayMs < 0  =>  позже. Если нужна противоположная конвенция —
%   поменяйте знак в функции gccphat_lag в конце файла.
%
%  Полный проход по ~40-минутной записи занимает несколько минут.
%% =====================================================================
close all; clear; clc;

%% 0. НАСТРОЙКИ (правьте под свою установку) ---------------------------
files = {'channel_0_merged.wav','channel_1_merged.wav', ...
         'channel_2_merged.wav','channel_3_merged.wav'};
refCh = 1;                     % опорный канал (столбец 1 = channel_0)

% --- сегментация обзора ---
frameSec   = 1.0;              % окно обзора, с
overlapSec = 0.5;              % перекрытие, с
specWinSec = 0.2;              % окно спектр. оценок внутри кадра, с

% --- поиск задержки: окно ±maxDelaySec ---
% Берите чуть больше Dmax/c (Dmax - самое большое расстояние между
% микрофонами, м; c = 343 м/с). НЕ завышайте: на тонах выше
% ~1/(2*maxDelaySec) задержка становится неоднозначной (лепестки).
maxDelaySec = 0.0015;           % ±2 мс (хватает для базы до ~0.6 м)

% --- частоты для графиков АЧХ/ФЧХ и полоса анализа ---
% (сигнал генератора 100..2000 Гц, поэтому 10 кГц убрано - там шум)
freqsOfInterest = [110 500 1000 1900];
fMinAnal = 50;   fMaxAnal = 3000;

% --- геометрия микрофонов: строки = channel_0..3, столбцы [x y], м ---
% ОБЯЗАТЕЛЬНО поставьте свои координаты для блока DOA!
micPos = [-0.2475 0; -0.0825 0; 0.0825 0; 0.2475 0.2];
cSnd = 343;                    % скоросVbrhjть звука, м/с

saveResults = true;            % писать tones.csv / analysis_results.mat
storePhase  = true;            % тепловая карта фазы (пара ch1-ch0)

%% 1. Загрузка и подготовка --------------------------------------------
N = numel(files);
raw = cell(N,1);
fsAll = zeros(N,1);
for i = 1:N
    [x, fs_i] = audioread(files{i});
    if size(x,2) > 1
        x = mean(x,2);         % если вдруг стерео — в моно
    end
    raw{i}   = x(:);
    fsAll(i) = fs_i;
    fprintf('ch%d: %g Гц, %.1f с\n', i-1, fs_i, numel(x)/fs_i);
end
if any(fsAll ~= fsAll(1))
    error('Частоты дискретизации файлов различаются: %s', mat2str(fsAll.'));
end
fs = fsAll(1);

minLen = min(cellfun(@numel, raw));
data = zeros(minLen, N);
for i = 1:N
    data(:,i) = raw{i}(1:minLen);
end
clear raw;

data = data - mean(data,1);    % убрать DC
L = size(data,1);

%% 2. Параметры сегментации --------------------------------------------
frameLen  = round(frameSec * fs);
overlap   = round(overlapSec * fs);
step      = frameLen - overlap;
numFrames = floor((L - overlap) / step);
t         = ((0:numFrames-1)*step + frameLen/2) / fs;

specWinLen   = round(specWinSec * fs);
specWin      = hann(specWinLen);
specNoverlap = round(0.5 * specWinLen);
specNfft     = 2^nextpow2(specWinLen);
Fgrid        = (0:specNfft/2)' * fs / specNfft;
nBins        = numel(Fgrid);
valid        = Fgrid > fMinAnal & Fgrid < fMaxAnal;

maxLag  = round(maxDelaySec * fs);
fLagMax = 1/(2*maxDelaySec);   % ниже этой частоты задержка на тонах однозначна

%% 3. Память под результаты --------------------------------------------
rmsVal   = nan(numFrames, N);
peakVal  = nan(numFrames, N);
delayMs  = nan(numFrames, N-1);
polarity = nan(numFrames, N-1);
cohMed   = nan(numFrames, N-1);
magAt    = nan(numFrames, N-1, numel(freqsOfInterest));
phaseAt  = nan(numFrames, N-1, numel(freqsOfInterest));
cohSum   = zeros(nBins, N-1);  % накопитель когерентности по частоте
cohCnt   = 0;
if storePhase
    phaseMat = nan(numFrames, nBins);
end

%% 4. Основной цикл ------------------------------------------------------
for k = 1:numFrames
    idx = (k-1)*step + (1:frameLen);
    seg = data(idx,:);

    % пропуск тишины (порог ~-80 dBFS); NaN остаются пропусками на графиках
    if max(sqrt(mean(seg.^2,1))) < 1e-4
        continue;
    end

    rmsVal(k,:)  = sqrt(mean(seg.^2,1));
    peakVal(k,:) = max(abs(seg),[],1);

    % спектры каналов — один раз на кадр
    Pxx = nan(nBins, N);
    for ch = 1:N
        [Pc, ~] = pwelch(seg(:,ch), specWin, specNoverlap, specNfft, fs);
        Pxx(:,ch) = Pc;
    end

    x = seg(:,refCh);
    for ch = 1:N
        if ch == refCh, continue; end
        y = seg(:,ch);
        j = ch - (ch > refCh);

        % --- GCC-PHAT: задержка и полярность ---
        [delayMs(k,j), polarity(k,j)] = gccphat_lag(x, y, fs, maxLag);

        % --- кросс-спектр, передаточная функция, когерентность ---
        [Pxy, ~] = cpsd(x, y, specWin, specNoverlap, specNfft, fs);
        H   = Pxy ./ Pxx(:,refCh);
        coh = abs(Pxy).^2 ./ (Pxx(:,refCh) .* Pxx(:,ch));
        cohSum(:,j) = cohSum(:,j) + coh;
        cohMed(k,j) = median(coh(valid));

        for fi = 1:numel(freqsOfInterest)
            [~, idxF] = min(abs(Fgrid - freqsOfInterest(fi)));
            magAt(k,j,fi)   = abs(H(idxF));
            phaseAt(k,j,fi) = angle(H(idxF));
        end

        if storePhase && ch == 2          % пара ch1-ch0
            phaseMat(k,:) = angle(H);
        end
    end
    cohCnt = cohCnt + 1;
end

%% 5. Обзорные графики ---------------------------------------------------
figure('Name','RMS по каналам');
plot(t, 20*log10(rmsVal + eps));
grid on; xlabel('Время, с'); ylabel('RMS, dBFS');
title('Уровень RMS'); legend(compose('ch%d',0:N-1));

figure('Name', sprintf('Задержка (GCC-PHAT, ±%.1f мс)', maxDelaySec*1000));
plot(t, delayMs);
grid on; xlabel('Время, с'); ylabel('Задержка, мс');
title('Задержка относительно опорного (>0 = канал раньше)');
legend(compose('ch%d-ch0',1:N-1));

figure('Name','Когерентность (медиана по полосе)');
plot(t, cohMed); ylim([0 1]); grid on;
xlabel('Время, с'); ylabel('Медианная когерентность');
legend(compose('ch%d-ch0',1:N-1));

figure('Name','Когерентность vs частота (средняя за запись)');
plot(Fgrid, cohSum / max(cohCnt,1)); ylim([0 1]); grid on;
xlim([fMinAnal fMaxAnal]);
xlabel('Частота, Гц'); ylabel('Когерентность');
legend(compose('ch%d-ch0',1:N-1));

if storePhase
    figure('Name','Разность фаз ch1-ch0');
    imagesc(t, Fgrid, phaseMat.');
    axis xy; colorbar;
    xlabel('Время, с'); ylabel('Частота, Гц');
    title('Разность фаз, рад (пара ch1-ch0)');
    ylim([fMinAnal fMaxAnal]);
end

% вектоскоп на самом громком кадре
v = rmsVal(:,refCh);
if any(~isnan(v))
    [~, kLoud] = max(v);
    idx = (kLoud-1)*step + (1:frameLen);
    figure('Name','Вектоскоп ch0-ch1');
    plot(data(idx,refCh), data(idx,2)); axis equal; grid on;
    xlabel('ch0'); ylabel('ch1');
    title(sprintf('Вектоскоп, t = %.1f с (самый громкий кадр)', t(kLoud)));
end

% АЧХ/ФЧХ на выбранных частотах (пара ch1-ch0; поменяйте 1 на другую j)
figure('Name','АЧХ и ФЧХ на выбранных частотах (пара ch1-ch0)');
colsF = lines(numel(freqsOfInterest));
for fi = 1:numel(freqsOfInterest)
    subplot(2,1,1);
    plot(t, 20*log10(magAt(:,1,fi)+eps), 'Color', colsF(fi,:), ...
         'DisplayName', sprintf('%d Гц', freqsOfInterest(fi)));
    hold on;
    subplot(2,1,2);
    plot(t, phaseAt(:,1,fi), 'Color', colsF(fi,:), ...
         'DisplayName', sprintf('%d Гц', freqsOfInterest(fi)));
    hold on;
end
subplot(2,1,1); grid on; ylabel('|H|, дБ');
title('Отношение амплитуд ch1/ch0'); legend show; hold off;
subplot(2,1,2); grid on; ylabel('Разность фаз, рад'); xlabel('Время, с');
legend show; hold off;

%% 6. ДЕТЕКЦИЯ СЕГМЕНТОВ (тон/пауза) по огибающей опорного канала --------
hopE = round(0.05*fs);   winE = round(0.2*fs);
nE = floor((L-winE)/hopE) + 1;
env = zeros(nE,1);
for e = 1:nE
    idx = (e-1)*hopE + (0:winE-1) + 1;
    env(e) = sqrt(mean(data(idx,refCh).^2));
end

% порог «тихо/громко»: 1D k-means по логарифму огибающей
le = log10(max(env, 1e-10));
mu = [min(le) max(le)];
for it = 1:100
    lab = abs(le - mu(2)) < abs(le - mu(1));   % true = «громко»
    if ~any(lab) || all(lab), break; end
    mu = [mean(le(~lab)), mean(le(lab))];
end
if 10^(mu(2)-mu(1)) < 10
    warning('Разница уровень/пауза < 10 дБ — сегментация может быть ненадёжной');
end
act = medfilt1(double(env > 10^mean(mu)), 5) > 0.5;

dd  = diff([false; act; false]);
stE = find(dd ==  1);   fnE = find(dd == -1) - 1;
ok  = (fnE - stE + 1)*hopE >= round(0.4*fs);
stE = stE(ok);   fnE = fnE(ok);
nT  = numel(stE);
fprintf('\nНайдено активных сегментов: %d\n', nT);

%% 7. ИЗМЕРЕНИЯ НА КАЖДОМ СЕГМЕНТЕ --------------------------------------
fT     = nan(nT,1);    % уточнённая частота тона, Гц
isTone = false(nT,1);  % сегмент — стабильный тон (не свип)
AT     = nan(nT,N);    % амплитуда 1-й гармоники, доли FS
THD    = nan(nT,N);    % THD, %
dLT    = nan(nT,N-1);  % задержка (GCC-PHAT) на тоне, мс
phT    = nan(nT,N-1);  % разность фаз Y-X на частоте тона, рад
nH     = 5;            % число гармоник для THD
trimFrac = 0.10;       % отбрасывать края сегмента при оценке фазы

for s = 1:nT
    i1 = (stE(s)-1)*hopE + 1;
    i2 = min((fnE(s)-1)*hopE + winE, L);
    n  = i2 - i1 + 1;
    if n < round(0.4*fs), continue; end
    seg = data(i1:i2,:);

    % стабильность частоты по четвертям сегмента (отсев свипов)
    fq = nan(1,4);
    for q = 1:4
        a = i1 + round((q-1)*n/4);
        b = i1 + round(q*n/4) - 1;
        fq(q) = fft_peak_freq(seg(a-i1+1:b-i1+1, refCh), fs, fMaxAnal);
    end
    if any(isnan(fq)), continue; end
    isTone(s) = (max(fq) - min(fq)) < 5;      % < 5 Гц — тон, иначе свип
    if ~isTone(s), continue; end

    % грубая частота -> уточнение частоты и фазы опорного канала
    f0 = fft_peak_freq(seg(:,refCh), fs, fMaxAnal);
    if isnan(f0), continue; end
    [f0, phiR, okR] = demod_freq_phase(seg(:,refCh), f0, fs, trimFrac);
    if ~okR, continue; end
    fT(s) = f0;

    tt = (0:n-1)'/fs;   tc = tt - mean(tt);
    w  = hann(n);       sw = sum(w);
    e0 = exp(-1i*2*pi*f0*tc);

    for ch = 1:N
        yw = seg(:,ch) .* w;
        Ah = zeros(1,nH);
        for h = 1:nH
            if h*f0 >= 0.45*fs, break; end
            Ah(h) = 2*abs(sum(yw .* (e0.^h))) / sw;
        end
        AT(s,ch)  = Ah(1);
        THD(s,ch) = 100*sqrt(sum(Ah(2:end).^2)) / max(Ah(1), eps);

        if ch ~= refCh
            j = ch - (ch > refCh);
            dLT(s,j) = gccphat_lag(seg(:,refCh), seg(:,ch), fs, maxLag);
            [~, phiC, okC] = demod_freq_phase(seg(:,ch), f0, fs, trimFrac);
            if okC
                phT(s,j) = phiC - phiR;   % фаза канала минус фаза опорного
            end
        end
    end
end

iT = isTone & ~isnan(fT);
haveTones = sum(iT) >= 4;
if ~haveTones
    warning(['Стабильных тонов не найдено (%d из %d). Анализ тонов и DOA ', ...
             'пропущены (запись не от генератора?)'], sum(iT), nT);
end

% значения по умолчанию (используются в сохранении/сводке)
lag0Ms   = nan(1, N-1);   % медианная задержка пары по низкочастотным тонам
noiseRms = nan(1, N);
u        = [NaN NaN];
angleDeg = NaN;

%% 8. ГРАФИКИ ПО ТОНАМ ---------------------------------------------------
if haveTones
    fTv  = fT(iT);   ATv  = AT(iT,:);   THDv = THD(iT,:);
    dLTv = dLT(iT,:); phTv = phT(iT,:);
    nTv  = numel(fTv);
    cols = lines(N-1);

    % 8.1 чувствительность каналов
    figure('Name','Уровни и чувствительность по тонам');
    subplot(2,1,1);
    plot(fTv, 20*log10(ATv + eps), '.'); grid on;
    xlabel('Частота, Гц'); ylabel('Амплитуда 1-й гармоники, dBFS');
    title('Абсолютный уровень по каналам'); legend(compose('ch%d',0:N-1));
    subplot(2,1,2); hold on;
    hs = gobjects(1,N-1);
    for j = 1:N-1
        hs(j) = plot(fTv, 20*log10(ATv(:,j+1)./ATv(:,refCh) + eps), ...
                     '.', 'Color', cols(j,:));
    end
    grid on; xlabel('Частота, Гц'); ylabel('дБ отн. опорного');
    title('Относительная чувствительность / АЧХ пар'); hold off;
    legend(hs, compose('ch%d/ch0',1:N-1));

    % 8.2 задержка по тонам
    figure('Name','Задержка по тонам'); hold on;
    hs = gobjects(1,N-1);
    lowMask = fTv <= fLagMax;
    for j = 1:N-1
        hs(j) = plot(fTv, dLTv(:,j), '.', 'Color', cols(j,:));
        if any(lowMask)
            lag0Ms(j) = median(dLTv(lowMask,j), 'omitnan');
            plot([fMinAnal fMaxAnal], lag0Ms(j)*[1 1], ...
                 'Color', cols(j,:), 'LineWidth', 1.5);
        end
    end
    grid on; xlabel('Частота, Гц'); ylabel('Задержка, мс'); hold off;
    title(sprintf(['Задержка по тонам (линии — медиана по f ≤ %d Гц); ', ...
        'разброс выше ~%d Гц — неоднозначность лепестков'], ...
        round(fLagMax), round(fLagMax)));
    legend(hs, compose('ch%d-ch0',1:N-1));

    % 8.3 полярность (фаза после компенсации задержки)
    figure('Name','Полярность (фаза после компенсации задержки)');
    for j = 1:N-1
        subplot(N-1,1,j);
        if ~isnan(lag0Ms(j)) && any(~isnan(phTv(:,j)))
            phc = wrap_pi(phTv(:,j) - 2*pi*fTv*lag0Ms(j)/1000);
            plot(fTv, phc, '.'); hold on;
            plot([fMinAnal fMaxAnal], [0 0], 'g--');
            plot([fMinAnal fMaxAnal], [pi pi], 'r--');
            plot([fMinAnal fMaxAnal], [-pi -pi], 'r--'); hold off;
            d0  = median(abs(wrap_pi(phc)), 'omitnan');        % близость к 0
            dpi = median(abs(wrap_pi(phc - pi)), 'omitnan');   % близость к ±pi
            if dpi < d0, verd = 'ИНВЕРСИЯ'; else, verd = 'совпадает'; end
            title(sprintf('ch%d-ch0: полярность — %s (медианы: %.2f и %.2f рад)', ...
                j, verd, d0, dpi));
        else
            plot(fTv, phTv(:,j), '.');
            title(sprintf('ch%d-ch0: задержка/фаза не оценены', j));
        end
        grid on; ylim([-pi pi]); ylabel('рад');
    end
    xlabel('Частота, Гц');

    % 8.4 THD
    figure('Name','THD по тонам');
    plot(fTv, THDv); grid on; set(gca,'YScale','log');
    xlabel('Частота, Гц'); ylabel('THD, %');
    title('Нелинейные искажения (гармоники 2-5)');
    legend(compose('ch%d',0:N-1));

    % 8.5 линейность тракта (наклон дБ/дБ, норма = 1)
    lv = [25 50 75 100];               % уровни генератора, %
    slopeAll = [];   slopeCh = [];
    i = 1;
    while i <= nTv
        j2 = i;
        while j2 < nTv && abs(fTv(j2+1)-fTv(i)) < 1, j2 = j2+1; end
        if j2 - i + 1 == numel(lv)     % группа из 4 уровней одной частоты
            xdb = 20*log10(lv);
            for ch = 1:N
                ydb = sort(20*log10(ATv(i:j2,ch) + eps));
                p = polyfit(xdb, ydb, 1);
                slopeAll(end+1,1) = p(1);   %#ok<AGROW>
                slopeCh(end+1,1)  = ch;     %#ok<AGROW>
            end
        end
        i = j2 + 1;
    end
    if ~isempty(slopeAll)
        figure('Name','Линейность тракта (наклон дБ/дБ, норма = 1)');
        for ch = 1:N
            subplot(2,2,ch);
            histogram(slopeAll(slopeCh==ch), 0.8:0.01:1.2);
            grid on;
            title(sprintf('ch%d (медиана %.3f)', ch-1, ...
                median(slopeAll(slopeCh==ch))));
            xlabel('наклон, дБ/дБ');
        end
    end
end

% клиппинг
fprintf('\nКлиппинг: сэмплов выше 0.995 FS:\n');
for ch = 1:N
    fprintf('  ch%d: %.4f %%\n', ch-1, 100*sum(abs(data(:,ch))>=0.995)/L);
end

%% 9. ШУМ В ПАУЗАХ --------------------------------------------------------
noiseMask = false(L,1);
for s = 1:numel(stE)
    a = (fnE(s)-1)*hopE + winE + round(0.3*fs);   % +0.3 с после сегмента
    if s < numel(stE)
        b = (stE(s+1)-1)*hopE - round(0.05*fs);   % до начала следующего
    else
        b = L;
    end
    a = max(a,1);   b = min(b,L);
    if b > a, noiseMask(a:b) = true; end
end
if nnz(noiseMask) >= round(2*fs)
    figure('Name','Спектр шума (паузы)'); hold on;
    for ch = 1:N
        [Pn, Fn] = pwelch(data(noiseMask,ch), hann(round(0.25*fs)), ...
                          round(0.125*fs), 2^nextpow2(round(0.25*fs)), fs);
        plot(Fn, 10*log10(Pn + eps));
    end
    grid on; xlim([0 min(5000, fs/2)]);
    xlabel('Частота, Гц'); ylabel('PSD, дБ/Гц');
    title('Шумовой пол в паузах (виден фон 50 Гц и пр.)');
    legend(compose('ch%d',0:N-1)); hold off;

    noiseRms = sqrt(mean(data(noiseMask,:).^2, 1));
else
    warning('Паузы не найдены — оценка шума пропущена');
end

%% 10-11. DOA ЛИНЕЙНОЙ РЕШЁТКИ, ДИАГРАММЫ НАПРАВЛЕННОСТИ, СОХРАНЕНИЕ --------
% Угол отсчитывается от нормали к решётке: 0° = источник по центру,
% +90° = со стороны +X (где ch3), -90° = со стороны -X (где ch0).
% Линейная решётка не различает перед/тыл — считаем источник спереди (y>0).
% Если измеренный угол зеркален к задаваемому (задано +30, вышло -30) —
% инвертируйте первый столбец micPos: micPos(:,1) = -micPos(:,1);

dMic      = 0.12;              % база соседних микрофонов, м
fAliasAdj = cSnd/(2*dMic);     % ~1429 Гц: выше фаза соседних пар неоднозначна

% --- истинный угол из имени папки «Угол_N» (если скрипт запущен из неё) ---
trueAngle = NaN;
mAng = regexp(pwd, 'Угол_?(-?\d+)', 'once', 'tokens');
if isempty(mAng)
    mAng = regexp(pwd, '(?i)angle_?(-?\d+)', 'once', 'tokens');
end
if ~isempty(mAng)
    trueAngle = str2double(mAng{1});
    fprintf('\nЗаданный угол (из имени папки): %g°\n', trueAngle);
end

% значения по умолчанию
angleDeg = NaN;  srcPos = [NaN NaN];  u = [NaN NaN];
tauAdjGCC = nan(1,3);  tauAdjPh = nan(1,3);  tauSweepMs = nan(1,N-1);
swAdjMs = nan(1,3);    thetaFull = nan(nT,1);

%% --- 10.2 Широкополосные задержки по СВИПАМ (GCC-PHAT) — самые надёжные ---
swTau = [];                                  % строки: задержки ch1,ch2,ch3 отн. ch0, мс
for s = 1:nT
    if isTone(s), continue; end              % обрабатываем только не-тоны (свипы)
    i1 = (stE(s)-1)*hopE + 1;
    i2 = min((fnE(s)-1)*hopE + winE, L);
    if (i2 - i1 + 1) < round(3.0*fs), continue; end
    cLen = min(i2 - i1 + 1, round(2.0*fs));  % центральные 2 с свипа
    a = i1 + floor((i2 - i1 + 1 - cLen)/2);
    b = a + cLen - 1;
    x = data(a:b, refCh);
    row = nan(1, N-1);
    for ch = 1:N
        if ch == refCh, continue; end
        j = ch - (ch > refCh);
        row(j) = gccphat_lag(x, data(a:b,ch), fs, maxLag);
    end
    swTau(end+1,:) = row;                    %#ok<AGROW>
end
if ~isempty(swTau)
    tauSweepMs = median(swTau, 1, 'omitnan');
    swAdjMs = [tauSweepMs(1), tauSweepMs(2)-tauSweepMs(1), tauSweepMs(3)-tauSweepMs(2)];
    fprintf('Задержки по свипам (отн. ch0), мс: %s\n', sprintf('%+7.4f ', tauSweepMs));
end

%% --- 10.3 Соседние пары: GCC по тонам и фазовый метод ----------------------
if haveTones
    fTv = fT(iT);  dLTv = dLT(iT,:);  phTv = phT(iT,:);  ATv = AT(iT,:);
    nTv = numel(fTv);

    % GCC по низкочастотным тонам (где корреляция однозначна)
    lowMask = fTv <= fLagMax;
    if any(lowMask)
        tauAdjGCC = median([dLTv(lowMask,1), ...
                            dLTv(lowMask,2)-dLTv(lowMask,1), ...
                            dLTv(lowMask,3)-dLTv(lowMask,2)], 1, 'omitnan');
    end

    % Фазовый метод: задержка соседей из разности фаз на частоте тона
    % (однозначна при f < c/(2d) = 1429 Гц — покрывает почти все тоны)
    phM = [wrap_pi(phTv(:,1)), ...
           wrap_pi(phTv(:,2)-phTv(:,1)), ...
           wrap_pi(phTv(:,3)-phTv(:,2))];           % nTv x 3, рад
    phMask = fTv <= 0.98*fAliasAdj;
    thT = nan(nTv, 3);                              % угол по каждой паре, °
    for j = 1:3
        ok = phMask & ~isnan(phM(:,j));
        if any(ok)
            tauAdjPh(j) = 1000*median(phM(ok,j)./(2*pi*fTv(ok)), 'omitnan');
            sT = cSnd*(phM(:,j)./(2*pi*fTv))/dMic;  % = sin(угла)
            thT(:,j) = asind(min(max(sT,-1),1));
        end
    end
    thT(~phMask,:) = NaN;
    thetaTone = median(thT, 2, 'omitnan');          % объединённый угол по тону
    thetaFull(iT) = thetaTone;

    figure('Name','DOA по тонам (соседние пары, фазовый метод)','Color','w');
    plot(fTv, thT, '.'); grid on; hold on;
    leg = compose('пара ch%d-ch%d',[0 1 2],[1 2 3]);
    if ~isempty(thetaTone)
        plot([100 2000], median(thetaTone,'omitnan')*[1 1], 'k-', 'LineWidth', 1.5);
        leg{end+1} = 'медиана';
    end
    if ~isnan(trueAngle)
        plot([100 2000], trueAngle*[1 1], 'r--', 'LineWidth', 1.5);
        leg{end+1} = 'задано';
    end
    hold off; xlim([100 2000]); ylim([-90 90]);
    xlabel('Частота тона, Гц'); ylabel('Угол, °');
    title(sprintf(['Угол по соседним парам (фазовый метод); f > %d Гц скрыты ', ...
        '— неоднозначность фазы'], round(0.98*fAliasAdj)));
    legend(leg);
end

%% --- 10.4 Дальнее поле: итоговый угол ---------------------------------------
est = [];  estName = {};
if haveTones
    for j = 1:3
        if ~isnan(tauAdjGCC(j))
            est(end+1) = asind(min(max(cSnd*tauAdjGCC(j)/1000/dMic,-1),1));
            estName{end+1} = sprintf('GCC-тоны, пара %d', j);      %#ok<AGROW>
        end
        if ~isnan(tauAdjPh(j))
            est(end+1) = asind(min(max(cSnd*tauAdjPh(j)/1000/dMic,-1),1));
            estName{end+1} = sprintf('фаза тонов, пара %d', j);    %#ok<AGROW>
        end
    end
end
if ~any(isnan(swAdjMs))
    for j = 1:3
        est(end+1) = asind(min(max(cSnd*swAdjMs(j)/1000/dMic,-1),1));
        estName{end+1} = sprintf('свип, пара %d', j);              %#ok<AGROW>
    end
end
if ~isempty(est)
    angleDeg = median(est);
    u = [sind(angleDeg), cosd(angleDeg)];
end

%% --- 10.5 Ближнее поле: положение источника (x, y, дистанция) ---------------
if ~any(isnan(tauSweepMs))
    tauAdjSec = swAdjMs/1000;      srcTauName = 'свипы';
elseif ~any(isnan(tauAdjPh))
    tauAdjSec = tauAdjPh/1000;     srcTauName = 'фаза тонов';
elseif ~any(isnan(tauAdjGCC))
    tauAdjSec = tauAdjGCC/1000;    srcTauName = 'GCC тонов';
else
    tauAdjSec = [];
end

if ~isempty(tauAdjSec) && max(abs(tauAdjSec)) > 15e-6
    % модель: tau_j = (|p_j - s| - |p_{j+1} - s|)/c   (tau>0: старший раньше)
    idxLo = [1 2 3];   idxHi = [2 3 4];
    costNF = @(sv) sum(((hypot(micPos(idxLo,1)-sv(1), micPos(idxLo,2)-sv(2)) ...
                       - hypot(micPos(idxHi,1)-sv(1), micPos(idxHi,2)-sv(2)))/cSnd ...
                       - tauAdjSec).^2);
    xg = -1.5:0.05:1.5;   yg = 0.1:0.05:4;   % диапазон поиска — подстройте под дистанцию
    [Xg, Yg] = meshgrid(xg, yg);
    Cg = arrayfun(@(xx,yy) costNF([xx yy]), Xg, Yg);
    [~, km] = min(Cg(:));
    sf = fminsearch(costNF, [Xg(km), Yg(km)], optimset('Display','off'));
    if all(isfinite(sf))
        sf(2) = abs(sf(2));                   % источник спереди (y > 0)
        srcPos = sf;
        fprintf('\nБлижнее поле (по %s): x=%+.3f м, y=%.3f м, R=%.3f м, угол %+.2f°\n', ...
            srcTauName, srcPos(1), srcPos(2), hypot(srcPos(1),srcPos(2)), ...
            atan2d(srcPos(1), srcPos(2)));
        if ~isnan(trueAngle) && abs(trueAngle) < 1
            fprintf('Проверка центровки: смещение источника от оси %+0.3f м\n', srcPos(1));
        end
        if abs(atan2d(srcPos(1),srcPos(2))) < 5
            fprintf('(при углах ~0° дистанция по TDOA определяется плохо)\n');
        end
    end
elseif ~isempty(tauAdjSec)
    fprintf('\nВсе задержки ~0: источник по центру (угол ~0°), позиция по TDOA не определяется\n');
end

%% --- 10.6 Диаграммы направленности решётки (по измеренным тонам) ------------
if haveTones
    fBeam = [200 500 1000 1900];   % 1900 Гц — выше f_гр, будут лепестки!
    thScan = -90:0.25:90;
    xMic = micPos(:,1);
    figure('Name','ДН решётки (Bartlett, восстановлена по тонам)','Color','w');
    for bi = 1:numel(fBeam)
        [~, iB] = min(abs(fTv - fBeam(bi)));
        f0 = fTv(iB);
        z = ATv(iB,:) .* exp(1i*[0, phTv(iB,:)]);     % комплексные амплитуды отн. ch0
        Bp = zeros(size(thScan));
        for ti = 1:numel(thScan)
            Bp(ti) = abs(sum(z .* exp(-1i*2*pi*f0*xMic*sind(thScan(ti))/cSnd)))^2;
        end
        Bp = 10*log10(Bp/max(Bp) + eps);
        subplot(2,2,bi);
        plot(thScan, Bp, 'LineWidth', 1); grid on; hold on;
        yl = [-25 3];
        if ~isnan(angleDeg),  plot(angleDeg*[1 1],  yl, 'r--'); end
        if ~isnan(trueAngle), plot(trueAngle*[1 1], yl, 'm:');  end
        hold off; xlim([-90 90]); ylim(yl);
        xlabel('Угол сканирования, °'); ylabel('дБ');
        if f0 > fAliasAdj
            title(sprintf('%d Гц — выше f_гр (%d Гц): дифракционные лепестки', ...
                round(f0), round(fAliasAdj)));
        else
            title(sprintf('%d Гц', round(f0)));
        end
    end
end

%% --- 10.7 Сводка DOA и сохранение -------------------------------------------
fprintf('\n--- СВОДКА DOA (линейка, d=%d мм, f_гр=%d Гц) ---\n', ...
    round(1000*dMic), round(fAliasAdj));
fprintf('Задержки соседних пар, мкс (tau>0 — старший канал раньше):\n');
fprintf('%10s %12s %12s %12s\n', 'пара', 'GCC(тоны)', 'фаза(тоны)', 'свип');
for j = 1:3
    fprintf('  ch%d-ch%d %12.1f %12.1f %12.1f\n', j-1, j, ...
        tauAdjGCC(j)*1e3, tauAdjPh(j)*1e3, swAdjMs(j)*1e3);
end
if ~isempty(est)
    fprintf('Оценки угла (дальнее поле):\n');
    for e = 1:numel(est)
        fprintf('  %-22s %+7.2f°\n', estName{e}, est(e));
    end
    fprintf('  ИТОГ (медиана)         %+7.2f°\n', angleDeg);
    if ~isnan(trueAngle)
        fprintf('  Задано %g°, ошибка %+.2f°\n', trueAngle, wrap_180(angleDeg-trueAngle));
    end
end

if saveResults
    Ttab = table(fT, isTone, AT, THD, dLT, phT, thetaFull, ...
        'VariableNames', {'f_Hz','isTone','Amp','THD_pct','delay_ms','phase_rad','angle_deg'});
    writetable(Ttab, 'tones.csv');
    save('analysis_results.mat', 'fs', 'micPos', 'fT', 'isTone', 'AT', 'THD', ...
         'dLT', 'phT', 'noiseRms', 'lag0Ms', 'tauAdjGCC', 'tauAdjPh', ...
         'tauSweepMs', 'angleDeg', 'srcPos', 'trueAngle', 'u');
    fprintf('Сохранено: tones.csv, analysis_results.mat\n');
end

%% 12. СВОДКА В КОМАНДНОЕ ОКНО ----------------------------------------------
fprintf('\n==================== СВОДКА ====================\n');
fprintf('Каналов: %d, fs = %g Гц, длительность %.1f с\n', N, fs, L/fs);
if haveTones
    fprintf('\nЧувствительность (медиана, дБ отн. ch%d) и THD:\n', refCh-1);
    aRef = max(median(ATv(:,refCh),'omitnan'), eps);
    for ch = 1:N
        fprintf('  ch%d: %+6.2f дБ   THD(медиана) %.2f %%\n', ch-1, ...
            20*log10(median(ATv(:,ch),'omitnan')/aRef), ...
            median(THDv(:,ch),'omitnan'));
    end
    fprintf('\nЗадержки (медиана по f ≤ %d Гц) и полярность:\n', round(fLagMax));
    for j = 1:N-1
        if isnan(lag0Ms(j)) || ~any(~isnan(phTv(:,j)))
            fprintf('  ch%d-ch0: не оценена\n', j);
        else
            phc = wrap_pi(phTv(:,j) - 2*pi*fTv*lag0Ms(j)/1000);
            d0  = median(abs(wrap_pi(phc)), 'omitnan');
            dpi = median(abs(wrap_pi(phc - pi)), 'omitnan');
            if dpi < d0, verd = 'ИНВЕРСИЯ'; else, verd = 'совпадает'; end
            fprintf('  ch%d-ch0: %+7.1f мкс   полярность: %s\n', ...
                j, lag0Ms(j)*1000, verd);
        end
    end
    if ~any(isnan(noiseRms))
        fprintf('\nШум и SNR:\n');
        sigRms = median(ATv, 1, 'omitnan') / sqrt(2);
        for ch = 1:N
            fprintf(['  ch%d: шум %6.1f dBFS, сигнал %6.1f dBFS, ', ...
                     'SNR %5.1f дБ\n'], ch-1, ...
                20*log10(max(noiseRms(ch),eps)), ...
                20*log10(max(sigRms(ch),eps)), ...
                20*log10(max(sigRms(ch),eps)/max(noiseRms(ch),eps)));
        end
    end
    if ~isnan(angleDeg)
        fprintf('\nНаправление на источник: %.1f° (0° = по центру, +90° = сторона ch3)\n', angleDeg);
    end
end
fprintf('=================================================\n');

%% ======================= ЛОКАЛЬНЫЕ ФУНКЦИИ ==============================

function [lagMs, pol] = gccphat_lag(x, y, fs, maxLag)
% GCC-PHAT + параболическая интерполяция пика.
% КОНВЕНЦИЯ: lagMs > 0  =>  y приходит РАНЬШЕ x на lagMs мс.
nfft = 2^nextpow2(numel(x));
G = fft(x, nfft) .* conj(fft(y, nfft));
r = real(ifft(G ./ (abs(G) + 1e-20)));
maxLag = max(1, min(maxLag, floor(nfft/2)-1));
c = [r(end-maxLag+1:end); r(1:maxLag+1)];      % лаги -maxLag .. +maxLag
[~, im] = max(abs(c));
pol = sign(c(im));
lag = im - (maxLag + 1);
if im > 1 && im < numel(c)                     % субвыборочная точность
    a = c(im-1);  b = c(im);  cc = c(im+1);
    den = a - 2*b + cc;
    if abs(den) > 1e-12
        lag = lag + 0.5*(a - cc)/den;
    end
end
lagMs = lag / fs * 1000;
end

function f = fft_peak_freq(x, fs, fmax)
% Частота главного пика спектра: окно Ханна, паддинг x4, парабола по log|X|.
x = x(:);
n = numel(x);
if n < 16, f = NaN; return; end
nfft = 2^nextpow2(4*n);
X = abs(fft(x .* hann(n), nfft));
hi = max(3, min(round(fmax/fs*nfft), floor(nfft/2)));
[~, ip] = max(X(2:hi));   ip = ip + 1;
if ip <= 1 || ip >= numel(X), f = NaN; return; end
a = log(X(ip-1)+eps);  b = log(X(ip)+eps);  c = log(X(ip+1)+eps);
den = a - 2*b + c;
if abs(den) < 1e-12
    f = (ip-1)*fs/nfft;
else
    f = (ip - 1 + 0.5*(a-c)/den) * fs/nfft;
end
end

function [f, phi, ok] = demod_freq_phase(x, f0, fs, trimFrac)
% Уточнение частоты тона и его фазы в ЦЕНТРЕ сегмента:
% демодуляция на f0, unwrap фазы, линейная аппроксимация.
% phi — фаза основной гармоники в центре сегмента (рад).
x = x(:);
n = numel(x);
i1 = round(trimFrac*n) + 1;
i2 = n - round(trimFrac*n);
if i2 - i1 + 1 < 64, f = f0; phi = NaN; ok = false; return; end
t  = ((i1:i2)' - 1) / fs;
tc = t - mean(t);
z  = x(i1:i2) .* exp(-1i*2*pi*f0*tc);
p  = unwrap(angle(z));
pf = polyfit(tc, p, 1);
df = pf(1) / (2*pi);
rr = rms(p - polyval(pf, tc));
ok = ~any(isnan(pf)) && abs(df) < 5 && rr < 0.5;
if ok
    f   = f0 + df;
    phi = pf(2);
else
    f = f0;  phi = NaN;
end
end

function a = wrap_pi(x)
a = mod(x + pi, 2*pi) - pi;
end

function a = wrap_180(x)
a = mod(x + 180, 360) - 180;
end