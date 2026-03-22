%% ============ a Multi-operator Differential Evolution Algorithm (IMODE) ============
% Enhanced with cluster-level adaptive operator assignment.
% Robust for high dimensions and small clusters.
% =========================================================================
function [x, xold, fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,archive_freq,current_eval,res_det,F,cr, cluster_history ] = ...
    IMODE( x,xold, fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,archive_freq, xmin, xmax,  n,...
    PopSize,  current_eval, I_fno,res_det,Printing,Max_FES, G_Max, gg,F,cr, Kclusters, cluster_history)

%% ====================== 前置可调参数 ======================
default_params.Kclusters = round(PopSize/10);   % 默认簇数
default_params.fixed_ratio_start = 0.3;         % 初始固定簇比例
default_params.fixed_ratio_end   = 0.7;         % 结束时固定簇比例
default_params.alpha = 0.15;                    % EMA 更新簇级概率速率
default_params.print_each_gen = false;           % 每代输出簇算子信息
%% ==========================================================

persistent cluster_prob_prev centers_prev prev_K;
if isempty(cluster_prob_prev)
    cluster_prob_prev = [];
    centers_prev = [];
    prev_K = [];
end

if ~exist('Kclusters','var') || isempty(Kclusters)
    Kclusters = default_params.Kclusters;
end

% === 固定簇比例线性变化 ===
progress_ratio = 0;
if exist('Max_FES','var') && Max_FES>0
    progress_ratio = min(1, max(0, current_eval / Max_FES)); % 当前进度 [0,1]
end
fixed_ratio = default_params.fixed_ratio_start + ...
              progress_ratio * (default_params.fixed_ratio_end - default_params.fixed_ratio_start);
fixed_ratio = max(0, min(1, fixed_ratio)); % 限制范围

alpha = default_params.alpha;
print_each_gen = default_params.print_each_gen;

if ~exist('cluster_history','var') || isempty(cluster_history)
    cluster_history = [];
end

vi=zeros(PopSize,n);
  
%% calc CR and F
mem_rand_index = ceil(memory_size * rand(PopSize, 1));
mu_sf = archive_f(mem_rand_index);
mu_cr = archive_Cr(mem_rand_index);

cr = normrnd(mu_cr, 0.1);
term_pos = find(mu_cr == -1);
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);

F = mu_sf + 0.1 * tan(pi * (rand(1,PopSize) - 0.5));
pos = find(F <= 0);
while ~ isempty(pos)
    F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1,length(pos)) - 0.5));
    pos = find(F <= 0);
end
F = min(F, 1);
F = F';
[fitx,inddd]=sort(fitx);
x=x(inddd,:);
[cr,~]=sort(cr);

%% ======================== prepare clustering =================================
if isfield(archive,'pop')
    popAll = [x; archive.pop];
else
    popAll = x;
end

%Kclusters = max(2, min(PopSize, floor(Kclusters)));
Kclusters = max(2, round(PopSize/10));
K_use = min(Kclusters, PopSize);
cluster_history = [cluster_history, K_use];

opts = statset('MaxIter',200,'Display','off');
try
    [labels, centers] = kmeans(x, K_use, 'Replicates',3, 'Options', opts);
catch
    labels = ones(PopSize,1);
    centers = mean(x,1);
    K_use = 1;
end

%% ================== initialize / match cluster-level probabilities ==========
reinit_cluster_prob = true;
if ~isempty(cluster_prob_prev) && ~isempty(centers_prev)
    Kp = size(centers_prev,1);
    Kc = size(centers,1);
    mapping = zeros(Kp,1);
    used = false(Kc,1);
    for i = 1:Kp
        if size(centers,2) ~= size(centers_prev,2)
            reinit_cluster_prob = true;
            break;
        end
        d = sum((centers - centers_prev(i,:)).^2, 2);
        [~, idxmin] = min(d + used*1e12);
        mapping(i) = idxmin;
        used(idxmin) = true;
    end
    cluster_prob = ones(K_use,3) / 3;
    for i = 1:min(Kp,K_use)
        cluster_prob(mapping(i), :) = cluster_prob_prev(i, :);
    end
    reinit_cluster_prob = false;
end

if reinit_cluster_prob
    cluster_prob = ones(K_use,3) / 3;
end

%% ================== decide fixed vs free clusters ===========================
% Use fixed_ratio to compute number of fixed clusters (rounded)
num_fixed = max(1, round(fixed_ratio * K_use));
if num_fixed >= K_use
    num_fixed = max(1, K_use - 1);
end
perm = randperm(K_use);
fixed_idx = sort(perm(1:num_fixed));
free_idx = sort(perm(num_fixed+1:end));

op_1 = false(PopSize,1);
op_2 = false(PopSize,1);
op_3 = false(PopSize,1);
probiter = prob(1,:);
l2 = sum(prob(1:2));

cluster_op_report = cell(K_use,1);

%% ======================== Cluster-wise mutation generation ==================
for kc = 1:K_use
    idx = find(labels == kc);
    nc = numel(idx);
    if nc == 0
        cluster_op_report{kc} = 'empty';
        continue;
    end
    r0_cluster = idx(:)';
    [r1c, r2c, r3c] = gnR1R2(nc, size(popAll, 1), r0_cluster);

    % cluster sorted for p-best selection (guard against tiny clusters)
    cluster_fits = fitx(idx);
    [~, localOrder] = sort(cluster_fits);
    localSortedIdx = idx(localOrder);

    % prepare phix_local
    pNP = max(round(0.25 * nc), 1);
    randindex = ceil(rand(1, nc) .* pNP);
    randindex(randindex < 1) = 1;
    % ensure not exceeding available sorted indices
    randindex(randindex > numel(localSortedIdx)) = numel(localSortedIdx);
    phix_local = x(localSortedIdx(randindex), :);

    % prepare phix_local3
    pNP3 = max(round(0.5 * nc), 2);
    randindex3 = ceil(rand(1, nc) .* pNP3);
    randindex3(randindex3 < 1) = 1;
    randindex3(randindex3 > numel(localSortedIdx)) = numel(localSortedIdx);
    phix_local3 = x(localSortedIdx(randindex3), :);

    if ismember(kc, fixed_idx)
        [~, opk] = max(cluster_prob(kc,:));
        cluster_op_report{kc} = sprintf('OP%d', opk);
        if opk == 1
            op_1(idx) = true;
            for ii_local = 1:nc
                gidx = idx(ii_local);
                vi(gidx,:) = x(gidx,:) + F(gidx) * ( phix_local(ii_local,:) - x(gidx,:) + x(r1c(ii_local),:) - popAll(r2c(ii_local),:) );
            end
        elseif opk == 2
            op_2(idx) = true;
            for ii_local = 1:nc
                gidx = idx(ii_local);
                vi(gidx,:) = x(gidx,:) + F(gidx) * ( phix_local(ii_local,:) - x(gidx,:) + x(r1c(ii_local),:) - x(r3c(ii_local),:) );
            end
        else
            op_3(idx) = true;
            for ii_local = 1:nc
                gidx = idx(ii_local);
                vi(gidx,:) = F(gidx) * x(r1c(ii_local),:) + F(gidx) * ( phix_local3(ii_local,:) - x(r3c(ii_local),:) );
            end
        end
    else
        bb_local = rand(nc,1);
        mask1 = bb_local <= probiter(1);
        mask2 = bb_local > probiter(1) & bb_local <= (probiter(1)+probiter(2));
        mask3 = bb_local > (probiter(1)+probiter(2));
        sel = find(mask1);
        if ~isempty(sel)
            for t = 1:numel(sel)
                ii_local = sel(t);
                gidx = idx(ii_local);
                op_1(gidx) = true;
                vi(gidx,:) = x(gidx,:) + F(gidx) * ( phix_local(ii_local,:) - x(gidx,:) + x(r1c(ii_local),:) - popAll(r2c(ii_local),:) );
            end
        end
        sel = find(mask2);
        if ~isempty(sel)
            for t = 1:numel(sel)
                ii_local = sel(t);
                gidx = idx(ii_local);
                op_2(gidx) = true;
                vi(gidx,:) = x(gidx,:) + F(gidx) * ( phix_local(ii_local,:) - x(gidx,:) + x(r1c(ii_local),:) - x(r3c(ii_local),:) );
            end
        end
        sel = find(mask3);
        if ~isempty(sel)
            for t = 1:numel(sel)
                ii_local = sel(t);
                gidx = idx(ii_local);
                op_3(gidx) = true;
                vi(gidx,:) = F(gidx) * x(r1c(ii_local),:) + F(gidx) * ( phix_local3(ii_local,:) - x(r3c(ii_local),:) );
            end
        end
        cnt1 = sum(op_1(idx)); cnt2 = sum(op_2(idx)); cnt3 = sum(op_3(idx));
        [mx, imx] = max([cnt1 cnt2 cnt3]);
        if mx == 0
            cluster_op_report{kc} = sprintf('mixed op1=%d op2=%d op3=%d', cnt1, cnt2, cnt3);
        else
            cluster_op_report{kc} = sprintf('major OP%d (op1=%d op2=%d op3=%d)', imx, cnt1, cnt2, cnt3);
        end
    end
end

%% ========== 输出每代簇选择信息 ==========
if print_each_gen && exist('Printing','var') && Printing==1
    fprintf('Generation %d (progress %.2f) fixed_ratio=%.3f, cluster operator assignments:\n', gg, progress_ratio, fixed_ratio);
    for kc = 1:K_use
        fprintf('  Cluster %2d → %s\n', kc, cluster_op_report{kc});
    end
end

%% handle boundaries
vi = han_boun(vi, xmax, xmin, x,PopSize,2);

%% crossover (use binomial / segment style as original)
if rand<0.4
    mask = rand(PopSize, n) > cr(:, ones(1, n));
    rows = (1 : PopSize)'; cols = floor(rand(PopSize, 1) * n)+1;
    jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
    ui = vi; ui(mask) = x(mask);
else
    ui=x;
    startLoc= randi(n,PopSize,1);
    for i=1:PopSize
        l=startLoc(i);
        while (rand<cr(i) && l<n)
            l=l+1;
        end
        for j=startLoc(i) : l
            ui(i,j)= vi(i,j);
        end
    end
end

%% evaluate
fitx_new = cec20_func(ui',I_fno);
current_eval =current_eval+PopSize;

%% calc. imprv. for Cr and F
diff = abs(fitx - fitx_new);
I =(fitx_new < fitx);
goodCR = cr(I == 1);
goodF = F(I == 1);

%% update archive
archive = updateArchive(archive, x(I == 1, :), fitx(I == 1)');

%% update global operator probs
diff2 = max(0,(fitx - fitx_new))./abs(fitx);
count_S = zeros(1,3);
if any(op_1)
    count_S(1) = max(0, mean(diff2(op_1==1)));
end
if any(op_2)
    count_S(2) = max(0, mean(diff2(op_2==1)));
end
if any(op_3)
    count_S(3) = max(0, mean(diff2(op_3==1)));
end

if any(count_S)
    prob = max(0.1, min(0.9, count_S ./ (sum(count_S) + eps)));
else
    prob = 1/3 * ones(1,3);
end

%% update cluster-level probabilities
for kc = 1:K_use
    idx = find(labels == kc);
    if isempty(idx)
        continue;
    end
    c_counts = zeros(1,3);
    if any(op_1(idx))
        c_counts(1) = max(0, mean(diff2(op_1(idx)==1)));
    end
    if any(op_2(idx))
        c_counts(2) = max(0, mean(diff2(op_2(idx)==1)));
    end
    if any(op_3(idx))
        c_counts(3) = max(0, mean(diff2(op_3(idx)==1)));
    end

    if any(c_counts)
        c_norm = c_counts / (sum(c_counts) + eps);
    else
        c_norm = ones(1,3) / 3;
    end

    cluster_prob(kc, :) = (1 - alpha) * cluster_prob(kc, :) + alpha * c_norm;
    cluster_prob(kc, :) = max(cluster_prob(kc, :), 1e-6);
    cluster_prob(kc, :) = cluster_prob(kc, :) / sum(cluster_prob(kc, :));
end

% persist cluster_prob and centers for next call
cluster_prob_prev = cluster_prob;   % 直接赋值，避免尺寸不匹配时报错
centers_prev = centers;
prev_K = K_use;

%% update population
fitx(I==1)= fitx_new(I==1);
xold(I == 1, :) = x(I == 1, :);
x(I == 1, :) = ui(I == 1, :);

%% update memory for F and CR
if exist('goodF','var') && ~isempty(goodF) && size(goodF,1)==1
    goodF = goodF';
end
if exist('goodCR','var') && ~isempty(goodCR) && size(goodCR,1)==1
    goodCR = goodCR';
end
num_success_params = 0;
if exist('goodCR','var') && ~isempty(goodCR)
    num_success_params = numel(goodCR);
end

if num_success_params > 0
    weightsDE = diff2(I == 1) ./ (sum(diff2(I == 1)) + eps);
    archive_f(hist_pos) = (weightsDE * (goodF .^ 2)) ./ (weightsDE * goodF + eps);
    if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
        archive_Cr(hist_pos)  = -1;
    else
        archive_Cr(hist_pos) = (weightsDE * (goodCR .^ 2)) / (weightsDE * goodCR + eps);
    end
    hist_pos= hist_pos+1;
    if hist_pos > memory_size;  hist_pos = 1; end
else
    archive_Cr(hist_pos)=0.5;
    archive_f(hist_pos)=0.5;
end

[fitx, ind]=sort(fitx);
x=x(ind,:);
xold = xold(ind,:);

if fitx(1)<bestold  && min(x(ind(1),:))>=-100 && max(x(ind(1),:))<=100
    bestold=fitx(1);
    bestx= x(1,:);
end

if exist('Printing','var') && Printing==1
    res_det= [res_det repmat(bestold,1,PopSize)];
end

end

