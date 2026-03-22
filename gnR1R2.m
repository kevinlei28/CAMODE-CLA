function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
%GN R1R2  逐个体生成三组互异的随机索引列向量
%   NP1 — 主种群大小
%   NP2 — 全体大小（主种群+Archive）
%   r0  — 基准索引列向量

NP0 = numel(r0);
r1 = zeros(NP0,1);
r2 = zeros(NP0,1);
r3 = zeros(NP0,1);

for i = 1:NP0
    %—— 生成 r1(i)
    cand1 = setdiff(1:NP1, r0(i));
    if isempty(cand1)
        cand1 = 1:NP1;      % 退化：直接用全部
    end
    r1(i) = cand1( randi(numel(cand1)) );

    %—— 生成 r2(i)
    cand2 = setdiff(1:NP2, [r0(i), r1(i)]);
    if isempty(cand2)
        cand2 = 1:NP2;
    end
    r2(i) = cand2( randi(numel(cand2)) );

    %—— 生成 r3(i)
    cand3 = setdiff(1:NP1, [r0(i), r1(i), r2(i)]);
    if isempty(cand3)
        cand3 = 1:NP1;
    end
    r3(i) = cand3( randi(numel(cand3)) );
end
end
