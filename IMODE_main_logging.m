%% ============ IMODE Main with Logging ============
function [outcome,com_time,SR,avgFE,res_det,bestx, CaseStudyData]= IMODE_main_logging(run,I_fno)

Par= Introd_Par(I_fno);
iter=0;             %% current generation

%% =================== Define a random seed ===============================
RandStream.setGlobalStream (RandStream('mt19937ar','seed',run));

%% define variables
current_eval=0;             %% current fitness evaluations
PS1=Par.PopSize;            %% define PS1
PS2=100;
InitPop1=PS2;

%% ====================== Initalize x ==================================
x=repmat(Par.xmin,Par.PopSize,1)+repmat((Par.xmax-Par.xmin),Par.PopSize,1).*rand(Par.PopSize,Par.n);

%% calc. fit. and update FES
tic;
fitx= cec20_func(x',I_fno);
current_eval =current_eval+Par.PopSize;
res_det= min(repmat(min(fitx),1,Par.PopSize), fitx); 

%% ====================== store the best ==================
[bestold, bes_l]=min(fitx);     bestx= x(bes_l,:);

%% ================== fill in for each Algorithm ===================================
%% IMODE
EA_1= x(1:PS1,:);    EA_obj1= fitx(1:PS1);   EA_1old = x(randperm(PS1),:);

%% CMA-ES
EA_2= x(PS1+1:size(x,1),:);    EA_obj2= fitx(PS1+1:size(x,1));

%% ================ define CMA-ES parameters ==============================
setting=[];bnd =[]; fitness = [];
[setting]= init_cma_par(setting,EA_2, Par.n, PS2);

%% ===== prob. of each DE operator
probDE1=1./Par.n_opr .* ones(1,Par.n_opr);

%% ===================== archive data ====================================
arch_rate=2.6;
archive.NP = arch_rate * PS1; 
archive.pop = zeros(0, Par.n); 
archive.funvalues = zeros(0, 1); 

%% ==================== to adapt CR and F =================================
hist_pos=1;
memory_size=20*Par.n;
archive_f= ones(1,memory_size).*0.2;
archive_Cr= ones(1,memory_size).*0.2;
archive_T = ones(1,memory_size).*0.1;
archive_freq = ones(1, memory_size).*0.5;

%% Loop Controls
stop_con=0; avgFE=Par.Max_FES; InitPop=PS1; thrshold=1e-08;
cy=0;indx = 0; Probs=ones(1,2);
F = normrnd(0.5,0.15,1,PS1);
cr= normrnd(0.5,0.15,1,PS1);

%% ================== Logging Initialization ==================
CaseStudyData.Gen = [];
CaseStudyData.FES = [];
CaseStudyData.Pop = {};      % Record Population positions
CaseStudyData.Fit = {};      % Record Fitness values
CaseStudyData.Probs = {};    % Record Global Probabilities
CaseStudyData.Labels = {}; % <--- 新增：存储标签
record_counter = 1;
log_interval = 20; % 每20代记录一次 (可调整)

%% main loop
while stop_con==0
    
    iter=iter+1;
    cy=cy+1; 
    
    Probs = [1 0]; 

    %% ====================== Applying IMODE ============================
    if (current_eval<Par.Max_FES)
        if rand<=Probs(1)
            
            %% Linear Reduction of PS1
            UpdPopSize = round((((Par.MinPopSize - InitPop) / Par.Max_FES) * current_eval) + InitPop);
            if PS1 > UpdPopSize
                reduction_ind_num = PS1 - UpdPopSize;
                if PS1 - reduction_ind_num <  Par.MinPopSize
                    reduction_ind_num = PS1 - Par.MinPopSize;
                end
                %% remove the worst ind.
                for r = 1 : reduction_ind_num
                    vv=PS1;
                    EA_1(vv,:)=[];
                    EA_1old(vv,:)=[];
                    EA_obj1(vv)=[];
                    PS1 = PS1 - 1;
                end
                archive.NP = round(arch_rate * PS1);
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1 : archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
            
            %% apply IMODE
            % 在 IMODE_main_logging.m 内部，找到这一行并确保是这样的：
[EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T, archive_freq, current_eval, res_det, F, cr, ~, cluster_probs_out,labels_out] = ...
    IMODE2( EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T,....
    archive_freq, Par.xmin, Par.xmax,  Par.n,  PS1,  current_eval, I_fno, res_det, Par.Printing, Par.Max_FES, Par.Gmax, iter, F, cr);
        end
    end
    
    %% ====================== CMA-ES (Skipped Logic) ======================
    if (current_eval<Par.Max_FES)
        if   rand<Probs(2)
            UpdPopSize = round((((Par.MinPopSize1 - InitPop1) / Par.Max_FES) * current_eval) + InitPop1);
            if PS2 > UpdPopSize
                reduction_ind_num = PS2 - UpdPopSize;
                if PS2 - reduction_ind_num <  Par.MinPopSize
                    reduction_ind_num = PS2 - Par.MinPopSize;
                end
                for r = 1 : reduction_ind_num
                    vv=PS2;
                    EA_2(vv,:)=[];
                    EA_obj2(vv)=[];
                    PS2 = PS2 - 1;
                end
            end
            [ EA_2, EA_obj2, setting,bestold,bestx,bnd,fitness,current_eval,res_det] = ...
                Scout( EA_2, EA_obj2, probSC, setting, iter,bestold,bestx,fitness,bnd,...
                Par.xmin,Par.xmax,Par.n,PS2,current_eval,I_fno,res_det,Par.Printing,Par.Max_FES);
        end
    end
    
    %% ====================== LS2 (Local Search) ======================
    if current_eval>0.85*Par.Max_FES && current_eval<Par.Max_FES
        if rand<Par.prob_ls
            old_fit_eva=current_eval;
            [bestx,bestold,current_eval,succ] = LS2 (bestx,bestold,Par,current_eval,I_fno,Par.Max_FES,Par.xmin,Par.xmax);
            if succ==1 
                EA_1(PS1,:)=bestx';
                EA_obj1(PS1)=bestold;
                [EA_obj1, sort_indx]=sort(EA_obj1);
                EA_1= EA_1(sort_indx,:);
                
                EA_2=repmat(EA_1(1,:), PS2, 1);
                [setting]= init_cma_par(setting,EA_2, Par.n, PS2);
                setting.sigma=1e-05;
                EA_obj2(1:PS2)= EA_obj1(1);
                Par.prob_ls=0.1;
            else
                Par.prob_ls=0.01; 
            end
            if Par.Printing==1
                res_det= [res_det repmat(bestold,1,(current_eval-old_fit_eva))];
            end
        end
    end

    %% ================== Data Logging Point ==================
    % 记录逻辑：每 log_interval 代 或者 结束时记录
    if mod(iter, log_interval) == 0 || current_eval >= Par.Max_FES - 4*UpdPopSize
        CaseStudyData.Gen(record_counter) = iter;
        CaseStudyData.FES(record_counter) = current_eval;
        CaseStudyData.Pop{record_counter} = EA_1;      % 记录当前种群位置
        CaseStudyData.Fit{record_counter} = EA_obj1;   % 记录当前适应度
        CaseStudyData.Probs{record_counter} = probDE1; % 记录算子概率
        CaseStudyData.ClusterProbs{record_counter} = cluster_probs_out; % 保存它
        CaseStudyData.Labels{record_counter} = labels_out; % <--- 关键！保存它！
        record_counter = record_counter + 1;
    end
    %% ========================================================
    
    %% ====================== stopping criterion check ====================
    if (current_eval>=Par.Max_FES-4*UpdPopSize)
        stop_con=1;
        avgFE=current_eval;
    end
    if ( (abs (Par.f_optimal - bestold)<= thrshold))
        stop_con=1;
        bestold=Par.f_optimal;
        avgFE=current_eval;
    end
    
    %% =============================== Final Print ==============================
    if stop_con
        com_time= toc;
        fprintf('Logging Run: Fitness\t %d, avg.FFE\t %d\n', abs(Par.f_optimal-bestold), avgFE);
        outcome= abs(Par.f_optimal-bestold);
        SR= (outcome==0);
    end
end
end