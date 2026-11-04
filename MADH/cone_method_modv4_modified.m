function I = cone_method_modv4_modified(Xtrue, s, K, half_reaction, partition, Ibd)
% cone_method_modv4_modified selects indices I from Xtrue based on the cone method.
% This modified version:
% 1. Produces a final I of length K + length(Ibd).
% 2. Splits the candidate indices based on s into two groups (s>0 and s<0)
%    and samples additional indices from each group.
%
% Jeongmin Chae and Stephen Quiton, University of Southern California, 2022
% Modified to enforce an even split between positive and negative s values.

    %% Input check and initial settings
    if partition && half_reaction
        me = MException('Partition mode not available for half reactions');
        throw(me)
    end

    if half_reaction
        init_columns = 2;
    else
        init_columns = 3 + length(Ibd);
    end

    %% Normalize Xtrue (first to [0,1] then to unit norm)
    for j = 1:size(Xtrue,2)
        Xtrue(:,j) = (Xtrue(:,j) - min(Xtrue(:,j))) / max(Xtrue(:,j));
    end
    for j = 1:size(Xtrue,2)
        Xtrue(:,j) = Xtrue(:,j) / sqrt(sum(Xtrue(:,j).^2));
    end

    %% Build S using s as the base vector
    xlist = s;
    S = zeros(init_columns, size(Xtrue,2));
    for i = 1:size(S,1)
        for j = 1:size(S,2)
            S(i,j) = xlist(j)^(size(S,1)-i);
        end
    end

    ts_index = find(s == 0);
    bd_columns = Xtrue(:, Ibd);

    % (Force half_reaction false here if it isn’t fully supported)
    half_reaction = false;

    %% Fit X using selected initial columns
    if half_reaction
        % Half reaction branch (kept for completeness)
        X = zeros(size(Xtrue,1), init_columns);
        X(:,1) = Xtrue(:,1);
        X(:,end) = Xtrue(:,end);
        Sfit = S(:, [1, end]);
        Q = X * Sfit'\(Sfit * Sfit');
        Xfithat = Q * S;
        Xfithat(:,1) = Xtrue(:,1);
        Xfithat(:,end) = Xtrue(:,end);
    else
        % Non-half reaction branch: select initial columns including boundaries and Ibd.
        X = zeros(size(Xtrue,1), init_columns);
        H = sort(unique([1, ts_index, size(Xtrue,2), Ibd]));
        for h = 1:length(H)
            index = H(h);
            X(:,h) = Xtrue(:, index);
        end
        Sfit = S(:, union([1, ts_index, size(Xtrue,2)], Ibd));
        Q = X * Sfit' / (Sfit * Sfit');
        Xfithat = Q * S;
        % Force key columns to be exact
        Xfithat(:,1)         = Xtrue(:,1);
        Xfithat(:,end)       = Xtrue(:,end);
        if ~isempty(ts_index)
            Xfithat(:, ts_index) = Xtrue(:, ts_index);
        end
    end

    %% Normalize Xfithat columns: [0,1] then unit norm.
    for i = 1:size(Xtrue,2)
        Xfithat(:,i) = (Xfithat(:,i) - min(Xfithat(:,i))) / max(Xfithat(:,i) - min(Xfithat(:,i)));
    end
    for i = 1:size(Xtrue,2)
        Xfithat(:,i) = Xfithat(:,i) / norm(Xfithat(:,i));
    end

    Xinit = Xfithat;
    Xori = Xinit;  % Xori holds the fitted, normalized data

    %% Initial centroid selection: include boundaries and Ibd
    if ~half_reaction
        I_initial = sort(unique([1, ts_index, size(Xtrue,2), Ibd]));
    else
        I_initial = sort(unique([1, size(Xtrue,2)]));
    end
    disp('Initial selected indices (I):');
    disp(I_initial);

    % Remove the already-selected initial columns from Xinit
    if ~half_reaction
        Xinit(:, I_initial) = [];
    else
        Xinit(:, [1, end]) = [];
    end

    % Create a companion vector for remaining indices (refer to original Xori)
    remainIdx = setdiff(1:size(Xori,2), I_initial, 'stable');

    %% Partition remainIdx into positive and negative groups (ignore s == 0)
    remainPos = remainIdx( s(remainIdx) > 0 );
    remainNeg = remainIdx( s(remainIdx) < 0 );
    
    % Also, split the already selected indices into positive and negative groups.
    Ipos = I_initial( s(I_initial) > 0 );
    Ineg = I_initial( s(I_initial) < 0 );
    % (Indices with s==0 are left in I_initial and will not affect the split.)

    %% Determine target length and how many indices to add from each segment
    targetLength = K + length(Ibd);
    nAddTotal = targetLength - length(I_initial);
    
    % Aim for balanced additional sampling
    nAddPos = ceil(nAddTotal/2);
    nAddNeg = nAddTotal - nAddPos;
    
    %% Define a helper function for computing cone cost for a candidate.
    % Given a candidate index idx and current centroid indices centIdx, cost is defined
    % as the maximum dot product between Xori(:, idx) and each centroid.
    coneCost = @(idx, centIdx) max( Xori(:, idx)' * Xori(:, centIdx) );
    
    %% Add new indices from the positive group
    for i = 1:nAddPos
        if isempty(remainPos)
            break;  % No more positive candidates
        end
        % Compute cost for each candidate in remainPos with respect to current Ipos.
        if isempty(Ipos)
            % If no positive centroids yet, select the candidate with smallest norm (or first one)
            [~, bestIdxRel] = min(cellfun(@(x) norm(Xori(:, x)), num2cell(remainPos)));
        else
            costs = zeros(length(remainPos),1);
            for j = 1:length(remainPos)
                costs(j) = coneCost(remainPos(j), Ipos);
            end
            [~, bestIdxRel] = min(costs);
        end
        chosenIdx = remainPos(bestIdxRel);
        % Append to Ipos
        Ipos = sort([Ipos, chosenIdx]);
        % Remove the chosen index from remainPos
        remainPos(bestIdxRel) = [];
        
        disp(['Positive group selected index: ' num2str(chosenIdx) ', s = ' num2str(s(chosenIdx))]);
    end

    %% Add new indices from the negative group
    for i = 1:nAddNeg
        if isempty(remainNeg)
            break;  % No more negative candidates available
        end
        if isempty(Ineg)
            [~, bestIdxRel] = min(cellfun(@(x) norm(Xori(:, x)), num2cell(remainNeg)));
        else
            costs = zeros(length(remainNeg),1);
            for j = 1:length(remainNeg)
                costs(j) = coneCost(remainNeg(j), Ineg);
            end
            [~, bestIdxRel] = min(costs);
        end
        chosenIdx = remainNeg(bestIdxRel);
        Ineg = sort([Ineg, chosenIdx]);
        remainNeg(bestIdxRel) = [];
        
        disp(['Negative group selected index: ' num2str(chosenIdx) ', s = ' num2str(s(chosenIdx))]);
    end

    %% Combine indices: initially selected plus added from each group.
    I_temp = sort(unique([I_initial, Ipos, Ineg]));

    %% If the desired target length has not been reached, choose additional candidates
    % from the remaining pool (from either group) based on the overall cone cost.
    while length(I_temp) < targetLength && ~isempty(remainIdx)
        % Recalculate overall remaining indices (from both groups)
        remainAll = setdiff(remainIdx, I_temp, 'stable');
        if isempty(remainAll)
            break;
        end
        Z = Xori(:, I_temp);
        % Compute cost for each candidate from remainAll relative to all current centroids Z.
        costs = zeros(length(remainAll), 1);
        for j = 1:length(remainAll)
            costs(j) = max( Z' * Xori(:, remainAll(j)) );
        end
        [~, idxBestRel] = min(costs);
        candidate = remainAll(idxBestRel);
        I_temp = sort(unique([I_temp, candidate]));
        % Also remove from remainIdx (if needed)
        remainIdx(remainIdx == candidate) = [];
        disp(['Additional selection chosen: ' num2str(candidate) ', s = ' num2str(s(candidate))]);
    end

    % Final output: ensure I is exactly targetLength elements.
    if length(I_temp) > targetLength
        % Optionally trim extra indices (here we simply take the first targetLength elements).
        I = I_temp(1:targetLength);
    else
        I = I_temp;
    end
end
