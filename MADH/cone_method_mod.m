function I = cone_method_mod(Xtrue,s, K, half_reaction, partition, Ibd)
%Jeongmin Chae and Stephen Quiton, University of Southern California, 2022

if partition && half_reaction
    me = MException('Partition mode not available for half reactions');
    throw(me)
end

if half_reaction
    init_columns = 2;
else
    init_columns = 3+length(Ibd);
end



% Normalize Xtrue 
for j=1:size(Xtrue,2)
    Xtrue(:,j)=(Xtrue(:,j)-min((Xtrue(:,j))))/max((Xtrue(:,j)));
end

for j=1:size(Xtrue,2)
    Xtrue(:,j)=Xtrue(:,j)/sqrt(sum(Xtrue(:,j).^2));
end

% Preprocessing -- X=QS



%xlist=[0:size(Xtrue,2)-1];
xlist= s;

S=zeros(init_columns,size(Xtrue,2));

% Get initial S
for i=1:size(S,1)
    for j=1:size(S,2)
        S(i,j)=xlist(j).^(size(S,1)-i);
    end
end


ts_index = find(s==0);
bd_columns=Xtrue(:, Ibd);

half_reaction=false;

if half_reaction
    % Sample initial columns
    X=zeros(size(Xtrue,1),init_columns);
    X(:,1)=Xtrue(:,1);
    X(:,end)=Xtrue(:,end);
    
    
    % Get initial Q
    Sfit = S(:,[1, end]); 
    Q = X*Sfit'*inv(Sfit*Sfit');
    Xfithat=Q*S; %Initial
    Xfithat(:,1)=Xtrue(:,1);
    Xfithat(:,end)=Xtrue(:,end);
else
    % Sample initial columns
    X=zeros(size(Xtrue,1),init_columns);
    H=sort([1 ts_index size(Xtrue,2) Ibd]);
    for h=1:size(H,2)
        index=H(h);
        X(:,h)=Xtrue(:,index);
    end


    % Get initial Q
    
    Sfit = S(:, union([1, ts_index, size(Xtrue, 2)], Ibd)); 
    Q = X*Sfit'*inv(Sfit*Sfit');
    Xfithat=Q*S; %Initial
    Xfithat(:,1)=Xtrue(:,1);
    Xfithat(:,end)=Xtrue(:,end);
    Xfithat(:,ts_index)=Xtrue(:,ts_index);

end

for i=1:size(Xtrue,2)

    Xfithat(:,i)=(Xfithat(:,i)-min((Xfithat(:,i))))/max((Xfithat(:,i)-min((Xfithat(:,i)))));

    %max((Xfithat(:,i)-min((Xfithat(:,i)))))

end

% Normalize each column
for i=1:size(Xtrue,2)
     Xfithat(:,i)= Xfithat(:,i)./norm(Xfithat(:,i));
end


Xinit=Xfithat;
Xori=Xinit;

% Initialization

z=Xinit(:,1); % Select the first centriod randomly
z2=Xinit(:,size(Xtrue,2)); 

Z=[z z2]; % Set of centroid selected


if ~half_reaction
    z3=Xinit(:,ts_index); 
    Z=[Z z3];
    Xinit(:,[1,ts_index,end])=[];
else
    Xinit(:,[1,end])=[];
end
Xtemp=Xinit;
%Add the initial columns to I
I=[];
for m=1:size(Z,2)
    indx=find(sum(Xori-Z(:,m))==0);
    I = [I indx];
end

sample_mode = 'all';

for k=1:K-length(Ibd)
    MAX=zeros(size(Xtemp,2),k+1);
    for j=1:size(Xtemp,2)

        for i=1:k+1
            MAX(j,i)=Z(:,i)'*Xtemp(:,j);
        end
        
        
    end
    
    for m=1:size(MAX,1)
        M_MAX(m)=max(MAX(m,:));
    end
    
    % What?
    if k>1
        M_MAX(end)=[];
    end

    switch sample_mode
        case 'all'
            [~,ind]=min(M_MAX);
            ztemp=Xtemp(:,ind);
            indx=find(sum(Xori-ztemp)==0);
        
    end

    Z=[Z ztemp];
    Xtemp(:,ind)=[];
    Xtempold=Xtemp;
    Xtemp=Xtempold;
    

    % Determine the next sample region. Select side with fewer columns
    % sampled

    I=[I indx];

end
I=union(I, Ibd);