load('lbp_WDRef.mat')
load('id_WDRef.mat')
labels = id_WDRef;
X = double(lbp_WDRef);
X = sqrt(X);
X = bsxfun(@rdivide,X,sum(X,2));
train_mean = mean(X, 1);
[COEFF,SCORE] = princomp(X,'econ');
train_x = SCORE(:,1:2000);

[mappedX, mapping] = JointBayesian(train_x, labels);
% Dis_matrix = repmat(mapping.c,1,size(train_x,1))+repmat(mapping.c,1,size(train_x,1))+train_x * mapping.G *train_x';
[classes, bar, labels] = unique(labels);
    nc = length(classes);
train_Intra = zeros(nc*2,2);
for i=1:nc
    train_Intra(2*i-1,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
    train_Intra(2*i,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
end;
train_Extra = reshape(randperm(length(labels),20000),10000,2);
train_Extra(labels(train_Extra(:,1))==labels(train_Extra(:,2)),:)=[];
train_Extra(size(train_Intra,1)+1:end,:)=[];
Dis_train_Intra = zeros(size(train_Intra,1),1);
Dis_train_Extra = zeros(size(train_Intra,1),1);

for i=1:size(train_Intra,1)
    Dis_train_Intra(i) = train_x(train_Intra(i,1),:) * mapping.A * train_x(train_Intra(i,1),:)' + train_x(train_Intra(i,2),:) * mapping.A * train_x(train_Intra(i,2),:)' - 2 * train_x(train_Intra(i,1),:) * mapping.G * train_x(train_Intra(i,2),:)';
    Dis_train_Extra(i) = train_x(train_Extra(i,1),:) * mapping.A * train_x(train_Extra(i,1),:)' + train_x(train_Extra(i,2),:) * mapping.A * train_x(train_Extra(i,2),:)' - 2 * train_x(train_Extra(i,1),:) * mapping.G * train_x(train_Extra(i,2),:)';
end;
group_train = [ones(size(Dis_train_Intra,1),1);zeros(size(Dis_train_Extra,1),1)];
training = [Dis_train_Intra;Dis_train_Extra];

load('lbp_lfw.mat')
load('pairlist_lfw.mat')
normX = double(lbp_lfw);
normX = sqrt(normX);
normX = bsxfun(@rdivide,normX,sum(normX,2));
%normX = bsxfun(@minus,normX,train_mean);
normX = normX * COEFF(:,1:2000);
test_Intra = pairlist_lfw.IntraPersonPair;
test_Extra = pairlist_lfw.ExtraPersonPair;

result_Intra = zeros(3000,1);
result_Extra = zeros(3000,1);
for i=1:3000
    result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
    result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
end;

group_sample = [ones(3000,1);zeros(3000,1)];
sample = [result_Intra;result_Extra];



%%% the method of classification     
[m,n]=size(group_train);
starderd=0;
eg1=0;
num1=0;
eg2=0;
num2=0;
for i=1:m
     if group_train(i,1)==0
        eg2=eg2+training(i,1);
        num2=num2+1;
     else
         eg1=eg1+training(i,1);
         num1=num1+1;
     end
end
starderd=(eg1+eg2)/(num1+num2);
 [m,n]=size(sample);
 label=zeros(m,1);
 accuracy=0;
 for i=1:m
     if sample(i,1)>starderd
        label(i,1)=1;
     else
        label(i,1)=0;
     end
     if label(i,1)==group_sample(i,1)
         accuracy=accuracy+1;
     end
 end
 accuracy/m
 %result(1,1)=result(1,1)+accuracy/m;


%method SVM
bestc=256;
% bestg=128;
% cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
%cmd = [' -t 0 -h 0'];
%group_train=int(group_train);
%group_sample=int(group_sample);
% trainXC_mean = mean(training); 
% trainXC_sd = sqrt(var(training)+0.01); 
% training1 = bsxfun(@rdivide, bsxfun(@minus, training, trainXC_mean), trainXC_sd); 
% sample1 = bsxfun(@rdivide, bsxfun(@minus, sample, trainXC_mean), trainXC_sd);

model = svmtrain(group_train,training,'-t 0 -h 0');
%[class,accTotal] = svmpredict(group_sample,sample,model);
[m,n]=size(sample);
predict_label=zeros(m,1);
for i=1:m
    %len=model.totalSV;
    %value=0;
    value=sum(model.sv_coef.*model.SVs.*sample(i,1));
%     for j=1:len
%         value=value+(model.sv_coef(j,1))*(model.SVs(j,1)*sample(i,1));
%     end
    value=value-model.rho;
    if value>0
        predict_label(i,1)=1;
    else
        predict_label(i,1)=0;
    end
end
sum(predict_label==group_sample)/size(group_sample,1)
%result(2,1)=result(2,1)+sum(predict_label==group_sample)/size(group_sample,1);
