%% UCI data generation

'generating data using UCI dataset'

%% set parameters

segment_size = 128;       %%Number of entries collected in 2.56 sec for 50hz data entry 
test_user_ids = [2, 4, 9, 10, 12, 13, 18, 20, 24];    %%random users selected for testing
train_user_ids = [];      %% Used to store which user did the activity in ith row of training data

strcat('segment lenght: ', num2str(segment_size))

%% open file and read labels

'loading labels'

fid = fopen('datasets/uci_raw_data/labels.txt');      %%opening labels file
A = textscan(fid, '%d%d%d%d%d', 'delimiter', ' ');    %%5 integers in a row separated by space are gathered in A
fclose(fid);                                          %%closing labels file

%% extract data about experiments

%%

exp_id = A{1};        %% 1st integer is the experiment id
usr_id = A{2};        %% 2nd integer is the user id telling which user does this activity
act_id = A{3};        %% 3rd integer tells whih activity is this
act_be = A{4};        %% starting row number of the activity
act_en = A{5};        %% ending row number of the activity



idx_files = [exp_id, usr_id];  %% making a list of all the experiment ids and user ids.
                                 
idx_files = unique(idx_files, 'rows');    %% ensuring that no two rows have same values

%% form the dataset

'generating new dataset'

% training data

x = []; gyro_x = [];    %% x,y,z has accelerometer values
y = []; gyro_y = [];    %% gyro_x,gyro_y,gyro_z has gyroscope values
z = []; gyro_z = [];
answers_raw = [];       %% the corresponding answer activity labels for the input row
features = [];          %% the statistical features extracted for every row (40 for each row of data)

% testing data

test_x = []; test_gyro_x = [];
test_y = []; test_gyro_y = [];
test_z = []; test_gyro_z = [];
test_answers_raw = [];
test_features = [];

for i = 1:size(idx_files, 1)          %% for all the rows in idx_files
    
    nexp = num2str(idx_files(i, 1));  %% extracting the experiment number from ith row of idx_file
    if idx_files(i, 1) < 10
       nexp = strcat('0', nexp);      %% appending 0 at the beginning if nexp is less than 10
    end
    
    nusr = num2str(idx_files(i, 2));  %% extracting the user id number from ith row of idx_file
    if idx_files(i, 2) < 10
       nusr = strcat('0', nusr);      %% appending 0 at the beginning if nexp is less than 10
    end
    
    
    %% opening the accelerometer data file with experiment no. nexp and user id no. nusr as extracted above.
    fid = fopen(strcat('datasets/uci_raw_data/acc_exp', nexp, '_user', nusr, '.txt'));  
    acc_data = textscan(fid, '%f%f%f', 'delimiter', ' ');
    fclose(fid);
    
    
    %% opening the gyroscope data file with experiment no. nexp and user id no. nusr as extracted above.
    fid = fopen(strcat('datasets/uci_raw_data/gyro_exp', nexp, '_user', nusr, '.txt'));
    gyro_data = textscan(fid, '%f%f%f', 'delimiter', ' ');
    fclose(fid);

    %% every raw file contains 3 floating values separated by space in each row.
    %% Now extracting the values for x,y,z axis respectively.
    
    data_x = acc_data{1};   %% x axis values of accelerometer 
    data_y = acc_data{2};   %% y axis values of accelerometer
    data_z = acc_data{3};   %% z axis values of accelerometer

    gdata_x = gyro_data{1}; %% x axis values of gyroscope
    gdata_y = gyro_data{2}; %% y axis values of gyroscope
    gdata_z = gyro_data{3}; %% y axis values of gyroscope
   
    exp_idx = find(exp_id == idx_files(i, 1));    %% Getting the indices that correspond to experiment no. in the A extracted above
    exp_data = [act_id(exp_idx), act_be(exp_idx), act_en(exp_idx)]; %% Storing the activity id, activity start row and activity end row for different users

    if length(find(test_user_ids == idx_files(i, 2))) == 0    %% checking that test users are not for train data
        
        for j = 1:size(exp_data, 1)   %% for all rows in exp_data
            
            if exp_data(j, 1) < 7     %% Taking activities that are 1-6. 7-12 labelled activities are not taken.
                
                k = exp_data(j, 2);   %% extracting the starting row of the activity by the particular user
                while k + segment_size <= exp_data(j, 3)    %% extracting segments of 128 till activity end row is reached
                   train_user_ids = [train_user_ids; idx_files(i,2)]; %% Storing which user did this activity
                   x_add = data_x(k : k + segment_size - 1)'; %% extracting 128 size segment of accelerometer data
                   y_add = data_y(k : k + segment_size - 1)';
                   z_add = data_z(k : k + segment_size - 1)';

                   x = [x; x_add];  %% apppending to the accelerometer x axis data
                   y = [y; y_add];
                   z = [z; z_add];

                   features = [features; Extract_basic_features(x_add, y_add, z_add)];  %% extracting features for every segment of every training example using only accelerometer data

                   gyro_x = [gyro_x; gdata_x(k : k + segment_size - 1)'];
                   gyro_y = [gyro_y; gdata_y(k : k + segment_size - 1)'];
                   gyro_z = [gyro_z; gdata_z(k : k + segment_size - 1)'];

                   answers_raw = [answers_raw; exp_data(j, 1)];   %% storing the activit/answer label for jth training example
                   k = k + segment_size/2;    %% incrementing k by 64 instead of 128 as the activities are not mutually exclusive
                   
                end
            end
            
	    end
    else
        
        for j = 1:size(exp_data, 1)
            
            if exp_data(j, 1) < 7
                k = exp_data(j, 2);
                while k + segment_size <= exp_data(j, 3)

                   x_add = data_x(k : k + segment_size - 1)';
                   y_add = data_y(k : k + segment_size - 1)';
                   z_add = data_z(k : k + segment_size - 1)';

                   test_x = [test_x; x_add];
                   test_y = [test_y; y_add];
                   test_z = [test_z; z_add];

                   test_features = [test_features; Extract_basic_features(x_add, y_add, z_add)];

                   test_gyro_x = [test_gyro_x; gdata_x(k : k + segment_size - 1)'];
                   test_gyro_y = [test_gyro_y; gdata_y(k : k + segment_size - 1)'];
                   test_gyro_z = [test_gyro_z; gdata_z(k : k + segment_size - 1)'];

                   test_answers_raw = [test_answers_raw; exp_data(j, 1)];
                   k = k + segment_size/2;
                end
            end
            
        end
    end
end

'data was generated'

%% transform answers into vectors

answer_vector = [];

for i = 1 : length(answers_raw)
    vect = zeros(6, 1)';
    vect(answers_raw(i)) = 1;
    answer_vector = [answer_vector; vect];    %% making one-hot embedding where the column number correspong
                                              %% to the activity label is 1 else all are 0
end

test_answ_vector = [];

for i = 1 : length(test_answers_raw)
    vect = zeros(6, 1)';
    vect(test_answers_raw(i)) = 1;
    test_answ_vector = [test_answ_vector; vect];
end


%% write data to file

'writing data to file'

all_data = [x, y, z, gyro_x, gyro_y, gyro_z];   %% all data is appended of accelerometer and gyroscope


%% writing everything to files with appropriate names
dlmwrite('uci_data/all_data.csv', all_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/answers.csv', answer_vector, 'delimiter', ',')
dlmwrite('uci_data/train_user_ids.csv', train_user_ids, 'delimiter', ',')
dlmwrite('uci_data/features.csv', features, 'delimiter', ',')

all_test_data = [test_x, test_y, test_z, test_gyro_x, test_gyro_y, test_gyro_z];

dlmwrite('uci_data/all_data_test.csv', all_test_data, 'delimiter', ',', 'precision', 4)
dlmwrite('uci_data/answers_test.csv', test_answ_vector, 'delimiter', ',')
dlmwrite('uci_data/test_features.csv', test_features, 'delimiter', ',')

'training and test data was generated'